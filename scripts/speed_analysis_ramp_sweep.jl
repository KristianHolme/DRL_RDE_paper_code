#=
speed_analysis_ramp_sweep.jl

CLI examples

- Full run (default settings; writes results + progress DB):

  julia -t auto --project=. scripts/speed_analysis_ramp_sweep.jl

- Quick debug run (fewer starts, shorter segments, smaller analysis window):

  julia -t auto --project=. scripts/speed_analysis_ramp_sweep.jl \
    --limit 5 \
    --hold-time 30 \
    --window 10

- Named run (custom output and progress DB basename):

  julia -t auto --project=. scripts/speed_analysis_ramp_sweep.jl --name baseline_v2

Notes
- Progress DB path is printed as: `Inspect progress with: mpm <db_path>`
- MultiProgressManagers updates are per ramp segment: once per u_start, `total_steps`
  is set from the ramp length; before each segment rollout, `update!` sets the message
  (phase and u_p); after each segment, `update!` advances the step only. There is no
  per-env-step progress.
- A single combined table is written to `datadir("speed_analysis_ramp")` as `ramp_results*.jld2` (DataFrame only). Use `--name` to change the stem.
- `u_start` grid spacing is `--starting-step` / `-S` (default `0.05`).
=#

using DrWatson
@quickactivate :DRL_RDE
using Accessors
using ArgParse
using Logging
using ProgressMeter
using Base.Threads
using Statistics
using DataFrames
using JLD2
using Dates
using MultiProgressManagers: ProgressManager, ProgressTask, get_task, update!, finish!, fail!
using CairoMakie

function parse_rampsweep_cli()
    s = ArgParseSettings(;
        description = "Speed analysis ramp sweep over u_start; use --limit / --hold-time for faster debug runs.",
    )
    @add_arg_table! s begin
        "--u-p-bottom", "-b"
        help = "Lower bound of u_p in the ramp cycle"
        arg_type = Float64
        default = 0.4
        "--u-p-top", "-u"
        help = "Upper bound of u_p in the ramp cycle"
        arg_type = Float64
        default = 1.03
        "--u-p-step", "-s"
        help = "u_p increment along each ramp leg"
        arg_type = Float64
        default = 0.001
        "--starting-step", "-S"
        help = "Spacing between u_start values in the sweep (grid along injection pressure)"
        arg_type = Float64
        default = 0.05
        "--limit", "-n"
        help = "Use only the first N u_start values (0 = full sweep)"
        arg_type = Int
        default = 0
        "--hold-time", "-H"
        help = "Simulated time per ramp segment (smaller => fewer steps per rollout, faster debug)"
        arg_type = Float64
        default = 300.0
        "--window", "-w"
        help = "Time units at end of each segment for stability/speed stats (0 = auto from hold-time)"
        arg_type = Int
        default = 0
        "--name"
        help = "Run label; used in the output .jld2 filename and progress DB name (empty = default names)"
        arg_type = String
        default = ""
        "--N", "-N"
        help = "Number of grid points per dimension"
        arg_type = Int
        default = 2048
    end
    return parse_args(s)
end

"""Strip and sanitize a user-supplied run name for use in file names."""
function sanitized_run_name(s::AbstractString)
    t = strip(string(s))
    if isempty(t)
        return ""
    end
    t = replace(t, r"[\\/]+" => "_")
    t = replace(t, r"\s+" => "-")
    t = replace(t, r"[^a-zA-Z0-9._-]+" => "_")
    return t
end

##
# Ramp parameters: bottom/top u_p, small steps, long hold per step; multiple cycles from different starting values
const dt = 10.0f0
const reward_threshold = 0.98f0

cli = parse_rampsweep_cli()
const u_p_bottom = Float32(cli["u-p-bottom"])
const u_p_top = Float32(cli["u-p-top"])
const u_p_step = Float32(cli["u-p-step"])
if !(u_p_step > 0)
    error("--u-p-step / -s must be positive, got ", u_p_step)
end
if !(u_p_bottom < u_p_top)
    error("Need u_p_bottom < u_p_top (got ", u_p_bottom, ", ", u_p_top, ")")
end
if (u_p_top - u_p_step) <= u_p_bottom
    error("u_p range too small for (u_p_top - u_p_step) > u_p_bottom with given step")
end
const starting_step = Float32(cli["starting-step"])
if !(starting_step > 0)
    error("--starting-step / -S must be positive, got ", starting_step)
end
const hold_time = Float32(cli["hold-time"])
if !(hold_time > 0)
    error("--hold-time / -H must be positive, got ", hold_time)
end
const w_cli = cli["window"]
window = if w_cli > 0
    min(w_cli, max(1, Int(floor(hold_time)) - 1))
else
    Int(min(50, max(1, floor(Int, hold_time / 2))))
end
if window >= hold_time
    window = max(1, Int(ceil(Float64(hold_time))) - 1)
end
if window < 1 || window >= hold_time
    error(
        "Need 1 <= window < hold_time (hold_time = ",
        hold_time,
        "). Increase -H / --hold-time or set -w / --window explicitly.",
    )
end
const N = Int(cli["N"])
if N < 1
    error("--N / -N must be >= 1, got ", N)
end

u_start_values = collect(u_p_bottom:starting_step:(u_p_top - u_p_step))
if isempty(u_start_values)
    error(
        "u_start sweep is empty; widen the range or reduce --starting-step / -S (got starting_step = ",
        starting_step,
        ")",
    )
end
cli_limit = cli["limit"]
if cli_limit < 0
    error("--limit / -n must be >= 0, got ", cli_limit)
end
if cli_limit > 0
    n_take = min(cli_limit, length(u_start_values))
    u_start_values = u_start_values[1:n_take]
    @info "CLI --limit: running a subset of u_start values" requested = cli_limit actual = n_take
end
run_name = sanitized_run_name(cli["name"])
if !isempty(strip(string(cli["name"]))) && isempty(run_name)
    error("--name must contain at least one letter, digit, or safe punctuation after sanitization")
end
@info "Ramp domain + timing" u_p_bottom u_p_top u_p_step starting_step hold_time window run_name = ifelse(isempty(run_name), nothing, run_name)

function build_cycle_ramp(u_start::Float32)
    up = collect(u_start:u_p_step:u_p_top)
    down = collect((u_p_top - u_p_step):(-u_p_step):u_p_bottom)
    if u_start > u_p_bottom
        back = collect((u_p_bottom + u_p_step):u_p_step:u_start)
    elseif u_start < u_p_bottom
        back = collect((u_p_bottom - u_p_step):(-u_p_step):u_start)
    else
        back = Float32[]
    end
    cycle_ramp = vcat(up, down, back)
    n_up = length(up)
    n_down = length(down)
    n_back = length(back)
    cycle_phase = vcat(fill("up", n_up), fill("down", n_down), fill("back", n_back))
    return cycle_ramp, cycle_phase
end

function process_segment(
        run_data::PolicyRunData,
        env::RDEEnv,
        u_p_start::Float32,
        segment_u_p::Float32,
        segment_phase::String,
        analysis_window::Int,
    )
    dx = RDE.get_dx(env.prob)
    state_ts = run_data.state_ts
    action_ts = run_data.action_ts
    rewards = run_data.rewards
    @assert allequal(run_data.u_ps) "u_ps must be equal for all segments"
    seg_end = env.prob.params.tmax
    seg_start_window = seg_end - analysis_window
    direction = segment_phase == "down" ? "down" : "up"
    @debug "u_p_start = $u_p_start, u_p_end = $segment_u_p, phase = $segment_phase, direction = $direction"

    state_ix = findall(t -> seg_start_window <= t <= seg_end, state_ts)
    @debug "state_ix endpoint indices: $(extrema(state_ix))"
    if isempty(state_ix)
        ts_rng = length(state_ts) > 0 ? extrema(state_ts) : (missing, missing)
        @debug "process_segment: empty state_ix (time window misses state_ts)" segment_u_p segment_phase u_p_start seg_start_window seg_end analysis_window ts_rng length(state_ts)
        return (;
            direction,
            shock_count = missing,
            shock_speed = missing,
            valid = false,
            u_start = u_p_start,
            phase = segment_phase,
        )
    end
    us, _ = RDE.split_sol(run_data.states)
    us_window = us[state_ix]
    ts_window = state_ts[state_ix]

    shocks_in_window = RDE.count_shocks.(us_window, dx)
    avg_shock_count = mean(shocks_in_window)
    any_shock = any(>(0), shocks_in_window)
    @debug "shocks_in_window: $(sort(unique(shocks_in_window))), (sorted unique)"

    action_ix = findall(t -> seg_start_window <= t <= seg_end, action_ts)
    sustained_high_reward = !isempty(action_ix) && all(rewards[action_ix] .> reward_threshold)
    shock_count_stable = all(==(first(shocks_in_window)), shocks_in_window)
    @debug "shock_count_stable: $shock_count_stable, sustained_high_reward: $sustained_high_reward"
    stable = sustained_high_reward && shock_count_stable && shocks_in_window[1] > 0
    @debug "stable: $stable"

    if !any_shock
        @debug "process_segment: no shock in window" segment_u_p segment_phase u_p_start avg_shock_count sustained_high_reward shock_count_stable
        return (;
            direction,
            shock_count = avg_shock_count,
            shock_speed = missing,
            valid = false,
            u_start = u_p_start,
            phase = segment_phase,
        )
    end

    block_speeds = RDE_Env.speed_tracking(us_window, ts_window, dx)
    block_means = Float32[]
    for block in block_speeds
        ismissing(block) && continue
        for shock_speed_series in block
            n_pts = length(shock_speed_series)
            n_pts <= 1 && continue
            tail_start = n_pts - min(10, n_pts - 1)
            push!(block_means, mean(shock_speed_series[tail_start:n_pts]))
        end
    end
    avg_speed = isempty(block_means) ? missing : mean(block_means)
    @debug "process_segment: result" segment_u_p segment_phase u_p_start valid = stable sustained_high_reward shock_count_stable first_shocks = shocks_in_window[1] n_in_window = length(state_ix)
    return (;
        direction,
        shock_count = avg_shock_count,
        shock_speed = avg_speed,
        valid = stable,
        u_start = u_p_start,
        phase = segment_phase,
    )
end

"""Minimal rollout + one segment analysis to compile hot paths; result discarded."""
function jit_warmup_ramp_sweep!()
    hold_warm = dt
    warm_window = 1
    cycle_ramp = Float32[u_p_bottom]
    cycle_phase = String["up"]
    tmax = hold_warm
    step_times = Float32[0.0f0]
    @debug "JIT warmup: RDEEnv + run_policy" tmax
    env = RDEEnv(;
        dt = dt,
        params = RDEParam(; tmax),
        observation_strat = SectionedStateObservation(),
        action_strat = DirectScalarPressureAction(),
        reward_strat = PeriodicityReward(),
        context_strat = FixedTargetShockCountStrategy(2),
    )
    _reset!(env)
    policy = StepwiseRDEPolicy(env, step_times, cycle_ramp)
    run_data = run_policy(policy, env)
    _ = process_segment(
        run_data,
        env,
        u_p_bottom,
        cycle_ramp[1],
        cycle_phase[1],
        warm_window,
    )
    @info "JIT warmup done (discarded)"
    return nothing
end

function run_one_u_start!(u_start::Float32, progress_task::Union{Nothing, ProgressTask})
    cycle_ramp, cycle_phase = build_cycle_ramp(u_start)
    n_segments = length(cycle_ramp)

    params = RDEParam(; tmax = hold_time, N)
    env = RDEEnv(;
        dt = dt,
        params = params,
        observation_strat = SectionedStateObservation(),
        action_strat = DirectScalarPressureAction(),
        reward_strat = PeriodicityReward(),
        context_strat = FixedTargetShockCountStrategy(2),
    )

    if progress_task !== nothing
        update!(progress_task; total_steps = n_segments)
    end

    rows = Any[]
    for k in 1:n_segments
        segment_u_p = cycle_ramp[k]
        segment_phase = cycle_phase[k]
        if k > 1
            u, λ = RDE.split_sol(copy(env.state))
            seg_reset = StateAndInjectionPressureReset(u, λ, segment_u_p)
            env = @set env.prob.reset_strategy = seg_reset
            u_init = copy(u)
            λ_init = copy(λ)
        else
            _reset!(env)
            u_init, λ_init = RDE.split_sol(copy(env.state))
        end

        if progress_task !== nothing
            update!(
                progress_task;
                message = "$(cycle_phase[k]) u_p=$(cycle_ramp[k])",
            )
        end

        policy = StepwiseRDEPolicy(env, Float32[0.0f0], Float32[cycle_ramp[k]])
        run_data = run_policy(policy, env)
        @debug "extrema start: u:$(extrema(run_data.states[1][1:N])), λ:$(extrema(run_data.states[1][(N + 1):end]))"
        @debug "extrema end: u:$(extrema(run_data.states[end][1:N])), λ:$(extrema(run_data.states[end][(N + 1):end]))"

        row = merge(
            process_segment(run_data, env, u_start, segment_u_p, segment_phase, window),
            (; u_p = segment_u_p, u_init, λ_init),
        )
        ts_lo, ts_hi = extrema(run_data.state_ts)
        @debug "run_one_u_start!: segment done" k n_segments u_p = cycle_ramp[k] phase = cycle_phase[k] valid = row.valid n_actions = length(run_data.action_ts) n_state_ts = length(run_data.state_ts) tmax = env.prob.params.tmax dt = env.dt state_ts_range = (ts_lo, ts_hi)

        push!(rows, row)
        if progress_task !== nothing
            update!(progress_task; step = k)
        end
    end
    if progress_task !== nothing
        finish!(progress_task)
    end
    return rows
end

##
# @debug lines only print with e.g. `JULIA_DEBUG=Main julia -t auto .../speed_analysis_ramp_sweep.jl`
@info "Ramp sweep starting" Threads.nthreads() length(u_start_values)
if Threads.nthreads() > 1 && length(u_start_values) > 1
    @info "Many threads × large RDE rollouts can OOM; if the process dies, try `julia -t 1` or fewer threads."
end

jit_warmup_ramp_sweep!()

n_starts = length(u_start_values)
chunks = Vector{Vector{Any}}(undef, n_starts)

db_dir = datadir("progresslogs")
mkpath(db_dir)
db_suffix = Dates.format(now(), "yyyy-mm-ddTHH-MM-ss")
db_stem = if isempty(run_name)
    "speed_ramp_sweep-" * db_suffix
else
    "speed_ramp_sweep-" * run_name * "-" * db_suffix
end
db_path = joinpath(db_dir, db_stem * ".db")
mpm_name = isempty(run_name) ? "speed_ramp_sweep" : "speed_ramp_sweep-" * run_name
manager = ProgressManager(
    mpm_name,
    n_starts;
    db_path = db_path,
    description = "Speed analysis ramp sweep (u_start)",
    task_descriptions = [string(u) for u in u_start_values],
)
tasks = [get_task(manager, i, :local) for i in 1:n_starts]
@info "Inspect progress with: mpm $(db_path)"

out_dir = datadir("speed_analysis_ramp")

try
    @showprogress desc = "Speed analysis ramp sweep (u_start)" @threads for i in 1:n_starts
        @debug "task start" i u_start_values[i] threadid = Threads.threadid()
        try
            chunks[i] = run_one_u_start!(u_start_values[i], tasks[i])
        catch e
            fail!(tasks[i], e)
            rethrow()
        end
        @debug "task finished" i threadid = Threads.threadid()
    end
finally
    if manager._listener_task !== nothing
        wait(manager._listener_task)
    end
    finish!(manager)
end

all_results = reduce(vcat, chunks)

df = DataFrame(all_results)

##
mkpath(out_dir)
n_segments_total = length(all_results)
out_name = if isempty(run_name)
    cli_limit > 0 ? "ramp_results_limit_$(length(u_start_values))_starts.jld2" : "ramp_results.jld2"
else
    if cli_limit > 0
        "ramp_results_$(run_name)_limit_$(length(u_start_values))_starts.jld2"
    else
        "ramp_results_$(run_name).jld2"
    end
end
out_path = joinpath(out_dir, out_name)
jldsave(out_path; df)
@info "Saved" out_path out_dir n_segments_total length(u_start_values)
