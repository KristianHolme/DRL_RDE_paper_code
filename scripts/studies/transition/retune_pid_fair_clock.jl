# Re-tune the PID baseline under the fair-clock evaluation protocol.
#
# The v1 gains (studies/transition/pid_transition_2) were tuned with control
# active from t=0 on the freshly initialised analytic profile, then evaluated
# after a 100 time-unit uncontrolled hold. Because the PID process variable is
# the injection pressure itself, its u_p(t) command is a deterministic ramp that
# is independent of the flow state, so the outcome depends only on the wave
# state when the ramp arrives. The 4->2 cell sits on the 1-vs-2 wave boundary
# under that mismatch and flips (5/5 without the hold, 2/10 with it).
#
# Fix: tune under the same protocol used for evaluation (100 TU hold, tmax=600,
# dt=0.5, StabilityReward detection), and select in two stages so that gains are
# not picked on single-episode noise as they were in v1.
#   stage 1: N_SAMPLES random gains, 1 trial per (target, start)
#   stage 2: top TOP_K per target re-validated with N_VALIDATE trials per start
# Selection stays one PID per target, as in v1.
#
# Both stages are parallelised at the level of individual runs, so the thread
# count can be raised to the core count of a large host.
#
# Usage: julia --project -t <ncores> scripts/studies/transition/retune_pid_fair_clock.jl
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using Accessors
using DataFrames
using IntervalSets
using JLD2
using ProgressMeter
using Random
using Statistics
using Base.Threads

# Protocol: identical to the fair-clock evaluation legs.
const RETUNE_DT = 0.5f0
const N_SAMPLES = parse(Int, get(ENV, "PID_RETUNE_SAMPLES", "512"))
const TOP_K = parse(Int, get(ENV, "PID_RETUNE_TOPK", "10"))
const N_VALIDATE = parse(Int, get(ENV, "PID_RETUNE_VALIDATE", "20"))
const SEED = 0
const TAG = get(ENV, "PID_RETUNE_TAG", "")

# Restrict to a subset of targets to split the job across hosts.
const TARGETS = let raw = get(ENV, "PID_RETUNE_TARGETS", "1,2,3,4")
    parse.(Int, split(raw, ","))
end

const KP_RANGE = 0.0f0 .. 5.0f0
const KI_RANGE = 0.0f0 .. 1.0f0
const KD_RANGE = 0.0f0 .. 3.0f0

const OUT_DIR = datadir("pid_baseline")
const OUT_PATH = joinpath(OUT_DIR, "pid_retune_fair_clock$(TAG).jld2")
##
mutable struct DelayedPIDPolicy{P, E, T <: AbstractFloat} <: AbstractRDEPolicy
    policy::P
    env::E
    start_time::T
    cache_reset::Bool
end

DRL_RDE.get_env(π::DelayedPIDPolicy) = π.env

function RDE_Env._predict_action(π::DelayedPIDPolicy, s)
    if π.env.t < π.start_time
        return zeros(Float32, 3)
    end
    if !π.cache_reset
        RDE_Env.reset_cache!(π.env.cache.action_cache)
        π.cache_reset = true
    end
    return RDE_Env._predict_action(π.policy, s)
end

function detect_pair_time(data, env, target)
    found, idx, _ = detect_transition(data, env.prob, target)
    if !found
        return nothing
    end
    reward_times = vcat(data.action_ts[2:end], data.state_ts[end])
    return reward_times[idx] - DRL_RDE.DELAY_TIME
end

# One trial of the fair-clock PID leg: 100 TU uncontrolled hold, then PID.
# Returns the transition time measured from control start, or NaN on failure.
function pid_trial(start, target, Kp, Ki, Kd)
    env_config = get_PID_eval_config(target).env_config
    @reset env_config.tmax = DRL_RDE.DEFAULT_TRANSITION_TMAX
    @reset env_config.dt = RETUNE_DT
    @reset env_config.reward_strat = StabilityReward(1.0f0)
    @reset env_config.reset_strategy = ShiftReset(NShock(start))
    @reset env_config.context_strat = FixedTargetShockCountStrategy(target)
    env = create_env(env_config)
    pid = PIDControllerPolicy(; Kp, Ki, Kd)
    delayed = DelayedPIDPolicy(pid, env, DRL_RDE.DELAY_TIME, false)
    data = run_policy(delayed, env)
    t = detect_pair_time(data, env, target)
    return t === nothing ? NaN : Float64(t)
end

# Run a flat list of (start, target, Kp, Ki, Kd) jobs across all threads.
function run_jobs(jobs, desc)
    out = Vector{Float64}(undef, length(jobs))
    p = Progress(length(jobs); desc, showspeed = true)
    @threads for k in eachindex(jobs)
        j = jobs[k]
        out[k] = pid_trial(j.start, j.target, j.Kp, j.Ki, j.Kd)
        next!(p)
    end
    return out
end

_success(times) = count(!isnan, times) / length(times)
_median_time(times) = (ok = filter(!isnan, times); isempty(ok) ? NaN : median(ok))
##
Random.seed!(SEED)
Kps = rand(KP_RANGE, N_SAMPLES)
Kis = rand(KI_RANGE, N_SAMPLES)
Kds = rand(KD_RANGE, N_SAMPLES)

println("threads: ", nthreads())
println("protocol: dt=", RETUNE_DT, " tmax=", DRL_RDE.DEFAULT_TRANSITION_TMAX, " hold=", DRL_RDE.DELAY_TIME)
println("targets: ", TARGETS)
##
# ── Stage 1: coarse search, one trial per (sample, target, start) ─────────
stage1_jobs = [
    (; sample = i, target = t, start = s, Kp = Kps[i], Ki = Kis[i], Kd = Kds[i])
        for t in TARGETS for i in 1:N_SAMPLES for s in setdiff(1:4, t)
]
println("stage 1: ", length(stage1_jobs), " runs")
stage1_times = run_jobs(stage1_jobs, "stage 1 (coarse)")

stage1 = let rows = NamedTuple[]
    for t in TARGETS, i in 1:N_SAMPLES
        ix = findall(j -> j.sample == i && j.target == t, stage1_jobs)
        times = stage1_times[ix]
        push!(
            rows, (;
                sample = i, target = t,
                Kp = Kps[i], Ki = Kis[i], Kd = Kds[i],
                mean_success = _success(times),
                median_time = _median_time(times),
            ),
        )
    end
    DataFrame(rows)
end
mkpath(OUT_DIR)
jldsave(OUT_PATH; stage1, dt = RETUNE_DT, targets = TARGETS)
println("stage 1 done, saved to ", OUT_PATH)
##
# ── Stage 2: re-validate the top candidates per target ───────────────────
candidates = let rows = NamedTuple[]
    for t in TARGETS
        cand = stage1[stage1.target .== t, :]
        sort!(cand, [order(:mean_success, rev = true), :median_time])
        for r in eachrow(first(cand, TOP_K))
            push!(rows, (; target = t, Kp = r.Kp, Ki = r.Ki, Kd = r.Kd))
        end
    end
    rows
end

stage2_jobs = [
    (; cand = c, candidates[c].target, start = s, candidates[c].Kp, candidates[c].Ki, candidates[c].Kd)
        for c in eachindex(candidates) for s in setdiff(1:4, candidates[c].target) for _ in 1:N_VALIDATE
]
println("stage 2: ", length(candidates), " candidates -> ", length(stage2_jobs), " runs")
stage2_times = run_jobs(stage2_jobs, "stage 2 (validate)")

stage2 = let rows = NamedTuple[]
    for c in eachindex(candidates)
        t = candidates[c].target
        starts = setdiff(1:4, t)
        per_start = Dict{Int, NamedTuple}()
        rates = Float64[]
        all_times = Float64[]
        for s in starts
            ix = findall(j -> j.cand == c && j.start == s, stage2_jobs)
            times = stage2_times[ix]
            push!(rates, _success(times))
            append!(all_times, filter(!isnan, times))
            per_start[s] = (;
                success_rate = _success(times),
                median_time = _median_time(times),
                times = filter(!isnan, times),
            )
        end
        push!(
            rows, (;
                target = t, candidates[c].Kp, candidates[c].Ki, candidates[c].Kd,
                mean_success = mean(rates),
                min_success = minimum(rates),
                median_time = isempty(all_times) ? NaN : median(all_times),
                per_start,
            ),
        )
    end
    DataFrame(rows)
end
##
# ── Select one PID per target: success first, then median time ───────────
best_pids = let rows = NamedTuple[]
    for t in TARGETS
        cand = stage2[stage2.target .== t, :]
        sort!(cand, [order(:mean_success, rev = true), :median_time])
        r = first(cand)
        push!(
            rows, (;
                target_shock_count = t,
                Kp = Float32(r.Kp), Ki = Float32(r.Ki), Kd = Float32(r.Kd),
                success_rate = r.mean_success,
                median_time = r.median_time,
            ),
        )
    end
    DataFrame(rows)
end

jldsave(
    OUT_PATH;
    stage1, stage2, best_pids,
    dt = RETUNE_DT, n_validate = N_VALIDATE, n_samples = N_SAMPLES, targets = TARGETS,
)
println("\nsaved ", OUT_PATH)
##
println("\nRe-tuned PID gains (fair-clock protocol, dt=", RETUNE_DT, "):")
for r in eachrow(best_pids)
    println(
        "  target ", r.target_shock_count,
        ": Kp=", round(r.Kp, digits = 4),
        " Ki=", round(r.Ki, digits = 4),
        " Kd=", round(r.Kd, digits = 4),
        "  mean success=", round(100 * r.success_rate, digits = 1), "%",
        "  median time=", round(r.median_time, digits = 1),
    )
end

println("\nPer-pair success at the selected gains (", N_VALIDATE, " trials each):")
for t in TARGETS
    cand = stage2[stage2.target .== t, :]
    sort!(cand, [order(:mean_success, rev = true), :median_time])
    r = first(cand)
    for (start, v) in sort(collect(r.per_start); by = first)
        println(
            "  ", start, "->", t,
            "  ", round(Int, 100 * v.success_rate), "%",
            "  median=", isnan(v.median_time) ? "fail" : string(round(v.median_time, digits = 1)),
        )
    end
end
