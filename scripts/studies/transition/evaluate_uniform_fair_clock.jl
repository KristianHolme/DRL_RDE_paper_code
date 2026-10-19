# Fair-clock evaluation of ar14 US4, per-target two-step, and per-target PID.
# Detection clock lock_dt=0.5, 5 trials, tmax=600, 100-time-unit hold, ShiftReset.
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using Accessors
using DataFrames
using JLD2
using ProgressMeter
using Statistics
using Base.Threads
##
function pair_from_trials(times; n_trials)
    n_ok = length(times)
    success_rate = n_ok / n_trials
    mean_time = n_ok == 0 ? NaN : Float64(mean(times))
    std_time = n_ok <= 1 ? NaN : Float64(std(times))
    return (;
        success_rate,
        mean_time,
        std_time,
        times = Float64.(times),
        n_trials,
        n_ok,
    )
end

function empty_pair_dict()
    return Dict{Tuple{Int, Int}, NamedTuple}()
end

function matrices_from_pairs(pairs; n_targets = 4)
    S = fill(NaN, n_targets, n_targets)
    T = fill(NaN, n_targets, n_targets)
    for ((s, t), r) in pairs
        S[s, t] = 100 * r.success_rate
        T[s, t] = r.mean_time
    end
    return S, T
end

function detect_pair_time(data, env, target)
    found, idx, _ = detect_transition(data, env.prob, target)
    if !found
        return nothing
    end
    reward_times = vcat(data.action_ts[2:end], data.state_ts[end])
    return reward_times[idx] - DRL_RDE.DELAY_TIME
end

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

function evaluate_two_step_pair(start, target, intermediate_u_p, intermediate_duration; n_trials = 5)
    times = Float32[]
    for _ in 1:n_trials
        env = RDEEnv(
            RDEParam(; tmax = DRL_RDE.DEFAULT_TRANSITION_TMAX);
            dt = 0.5f0,
            action_strat = DirectScalarPressureAction(),
            observation_strat = StateObservation(),
            reset_strategy = ShiftReset(NShock(start)),
            reward_strat = StabilityReward(1.0f0),
            context_strat = FixedTargetShockCountStrategy(target),
        )
        policy = StepwiseRDEPolicy(
            env,
            Float32[0, DRL_RDE.DELAY_TIME, DRL_RDE.DELAY_TIME + intermediate_duration],
            Float32[RDE.SHOCK_PRESSURES[start], intermediate_u_p, RDE.SHOCK_PRESSURES[target]],
        )
        data = run_policy(policy, env)
        t = detect_pair_time(data, env, target)
        t === nothing || push!(times, t)
    end
    return pair_from_trials(times; n_trials)
end

# PID control step used in fair-clock gain tuning (see
# scripts/studies/transition/retune_pid_fair_clock.jl). The PID law updates
# integral/derivative with env.dt, so this must match the retune protocol.
# SU instead holds its trained-period action on the same 0.5 grid.
const PID_EVAL_DT = 0.5f0

function evaluate_pid_pair(start, target, Kp, Ki, Kd; n_trials = 5)
    env_config = get_PID_eval_config(target).env_config
    @reset env_config.tmax = DRL_RDE.DEFAULT_TRANSITION_TMAX
    @reset env_config.dt = PID_EVAL_DT
    @reset env_config.reward_strat = StabilityReward(1.0f0)
    @reset env_config.reset_strategy = ShiftReset(NShock(start))
    @reset env_config.context_strat = FixedTargetShockCountStrategy(target)
    pid = PIDControllerPolicy(; Kp, Ki, Kd)
    times = Float32[]
    for _ in 1:n_trials
        env = create_env(env_config)
        delayed = DelayedPIDPolicy(pid, env, DRL_RDE.DELAY_TIME, false)
        data = run_policy(delayed, env)
        t = detect_pair_time(data, env, target)
        t === nothing || push!(times, t)
    end
    return pair_from_trials(times; n_trials)
end

function us4_checkpoint_path(seed, target)
    return datadir(
        "studies", "ar14_all_target", "fair_clock",
        "us4_seed=$(seed)_target=$(target).jld2",
    )
end

function evaluate_us4_job(job, models_dir, io_lock; n_trials = 5)
    path = us4_checkpoint_path(job.seed, job.target)
    if isfile(path)
        return lock(io_lock) do
            return load(path, "record")
        end
    end
    experiment = job.experiment
    model_path = joinpath(models_dir, basename(job.model_path))
    control_dt = experiment.env_config.dt
    @reset experiment.env_config.reward_strat = StabilityReward(1.0f0)
    @reset experiment.env_config.context_strat = FixedTargetShockCountStrategy(job.target)
    @reset experiment.env_config.tmax = DRL_RDE.DEFAULT_TRANSITION_TMAX
    @reset experiment.env_config.dt = 0.5f0
    jl_policy = lock(io_lock) do
        return get_jl_policy(experiment, model_path)
    end
    pairs = empty_pair_dict()
    for start in setdiff(1:4, job.target)
        @reset experiment.env_config.reset_strategy = ShiftReset(NShock(start))
        times = Float32[]
        for _ in 1:n_trials
            env = create_env(experiment.env_config)
            delayed = DelayedPolicy(jl_policy, DRL_RDE.DELAY_TIME, env)
            policy = HeldActionPolicy(delayed, env, control_dt)
            data = run_policy(policy, env)
            t = detect_pair_time(data, env, job.target)
            t === nothing || push!(times, t)
        end
        pairs[(start, job.target)] = pair_from_trials(times; n_trials)
    end
    S, T = matrices_from_pairs(pairs)
    record = (;
        recipe = "us4",
        seed = job.seed,
        target = job.target,
        path = job.path,
        model_path,
        pairs,
        success_matrix = S,
        time_matrix = T,
    )
    mkpath(dirname(path))
    lock(io_lock) do
        jldsave(path; record)
        return nothing
    end
    return record
end

function column_metrics(record)
    t = record.target
    starts = setdiff(1:4, t)
    rates = [record.pairs[(s, t)].success_rate for s in starts]
    times = [record.pairs[(s, t)].mean_time for s in starts]
    n_complete = count(s -> record.pairs[(s, t)].n_ok == record.pairs[(s, t)].n_trials, starts)
    mean_success = mean(rates)
    finite_times = filter(isfinite, times)
    mean_time = isempty(finite_times) ? Inf : mean(finite_times)
    return (; mean_success, mean_time, n_complete)
end

function select_us4_winners(records)
    winners = Dict{Int, Any}()
    for t in 1:4
        cand = filter(r -> r.target == t, records)
        metrics = [column_metrics(r) for r in cand]
        best = 1
        for i in eachindex(cand)
            m = metrics[i]
            b = metrics[best]
            better = m.mean_success > b.mean_success ||
                (m.mean_success == b.mean_success && m.mean_time < b.mean_time)
            if better
                best = i
            end
        end
        winners[t] = cand[best]
    end
    return winners
end

function winner_matrices(winners)
    S = fill(NaN, 4, 4)
    T = fill(NaN, 4, 4)
    seeds = zeros(Int, 4)
    for t in 1:4
        rec = winners[t]
        seeds[t] = rec.seed
        for s in setdiff(1:4, t)
            S[s, t] = 100 * rec.pairs[(s, t)].success_rate
            T[s, t] = rec.pairs[(s, t)].mean_time
        end
    end
    return S, T, seeds
end

function offdiag_values(mat)
    return [mat[s, t] for t in 1:4 for s in setdiff(1:4, t)]
end

function paired_wins(a, b)
    va = offdiag_values(a)
    vb = offdiag_values(b)
    n_win = 0
    n_lose = 0
    n_tie = 0
    for (x, y) in zip(va, vb)
        ax = isfinite(x)
        ay = isfinite(y)
        if ax && !ay
            n_win += 1
        elseif !ax && ay
            n_lose += 1
        elseif ax && ay
            if x < y
                n_win += 1
            elseif x > y
                n_lose += 1
            else
                n_tie += 1
            end
        else
            n_tie += 1
        end
    end
    return (;
        n_win,
        n_lose,
        n_tie,
        median_a = median(filter(isfinite, va)),
        median_b = median(filter(isfinite, vb)),
    )
end

function fmt_time(x)
    return isfinite(x) ? string(round(x; digits = 1)) : "fail"
end

function write_fair_clock_summary(io, us4_records, winners, sus_success, sus_time, sus_seeds, two_step_success, two_step_time, pid_success, pid_time, summary)
    println(io, "Uniform fair-clock evaluation")
    println(io, "Protocol: 5 trials, tmax=600, delay=100, lock_dt=0.5, ShiftReset")
    println(io, "SUS selection: best-of-six per target, max mean success then min mean time")
    println(io, "Selected SUS seeds by target: ", sus_seeds)
    println(io)
    println(io, "Aggregate median successful transition time")
    println(io, "  SUS:      ", summary.sus_median)
    println(io, "  Two-step: ", summary.two_step_median)
    println(io, "  PID:      ", summary.pid_median)
    println(io)
    println(io, "Paired wins (faster time; success beats failure): W-L-T")
    println(io, "  SUS vs two-step: ", summary.vs_two_step.n_win, "-", summary.vs_two_step.n_lose, "-", summary.vs_two_step.n_tie)
    println(io, "  SUS vs PID:      ", summary.vs_pid.n_win, "-", summary.vs_pid.n_lose, "-", summary.vs_pid.n_tie)
    println(io)
    println(io, "Cells where SUS does not beat two-step:")
    for t in 1:4, s in setdiff(1:4, t)
        st = sus_time[s, t]
        bt = two_step_time[s, t]
        ss = sus_success[s, t]
        bs = two_step_success[s, t]
        sus_better = (isfinite(st) && !isfinite(bt)) || (isfinite(st) && isfinite(bt) && st < bt)
        if !sus_better
            println(io, "  ", s, "→", t, "  SUS ", fmt_time(st), " (", round(ss; digits = 1), "%)  two-step ", fmt_time(bt), " (", round(bs; digits = 1), "%)")
        end
    end
    println(io)
    println(io, "Cells where SUS does not beat PID:")
    for t in 1:4, s in setdiff(1:4, t)
        st = sus_time[s, t]
        bt = pid_time[s, t]
        ss = sus_success[s, t]
        bs = pid_success[s, t]
        sus_better = (isfinite(st) && !isfinite(bt)) || (isfinite(st) && isfinite(bt) && st < bt)
        if !sus_better
            println(io, "  ", s, "→", t, "  SUS ", fmt_time(st), " (", round(ss; digits = 1), "%)  PID ", fmt_time(bt), " (", round(bs; digits = 1), "%)")
        end
    end
    println(io)
    println(io, "Selected SUS cells (start → target): success%, mean time, trial times")
    for t in 1:4
        rec = winners[t]
        println(io, "  target ", t, " seed ", rec.seed)
        for s in setdiff(1:4, t)
            r = rec.pairs[(s, t)]
            println(io, "    ", s, "→", t, "  ", round(100 * r.success_rate; digits = 1), "%  ", fmt_time(r.mean_time), "  ", r.times)
        end
    end
    println(io)
    println(io, "All US4 seeds (robustness)")
    for rec in sort(us4_records; by = r -> (r.target, r.seed))
        m = column_metrics(rec)
        println(io, "  t=", rec.target, " seed=", rec.seed, " mean_success=", round(100 * m.mean_success; digits = 1), "% mean_time=", fmt_time(m.mean_time), " complete_starts=", m.n_complete)
        for s in setdiff(1:4, rec.target)
            r = rec.pairs[(s, rec.target)]
            println(io, "    ", s, "→", rec.target, "  ", round(100 * r.success_rate; digits = 1), "%  ", fmt_time(r.mean_time), "  ", r.times)
        end
    end
    return nothing
end
##
out_dir = datadir("studies", "ar14_all_target", "fair_clock")
out_path = joinpath(out_dir, "uniform_fair_clock.jld2")
summary_path = joinpath(out_dir, "uniform_fair_clock.txt")
mkpath(out_dir)
##
study = "ar14_all_target"
results_dir = datadir("studies", study, "results")
models_dir = datadir("studies", study, "models")
us4_files = sort(
    filter(
        p -> occursin("recipe=us4_run", basename(p)) && endswith(p, ".jld2"),
        readdir(results_dir; join = true),
    ),
)
if length(us4_files) != 24
    error("Expected 24 US4 result files, got $(length(us4_files)) in $results_dir")
end
@info "US4 files" length(us4_files)
us4_jobs = map(us4_files) do path
    data = load(path)
    return (;
        seed = Int(data["experiment_seed"]),
        target = Int(data["target_shock_count"]),
        path,
        model_path = String(data["model_path"]),
        experiment = data["experiment"],
    )
end
##
us4_records = Vector{Any}(undef, length(us4_jobs))
io_lock = ReentrantLock()
p = Progress(length(us4_jobs); desc = "US4 fair clock", showspeed = true)
Threads.@threads for i in eachindex(us4_jobs)
    us4_records[i] = evaluate_us4_job(us4_jobs[i], models_dir, io_lock)
    next!(p)
end
us4_records = identity.(us4_records)
winners = select_us4_winners(us4_records)
sus_success, sus_time, sus_seeds = winner_matrices(winners)
@info "Selected SUS seeds" sus_seeds
##
best_two_step = get_best_configs_per_target()
two_step_pairs = empty_pair_dict()
p = Progress(12; desc = "Two-step fair clock", showspeed = true)
for t in 1:4
    cfg = get_best_config_for_target(best_two_step, t)
    for s in setdiff(1:4, t)
        two_step_pairs[(s, t)] = evaluate_two_step_pair(
            s, t, cfg.intermediate_u_p, cfg.intermediate_duration,
        )
        next!(p)
    end
end
two_step_success, two_step_time = matrices_from_pairs(two_step_pairs)
##
best_pids = load(
    datadir("pid_baseline", "pid_retune_fair_clock.jld2"),
    "best_pids",
)
pid_pairs = empty_pair_dict()
p = Progress(12; desc = "PID fair clock", showspeed = true)
for t in 1:4
    row = get_best_pid_for_target(best_pids, t)
    for s in setdiff(1:4, t)
        pid_pairs[(s, t)] = evaluate_pid_pair(s, t, row.Kp, row.Ki, row.Kd)
        next!(p)
    end
end
pid_success, pid_time = matrices_from_pairs(pid_pairs)
##
vs_two = paired_wins(sus_time, two_step_time)
vs_pid = paired_wins(sus_time, pid_time)
summary = (;
    sus_seeds,
    vs_two_step = vs_two,
    vs_pid = vs_pid,
    sus_median = median(filter(isfinite, offdiag_values(sus_time))),
    two_step_median = median(filter(isfinite, offdiag_values(two_step_time))),
    pid_median = median(filter(isfinite, offdiag_values(pid_time))),
)
@info "Aggregate" summary
##
jldsave(
    out_path;
    us4_records,
    winners,
    sus_success,
    sus_time,
    sus_seeds,
    two_step_success,
    two_step_time,
    two_step_pairs,
    pid_success,
    pid_time,
    pid_pairs,
    summary,
)
open(summary_path, "w") do io
    write_fair_clock_summary(
        io, us4_records, winners, sus_success, sus_time, sus_seeds,
        two_step_success, two_step_time, pid_success, pid_time, summary,
    )
end
println(read(summary_path, String))
println("saved ", out_path)
println("saved ", summary_path)
