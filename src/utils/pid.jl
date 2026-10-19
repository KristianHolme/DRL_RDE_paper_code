@kwdef struct PIDExperiment <: AbstractExperiment
    study::String = "PID_baseline"
    eval_config::RDEEvalConfig = RDEEvalConfig()
    pid_policy::PIDControllerPolicy
end

@kwdef struct PIDTransitionExperiment <: AbstractExperiment
    study::String = "pid_transition"
    eval_config::RDEEvalConfig = RDEEvalConfig()
    pid_policy::PIDControllerPolicy = PIDControllerPolicy()
    target_shock_count::Int = 1
    init_shock_count::Int = 1
end

function get_PID_eval_config(target_shock_count::Int)
    env_config = RDEEnvConfig(
        observation_strat = MeanInjectionPressureObservation(),
        reward_strat = TransitionBasedReward(StabilityReward(1.0f0)),
        tmax = 500.0f0,
        reset_strategy = ShiftReset(CycleShockReset(shocks = setdiff(1:4, target_shock_count))),
        context_strat = FixedTargetShockCountStrategy(target_shock_count),
        action_strat = PIDAction()
    )
    return RDEEvalConfig(env_config = env_config, n_eval_episodes = 3 * 10)
end


function experiment_to_dict(experiment::PIDExperiment)
    cfg = Dict("Kp" => experiment.pid_policy.Kp, "Ki" => experiment.pid_policy.Ki, "Kd" => experiment.pid_policy.Kd, "target_shock_count" => experiment.eval_config.env_config.reward_strat.target_shock_count)
    return cfg
end

function run_pid_experiment(experiment::PIDExperiment; progress_bar = nothing)
    eval_config = experiment.eval_config
    env = create_env(eval_config.env_config)
    policy = experiment.pid_policy
    mean_rewards = []
    for i in 1:eval_config.n_eval_episodes
        # reset_pid_cache!(policy)
        data = run_policy(policy, env) #this resets the env
        rewards = data.rewards
        mean_reward = mean(rewards)
        push!(mean_rewards, mean_reward)
    end
    cfg = experiment_to_dict(experiment)
    mean_reward = mean(mean_rewards)
    # @info "$(savename(cfg)) : $mean_reward"
    results = Dict("mean_reward" => mean_reward, "experiment" => experiment)
    dir = datadir("studies", "transition", "pid_baseline", "results")
    safesave(joinpath(dir, experiment_savename(cfg, "jld2")), merge(results, cfg))

    return mean_rewards
end


function run_pid_transition(experiment::PIDTransitionExperiment; make_plot = true, display_plot = false, save = true)
    target_shock_count = experiment.target_shock_count
    init_shock_count = experiment.init_shock_count

    eval_config = experiment.eval_config
    policy = experiment.pid_policy

    env_config = eval_config.env_config
    # env_config = @set env_config.reward_strat = PeriodicityReward()
    env_config = @set env_config.reward_strat = StabilityReward(1.0f0)
    env_config = @set env_config.reset_strategy = ShiftReset(NShock(init_shock_count))
    env = create_env(env_config)

    # reset_pid_cache!(policy)
    data = run_policy(policy, env)
    found, transition_idx, stability_end_idx = detect_transition(data, env.prob, target_shock_count)
    if found
        transition_time = data.action_ts[transition_idx]
        @debug "Transition time: $transition_time"
    else
        transition_time = NaN
    end
    cfg = Dict(
        "target_shock_count" => target_shock_count,
        "init_shock_count" => init_shock_count,
        "Kp" => policy.Kp,
        "Ki" => policy.Ki,
        "Kd" => policy.Kd
    )
    result = Dict(
        "experiment" => experiment,
        "transition_time" => transition_time,
    )
    if save
        safesave(
            datadir("studies", "transition", experiment.study, experiment_savename(cfg, "jld2")),
            merge(result, cfg)
        )
    end

    if make_plot
        title = experiment_savename(cfg)
        fig = plot_shifted_history(data, env.prob.x, title = title, target_shock_count = target_shock_count)
        safesave(plotsdir("studies", "transition", experiment.study, title * ".png"), fig)
        if !isnan(transition_time)
            for content in fig.content
                if content isa Axis
                    vlines!(content, data.action_ts[transition_idx], color = :red, linestyle = :dash)
                    vlines!(content, data.action_ts[stability_end_idx], color = :red, linestyle = :dash, alpha = 0.5)
                end
            end
        end
        if display_plot
            display(fig)
        end
    end
    return transition_time
end

function plot_pid_run(experiment::PIDExperiment)
    env = create_env(experiment.eval_config.env_config)
    policy = experiment.pid_policy
    for i in 1:3
        reset!(env)
        # reset_pid_cache!(policy)
        data = run_policy(policy, env)
        fig = plot_shifted_history(data, env.prob.x, title = experiment_savename(experiment_to_dict(experiment)))
        display(fig)
    end
    return mean_rewards
end

function run_pid_transition_experiments(experiments::Vector{PIDTransitionExperiment})
    p = Progress(length(experiments), showspeed = true, desc = "Running PID transition experiments...")
    for experiment in experiments
        run_pid_transition(experiment; make_plot = true, display_plot = false)
        next!(p)
    end
    return
end

function run_pid_transition_experiments_threaded(experiments::Vector{PIDTransitionExperiment})
    p = Progress(length(experiments), showspeed = true, desc = "Running PID transition experiments...")
    Threads.@threads for experiment in experiments
        run_pid_transition(experiment; make_plot = false, display_plot = false)
        next!(p)
    end
    return nothing
end


# ---------------------------------------------------------------------------
# PID transition analysis functions
# ---------------------------------------------------------------------------

const _DEFAULT_PID_TRANSITION_STUDY = "pid_transition_2"
const _PID_GROUP_COLS = [:target_shock_count, :Kp, :Ki, :Kd]

_is_valid_tt(x::Real) = !isnan(x)
_is_valid_tt(::Missing) = false

_ceq(a, b) = coalesce.(a .== b, false)

default_pid_transition_study() = _DEFAULT_PID_TRANSITION_STUDY

function get_pid_transition_results_dir(study::AbstractString = default_pid_transition_study())
    return datadir("studies", "transition", study)
end

function get_pid_transition_analysis_dir(study::AbstractString = default_pid_transition_study())
    return datadir("studies", "transition", study * "_per_target")
end

function get_pid_transition_best_pids_path(study::AbstractString = default_pid_transition_study())
    return joinpath(get_pid_transition_analysis_dir(study), "best_pids.jld2")
end

function get_pid_transition_time_matrix_path(study::AbstractString = default_pid_transition_study())
    return joinpath(get_pid_transition_results_dir(study), "time_matrix.jld2")
end

function get_pid_transition_time_matrix_per_target_path(study::AbstractString = default_pid_transition_study())
    return joinpath(get_pid_transition_analysis_dir(study), "time_matrix.jld2")
end

"""
    load_pid_transition_results(study) -> DataFrame

Load raw PID transition experiment results via `DrWatson.collect_results!`.
"""
function load_pid_transition_results(study::AbstractString = default_pid_transition_study())
    results_dir = get_pid_transition_results_dir(study)
    return DrWatson.collect_results!(results_dir; black_list = ["time_matrix"])
end

"""
    aggregate_pid_transition_results(res::DataFrame) -> DataFrame

Group raw results by `(target_shock_count, Kp, Ki, Kd)` and compute
`:success_rate` (fraction of non-NaN transitions) and `:mean_time`
(mean of successful transition times).
"""
function aggregate_pid_transition_results(res::DataFrame)
    return combine(
        groupby(res, _PID_GROUP_COLS),
        :transition_time => (tt -> mean(_is_valid_tt.(tt))) => :success_rate,
        :transition_time => (
            tt -> begin
                vals = filter(_is_valid_tt, tt)
                return isempty(vals) ? NaN : mean(vals)
            end
        ) => :mean_time,
    )
end

"""
    get_best_pid_configs_per_target(res::DataFrame) -> DataFrame

For each target shock count, select the single best PID configuration
(highest success rate, then lowest mean time as tiebreaker).
Returns a DataFrame with columns
`[:target_shock_count, :Kp, :Ki, :Kd, :success_rate, :mean_time]`.
"""
function get_best_pid_configs_per_target(res::DataFrame)
    pid_agg = aggregate_pid_transition_results(res)
    best_pids = DataFrame()
    targets = sort(collect(skipmissing(unique(res.target_shock_count))))
    for t in targets
        cand = pid_agg[_ceq(pid_agg.target_shock_count, t), :]
        if nrow(cand) == 0
            continue
        end
        max_success = maximum(cand.success_rate)
        cand2 = cand[_ceq(cand.success_rate, max_success), :]
        best_idx = argmin(cand2.mean_time)
        push!(best_pids, cand2[best_idx, :])
    end
    return best_pids
end

"""
    get_best_pid_for_target(best_pids::DataFrame, target::Int) -> DataFrameRow

Return the best PID configuration row for a specific target shock count.
"""
function get_best_pid_for_target(best_pids::DataFrame, target::Int)
    idx = findfirst(==(target), best_pids.target_shock_count)
    isnothing(idx) && error("No best PID found for target shock count $target")
    return best_pids[idx, :]
end

"""
    pid_transition_time_matrix_per_target(res, best_pids) -> Matrix{Float64}

Build a 4x4 time matrix where entry `[init, target]` is the median
transition time using the per-target best PID configuration.
Diagonal entries are 0.0; missing transitions remain NaN.
"""
function pid_transition_time_matrix_per_target(res::DataFrame, best_pids::DataFrame)
    mat = fill(NaN, 4, 4)
    for i in 1:4
        mat[i, i] = 0.0
    end
    for row in eachrow(best_pids)
        target = row.target_shock_count
        cfg_subset = res[
            _ceq(res.target_shock_count, target) .&
                _ceq(res.Kp, row.Kp) .&
                _ceq(res.Ki, row.Ki) .&
                _ceq(res.Kd, row.Kd),
            :,
        ]
        for init in setdiff(1:4, target)
            df_start = cfg_subset[_ceq(cfg_subset.init_shock_count, init), :]
            if !isempty(df_start)
                tt = filter(_is_valid_tt, df_start.transition_time)
                if !isempty(tt)
                    mat[init, target] = median(tt)
                end
            end
        end
    end
    return mat
end

"""
    pid_transition_best_time_matrix(res) -> Matrix{Float64}

Build a 4x4 matrix of the global best (minimum) observed transition time
across all sampled PID configurations for each `(init, target)` pair.
"""
function pid_transition_best_time_matrix(res::DataFrame)
    mat = fill(NaN, 4, 4)
    for i in 1:4
        mat[i, i] = 0.0
    end
    for init in 1:4
        df_init = res[_ceq(res.init_shock_count, init), :]
        for target in setdiff(1:4, init)
            df_target = df_init[_ceq(df_init.target_shock_count, target), :]
            transition_times = filter(_is_valid_tt, df_target.transition_time)
            if !isempty(transition_times)
                mat[init, target] = minimum(transition_times)
            end
        end
    end
    return mat
end

"""
    save_pid_transition_analysis(study, best_pids, time_matrix, time_matrix_per_target)

Persist the analysis artifacts (best PIDs DataFrame, global time matrix,
per-target time matrix) to JLD2 files.
"""
function save_pid_transition_analysis(
        study::AbstractString,
        best_pids::DataFrame,
        time_matrix::Matrix{Float64},
        time_matrix_per_target::Matrix{Float64},
    )
    analysis_dir = get_pid_transition_analysis_dir(study)
    mkpath(analysis_dir)

    safesave(
        get_pid_transition_best_pids_path(study),
        Dict("best_pids" => best_pids),
    )
    safesave(
        get_pid_transition_time_matrix_per_target_path(study),
        Dict("time_matrix" => time_matrix_per_target),
    )
    safesave(
        get_pid_transition_time_matrix_path(study),
        Dict("time_matrix" => time_matrix),
    )
    @info "PID transition analysis saved" study analysis_dir
    return nothing
end

"""
    analyze_pid_transition_results(study) -> (best_pids, time_matrix, time_matrix_per_target)

Full pipeline: load raw results, compute best PID per target,
build both time matrices, save everything, and return the artifacts.
"""
function analyze_pid_transition_results(study::AbstractString = default_pid_transition_study())
    @info "Loading PID transition results..." study
    res = load_pid_transition_results(study)
    @info "Loaded $(nrow(res)) result rows"

    best_pids = get_best_pid_configs_per_target(res)
    time_matrix_pt = pid_transition_time_matrix_per_target(res, best_pids)
    time_matrix = pid_transition_best_time_matrix(res)

    save_pid_transition_analysis(study, best_pids, time_matrix, time_matrix_pt)
    return (; best_pids, time_matrix, time_matrix_per_target = time_matrix_pt)
end

"""
    load_pid_transition_best_pids(study) -> DataFrame

Load the saved best-PID-per-target DataFrame. If the saved file does not
exist, recomputes from raw results, saves, and returns.
"""
function load_pid_transition_best_pids(study::AbstractString = default_pid_transition_study())
    path = get_pid_transition_best_pids_path(study)
    if isfile(path)
        data = load(path)
        return data["best_pids"]::DataFrame
    end
    @info "best_pids not found on disk, recomputing..." path
    result = analyze_pid_transition_results(study)
    return result.best_pids
end
