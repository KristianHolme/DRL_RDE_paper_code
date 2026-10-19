uses_moving_frame(::MultiCenteredMovingFrameObservation) = true
uses_moving_frame(::SectionedStateMovingFrameObservation) = true
uses_moving_frame(obs::MultiAgentObservationWithSinusoidalPE) = uses_moving_frame(obs.inner)
uses_moving_frame(::Any) = false

function create_env(env_config::EnvConfig; spatial_kernel_width::Int = 0)
    params = RDEParam(tmax = env_config.tmax)
    if uses_moving_frame(env_config.observation_strat)
        control_shift_strategy = MovingFrameControlShift()
    else
        control_shift_strategy = ZeroControlShift()
    end
    env = RDEEnv(
        params;
        dt = env_config.dt,
        observation_strat = deepcopy(env_config.observation_strat),
        action_strat = deepcopy(env_config.action_strat),
        reward_strat = deepcopy(env_config.reward_strat),
        reset_strategy = deepcopy(env_config.reset_strategy),
        context_strat = deepcopy(env_config.context_strat),
        control_shift_strategy = control_shift_strategy,
        spatial_kernel_width = spatial_kernel_width
    )
    _reset!(env)
    return env
end

function create_envs(env_config::EnvConfig, n_envs::Int; spatial_kernel_width::Int = 0)
    envs = [create_env(env_config; spatial_kernel_width = spatial_kernel_width) for _ in 1:n_envs]
    return envs
end

function make_train_env(env_config::EnvConfig, dril_config::DrillConfig; spatial_kernel_width::Int = 0)
    envs = create_envs(env_config, dril_config.n_envs; spatial_kernel_width = spatial_kernel_width)
    # _reset!.(envs)
    if env_config.observation_strat isa AbstractMultiAgentObservationStrategy
        dril_envs = MultiAgentRDEEnv.(envs)
        par_env = Drill.MultiAgentParallelEnv(dril_envs)
    else
        dril_envs = envs
        par_env = dril_config.parallel_type(dril_envs)
    end
    monitor_env = Drill.MonitorWrapperEnv(par_env)
    if dril_config.normalize_env
        env = Drill.NormalizeWrapperEnv(monitor_env, gamma = dril_config.algorithm.gamma)
    else
        env = monitor_env
    end
    return env
end

function make_eval_env(env_config::EnvConfig, dril_config::DrillConfig, train_env::AbstractParallelEnv; spatial_kernel_width::Int = 0)
    #only one parallel env for eval
    @reset dril_config.n_envs = 1
    env = make_train_env(env_config, dril_config; spatial_kernel_width = spatial_kernel_width)
    if dril_config.normalize_env
        Drill.sync_normalization_stats!(env, train_env)
    end
    return env
end

function maybe_update_progress!(::Nothing, ::AbstractString)
    return nothing
end

function maybe_update_progress!(task::ProgressTask, message::AbstractString)
    update!(task; message = String(message))
    return nothing
end

@kwdef mutable struct DrillTaskProgressCallback <: Drill.AbstractCallback
    task::ProgressTask
    current_step::Int = 0
    total_steps::Union{Int, Nothing} = nothing
end

function Drill.on_training_start(callback::DrillTaskProgressCallback, locals::Dict)
    total_steps = locals[:total_steps]
    env = locals[:env]
    @assert total_steps % DrillInterface.number_of_envs(env) == 0 "total_steps must be divisible by number of environments"
    callback.current_step = 0
    callback.total_steps = total_steps
    update!(callback.task; total_steps, message = "Training...")
    return true
end

function Drill.on_rollout_start(callback::DrillTaskProgressCallback, locals::Dict)
    update!(callback.task; message = "Rollout...")
    return true
end

function Drill.on_step(callback::DrillTaskProgressCallback, locals::Dict)
    env = locals[:env]
    n_envs = DrillInterface.number_of_envs(env)
    callback.current_step += n_envs
    total_steps = something(callback.total_steps, callback.current_step)
    update!(
        callback.task;
        step = callback.current_step,
        total_steps = total_steps,
        message = "Rollout...",
    )
    return true
end

function Drill.on_rollout_end(callback::DrillTaskProgressCallback, locals::Dict)
    update!(callback.task; message = "Learning...")
    return true
end

function create_dril_progress_callback(task::ProgressTask)
    return DrillTaskProgressCallback(task = task)
end

function load_checkpoint(path::String)
    try
        return load(path)
    catch
        return JLD2.load(path; typemap = create_injection_migration_typemap())
    end
end

function get_jl_policy(experiment::DrillExperiment, path::String)
    data = load_checkpoint(path)
    agent = data["agent"]
    train_env = data["train_env"]
    dril_config = experiment.dril_config
    if dril_config.normalize_env
        policy = Drill.extract_policy(agent, train_env)
    else
        policy = Drill.extract_policy(agent)
    end
    return policy
end

function get_agent(experiment::DrillExperiment, path::String)
    data = load(path)
    agent = data["agent"]
    return agent
end

function get_train_env(experiment::DrillExperiment, path::String)
    data = load(path)
    train_env = data["train_env"]
    return train_env
end

function filter_for_hparams(cfg::Dict)
    hparams = Dict{String, Any}()
    for (k, v) in cfg
        if !(v isa Number || v isa String || v isa Bool || v isa Nothing)
            hparams[k] = string(v)
        else
            hparams[k] = v
        end
    end
    return hparams
end

function setup_dril_config(
        algorithm::String,
        n_envs::Int,
        normalize_env::Bool,
        learn_config::DrillLearnConfig;
        parallel_type = BroadcastedParallelEnv,
        alg_kwargs = (;),
        layer_kwargs = (;),
        kwargs...
    )
    if algorithm == "PPO"
        alg = Drill.PPO(; alg_kwargs...)
        dril_config = DrillConfig(;
            algorithm = alg,
            n_envs,
            policy_config = LayerKwargs(layer_kwargs),
            normalize_env,
            parallel_type,
            learn_config
        )
    elseif algorithm == "SAC"
        ent_coef = Drill.AutoEntropyCoefficient(initial_value = alg_kwargs.ent_coef)
        alg_kwargs = @set alg_kwargs.ent_coef = ent_coef
        alg = Drill.SAC(; alg_kwargs...)
        dril_config = DrillConfig(;
            algorithm = alg,
            n_envs,
            policy_config = LayerKwargs(layer_kwargs),
            normalize_env,
            parallel_type,
            learn_config
        )
    else
        error("Unsupported algorithm: $algorithm")
    end
    return dril_config
end
