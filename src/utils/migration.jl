# ==============================================================================
# JLD2 Migration Support for moved types from DRL_RDE_utils
# ==============================================================================

# Using JLD2's rconvert system for proper type migration
# This handles upgrading old struct versions to new ones

# Robust reconstruction for ShiftReset reset strategies
function JLD2.rconvert(::Type{ShiftReset}, nt::NamedTuple)
    inner_reset = get(nt, :reset_strategy, RandomShock())
    return ShiftReset(inner_reset)
end

# Convert old RDEEnvConfig structs that may have different field names/missing fields
function JLD2.rconvert(::Type{RDEEnvConfig{T}}, nt::NamedTuple) where {T}
    # Extract basic fields that should always exist
    dt = T(get(nt, :dt, 1.0))
    tmax = T(get(nt, :tmax, 200.0))

    # Handle different field names for observation strategy
    observation_strat = if haskey(nt, :observation_strat)
        nt.observation_strat
    elseif haskey(nt, :observables)
        nt.observables  # Use old field name if it exists
    else
        SectionedStateObservation(minisections = 32, target_shock_count = 3)  # Default
    end

    # Provide defaults for potentially missing fields
    reset_strategy = get(nt, :reset_strategy, ShiftReset(RandomShockOrCombination()))
    action_strat = get(nt, :action_strat, ScalarPressureAction())
    reward_strat = get(nt, :reward_strat, CompositeReward(target_shock_count = 3, lowest_action_magnitude_reward = 0.0f0))

    return RDEEnvConfig{T}(
        dt = dt,
        tmax = tmax,
        reset_strategy = reset_strategy,
        observation_strat = observation_strat,
        action_strat = action_strat,
        reward_strat = reward_strat
    )
end

# Convert old RDEEvalConfig structs
function JLD2.rconvert(::Type{RDEEvalConfig{T}}, nt::NamedTuple) where {T}
    # Convert the nested env_config if needed
    env_config = if haskey(nt, :env_config)
        env_cfg = nt.env_config
        # If it's already the right type, use as-is, otherwise it will be converted by JLD2
        isa(env_cfg, RDEEnvConfig) ? env_cfg : env_cfg
    else
        RDEEnvConfig{T}()  # Default
    end

    n_eval_episodes = get(nt, :n_eval_episodes, 60)

    return RDEEvalConfig{T}(
        env_config = env_config,
        n_eval_episodes = n_eval_episodes
    )
end

# Convert old DrillConfig structs
function JLD2.rconvert(::Type{DrillConfig{A, P}}, nt::NamedTuple) where {A, P}
    algorithm = get(nt, :algorithm, Drill.PPO())
    n_envs = get(nt, :n_envs, 16)
    log_std_init = get(nt, :log_std_init, 0.0f0)
    normalize_env = get(nt, :normalize_env, true)
    parallel_type = get(nt, :parallel_type, MultiThreadedParallelEnv)
    learn_config = get(nt, :learn_config, DrillLearnConfig())

    return DrillConfig{typeof(algorithm), typeof(parallel_type)}(
        algorithm = algorithm,
        n_envs = n_envs,
        log_std_init = log_std_init,
        normalize_env = normalize_env,
        parallel_type = parallel_type,
        learn_config = learn_config
    )
end

# Fallback for when we don't know the type parameters
function JLD2.rconvert(::Type{DrillConfig}, nt::NamedTuple)
    algorithm = get(nt, :algorithm, Drill.PPO())
    n_envs = get(nt, :n_envs, 16)
    log_std_init = get(nt, :log_std_init, 0.0f0)
    normalize_env = get(nt, :normalize_env, true)
    parallel_type = get(nt, :parallel_type, MultiThreadedParallelEnv)
    learn_config = get(nt, :learn_config, DrillLearnConfig())

    return DrillConfig(
        algorithm = algorithm,
        n_envs = n_envs,
        log_std_init = log_std_init,
        normalize_env = normalize_env,
        parallel_type = parallel_type,
        learn_config = learn_config
    )
end

# NOTE: JLD2.rconvert/Upgrade/default_typemap/JLDFile are non-public APIs used
# for legacy migration support.

# Legacy JLD2 typepath → Upgrade(typeof(Drill.PPO())) for old single-parameter PPO{T} blobs
function _legacy_ppo_jld2_upgrade()
    return JLD2.Upgrade(typeof(Drill.PPO()))
end

# Old PPO{T} (@kwdef) → new PPO{T,AS,CVS,KLS} via keyword constructor; pass-through for new-layout NT
function JLD2.rconvert(::Type{<:Drill.PPO}, nt::NamedTuple)
    if haskey(nt, :advantage_strategy)
        if fieldcount(typeof(Drill.PPO())) == 12
            return Drill.PPO(
                nt.gamma,
                nt.gae_lambda,
                nt.clip_range,
                nt.ent_coef,
                nt.vf_coef,
                nt.max_grad_norm,
                nt.advantage_strategy,
                nt.clip_vf_strategy,
                nt.target_kl_strategy,
                nt.n_steps,
                nt.batch_size,
                nt.epochs,
                nt.learning_rate,
            )
        end
        error(
            "JLD2 PPO: NamedTuple has advantage_strategy but Drill.PPO is pre-strategy (13 fields); upgrade Drill.jl to migrate.",
        )
    end
    return Drill.PPO(;
        gamma = get(nt, :gamma, 0.99f0),
        gae_lambda = get(nt, :gae_lambda, 0.95f0),
        clip_range = get(nt, :clip_range, 0.2f0),
        clip_range_vf = get(nt, :clip_range_vf, nothing),
        ent_coef = get(nt, :ent_coef, 0.0f0),
        vf_coef = get(nt, :vf_coef, 0.5f0),
        max_grad_norm = get(nt, :max_grad_norm, 0.5f0),
        target_kl = get(nt, :target_kl, nothing),
        normalize_advantage = get(nt, :normalize_advantage, true),
        n_steps = get(nt, :n_steps, 2048),
        batch_size = get(nt, :batch_size, 64),
        epochs = get(nt, :epochs, 10),
        learning_rate = get(nt, :learning_rate, 3.0f-4),
    )
end

# Pre-`cache` Agent checkpoints → current 12-field Agent with cache = nothing (when Drill has `cache`)
function JLD2.rconvert(::Type{T}, nt::NamedTuple) where {T <: Drill.Agent}
    if fieldcount(T) == 11
        return T(
            nt.layer,
            nt.algorithm,
            nt.action_adapter,
            nt.train_state,
            nt.optimizer_type,
            nt.stats_window,
            nt.logger,
            nt.verbose,
            nt.rng,
            nt.stats,
            nt.aux,
        )
    end
    cache = get(nt, :cache, nothing)
    return T(
        nt.layer,
        nt.algorithm,
        nt.action_adapter,
        nt.train_state,
        nt.optimizer_type,
        nt.stats_window,
        nt.logger,
        nt.verbose,
        nt.rng,
        nt.stats,
        nt.aux,
        cache,
    )
end

# When loading saved models, TBLoggerBackend often becomes JLD2.ReconstructedMutable,
# which Drill's convert(AbstractTrainingLogger, x) rejects. Upgrade to NoTrainingLogger.
function JLD2.rconvert(::Type{Drill.NoTrainingLogger}, nt::NamedTuple)
    return Drill.NoTrainingLogger()
end

# Convert old DrillLearnConfig structs
function JLD2.rconvert(::Type{DrillLearnConfig}, nt::NamedTuple)
    return DrillLearnConfig(
        max_timesteps = get(nt, :max_timesteps, 100_000),
        verbose = get(nt, :verbose, 1)
    )
end

# Convert old DrillExperiment structs
function JLD2.rconvert(::Type{DrillExperiment}, nt::NamedTuple)
    return DrillExperiment(
        study = get(nt, :study, ""),
        run = get(nt, :run, 1),
        seed = get(nt, :seed, 0),
        env_config = get(nt, :env_config, RDEEnvConfig()),
        eval_config = get(nt, :eval_config, RDEEvalConfig()),
        dril_config = get(nt, :dril_config, DrillConfig())
    )
end

# Convert old PIDExperiment structs
function JLD2.rconvert(::Type{PIDExperiment}, nt::NamedTuple)
    return PIDExperiment(
        study = nt.study,
        eval_config = nt.eval_config,
        pid_policy = nt.pid_policy
    )
end

# Convert old PIDTransitionExperiment structs
function JLD2.rconvert(::Type{PIDTransitionExperiment}, nt::NamedTuple)
    return PIDTransitionExperiment(
        study = nt.study,
        eval_config = nt.eval_config,
        pid_policy = nt.pid_policy,
        target_shock_count = nt.target_shock_count,
        init_shock_count = nt.init_shock_count
    )
end

# Convert old FVCache (control fields on the cache) to live u_p/s workspace.
function JLD2.rconvert(::Type{RDE.FVCache{T}}, nt::NamedTuple) where {T}
    N = Int(nt.N)
    u_p = if haskey(nt, :u_p)
        nt.u_p
    elseif haskey(nt, :u_p_current)
        u = nt.u_p_current
        u isa AbstractVector ? Vector{T}(u) : fill(T(u), N)
    else
        zeros(T, N)
    end
    s = if haskey(nt, :s)
        nt.s
    elseif haskey(nt, :s_current)
        sc = nt.s_current
        sc isa AbstractVector ? Vector{T}(sc) : fill(T(sc), N)
    else
        zeros(T, N)
    end
    return RDE.FVCache{T}(
        nt.u_xx,
        nt.λ_xx,
        nt.ωu,
        nt.ξu,
        nt.βu,
        nt.σ,
        nt.UL,
        nt.UR,
        nt.F̂,
        nt.adv,
        nt.dx,
        N,
        u_p,
        s,
    )
end

# Placeholder so JLD2 can read pre-injection ODEProblem blobs; RDEEnv rconvert rebuilds.
struct MigratedODEProblem end

function JLD2.rconvert(::Type{MigratedODEProblem}, ::NamedTuple)
    return MigratedODEProblem()
end

function _legacy_action_is_spatial(A::Type)
    return A <: AbstractVectorActionStrategy && hasfield(A, :n_sections)
end

function _inner_injection_type(::Type{T}, spatial::Bool) where {T <: AbstractFloat}
    S = RDE.CinftySmoother{T}
    if spatial
        return RDE.SpatialMultiStepPressureProfile{T, S}
    else
        return RDE.UniformMultiStepPressureProfile{T, S}
    end
end

function _legacy_injection_type(::Type{T}, Cshift::Type; spatial::Bool) where {T <: AbstractFloat}
    inner = _inner_injection_type(T, spatial)
    if Cshift <: RDE.ZeroControlShift
        return inner
    else
        return RDE.MovingFrameInjectionProfile{T, inner, Cshift}
    end
end

function _legacy_injection_type(::Type{T}, A::Type, Cshift::Type) where {T <: AbstractFloat}
    return _legacy_injection_type(T, Cshift; spatial = _legacy_action_is_spatial(A))
end

function _make_injection(
        ::Type{RDE.UniformMultiStepPressureProfile{T, S}},
        params::RDEParam{T},
        ::RDE.AbstractControlShift,
        τ::T,
    ) where {T <: AbstractFloat, S}
    return RDE.default_uniform_injection(params, τ)
end

function _make_injection(
        ::Type{RDE.SpatialMultiStepPressureProfile{T, S}},
        params::RDEParam{T},
        ::RDE.AbstractControlShift,
        τ::T,
    ) where {T <: AbstractFloat, S}
    return RDE.default_spatial_injection(params, τ, 0)
end

function _make_injection(
        ::Type{RDE.MovingFrameInjectionProfile{T, I, S}},
        params::RDEParam{T},
        shift::RDE.AbstractControlShift,
        τ::T,
    ) where {T <: AbstractFloat, I, S}
    inner = _make_injection(I, params, RDE.ZeroControlShift(), τ)
    return RDE.wrap_injection(inner, shift, params)
end

function _legacy_control_shift(nt::NamedTuple)
    if haskey(nt, :control_shift_strategy)
        return nt.control_shift_strategy
    elseif haskey(nt, :injection)
        return RDE.control_shift(nt.injection)
    else
        return RDE.ZeroControlShift()
    end
end

function JLD2.rconvert(::Type{RP}, nt::NamedTuple) where {RP <: RDE.RDEProblem}
    params = nt.params
    T = typeof(params).parameters[1]
    τ = T(1)
    Inj = RP.parameters[4]
    inj = if haskey(nt, :injection) && nt.injection isa Inj
        nt.injection
    else
        shift = _legacy_control_shift(nt)
        _make_injection(Inj, params, shift, τ)
    end
    return RP(
        params,
        nt.u0,
        nt.λ0,
        nt.x,
        nt.reset_strategy,
        nt.sol,
        nt.method,
        inj,
    )
end

function JLD2.rconvert(::Type{E}, nt::NamedTuple) where {E <: RDEEnv}
    T = E.parameters[1]
    Inj = E.parameters[10]
    τ = T(nt.τ_smooth)
    old_prob = nt.prob
    params = old_prob.params
    shift = RDE.control_shift(old_prob.injection)
    inj = _make_injection(Inj, params, shift, τ)
    M = typeof(old_prob.method)
    RS = typeof(old_prob.reset_strategy)
    new_prob = RDE.RDEProblem{T, M, RS, Inj}(
        params,
        old_prob.u0,
        old_prob.λ0,
        old_prob.x,
        old_prob.reset_strategy,
        old_prob.sol,
        old_prob.method,
        inj,
    )
    ode_problem = SciMLBase.ODEProblem{true, SciMLBase.FullSpecialize}(
        RDE.RDE_RHS!,
        nt.state,
        (zero(T), T(nt.dt)),
        new_prob,
    )
    N = length(new_prob.x)
    reward_cache = initialize_cache(nt.reward_strat, N, T)
    action_cache = initialize_cache(nt.action_strat, N, T)
    observation_cache = initialize_cache(nt.observation_strat, N, T)
    context = initialize_cache(nt.context_strat, N, T)
    cache = RDEEnvCache{T, typeof(reward_cache), typeof(action_cache), typeof(observation_cache), typeof(context)}(
        N; reward_cache, action_cache, observation_cache, context
    )
    return E(
        new_prob,
        nt.state,
        nt.observation,
        nt.dt,
        nt.t,
        nt.done,
        nt.truncated,
        nt.terminated,
        nt.reward,
        nt.smax,
        nt.u_pmax,
        τ,
        cache,
        nt.action_strat,
        nt.observation_strat,
        nt.reward_strat,
        nt.context_strat,
        nt.verbose,
        nt.info,
        nt.steps_taken,
        ode_problem,
    )
end

function _is_legacy_control_shift(P)
    return P isa Type && P <: RDE.AbstractControlShift && !(P <: RDE.AbstractInjectionProfile)
end

function _upgrade_legacy_rdeenv(params::Vector)
    T, A, Cshift = params[1], params[2], params[10]
    inj = _legacy_injection_type(T, A, Cshift)
    new_params = copy(params)
    new_params[10] = inj
    return JLD2.Upgrade(Core.apply_type(RDEEnv, new_params...))
end

function _upgrade_legacy_rdeproblem(params::Vector)
    T, M, R, Cshift = params[1], params[2], params[3], params[4]
    inj = _legacy_injection_type(T, Cshift; spatial = false)
    return JLD2.Upgrade(Core.apply_type(RDE.RDEProblem, T, M, R, inj))
end

"""
    _jld2_is_legacy_single_typeparam_ppo(typepath, params)

JLD2 stores `Drill.PPO{Float32}` as path `\"Drill.PPO\"` with `params == [Float32]`.
That shape matches old `PPO{T}` checkpoints; new `PPO{T,AS,CVS,KLS}` uses four parameters.
"""
function _jld2_is_legacy_single_typeparam_ppo(typepath::String, params::Vector)
    if typepath != "Drill.PPO" && typepath != "DRiL.PPO"
        return false
    end
    if length(params) != 1
        return false
    end
    p = params[1]
    return p isa Type && p <: AbstractFloat
end

"""
    _jld2_is_drill_agent_six_params(typepath, params)

JLD2 stores `Drill.Agent{L,...}` as path `\"Drill.Agent\"` with six type parameters.
Used to route loads through `Upgrade` + `rconvert` so pre-`cache` blobs get `cache = nothing`.
"""
function _jld2_is_drill_agent_six_params(typepath::String, params::Vector)
    if typepath != "Drill.Agent" && typepath != "DRiL.Agent"
        return false
    end
    if length(params) != 6
        return false
    end
    return all(p -> p isa Type, params)
end

# Typemap for old DRL_RDE_utils types and legacy single-param `PPO{T}` (via Upgrade + rconvert).
function create_migration_typemap()
    ppo_u = _legacy_ppo_jld2_upgrade()
    static = Dict{String, Any}(
        "DRL_RDE_utils.RDEEnvConfig" => JLD2.Upgrade(RDEEnvConfig{Float32}),
        "DRL_RDE_utils.RDEEvalConfig" => JLD2.Upgrade(RDEEvalConfig{Float32}),
        "DRL_RDE_utils.DrillExperiment" => JLD2.Upgrade(DrillExperiment),
        "DRL_RDE_utils.DrillConfig" => JLD2.Upgrade(DrillConfig),
        "DRL_RDE_utils.DrillLearnConfig" => JLD2.Upgrade(DrillLearnConfig),
        "DRL_RDE_utils.PIDExperiment" => JLD2.Upgrade(PIDExperiment),
        "DRL_RDE_utils.PIDTransitionExperiment" => JLD2.Upgrade(PIDTransitionExperiment),
        "RDE.FVCache" => JLD2.Upgrade(RDE.FVCache{Float32}),
        "Drill.PPO{Float32}" => ppo_u,
        "Drill.PPO{Float64}" => ppo_u,
        "DRiL.PPO{Float32}" => ppo_u,
        "DRiL.PPO{Float64}" => ppo_u,
    )
    return function structs_migration_typemap(f::JLD2.JLDFile, typepath::String, params::Vector)
        if _jld2_is_legacy_single_typeparam_ppo(typepath, params)
            return ppo_u
        end
        if _jld2_is_drill_agent_six_params(typepath, params)
            return JLD2.Upgrade(Core.apply_type(Drill.Agent, params...))
        end
        if haskey(static, typepath)
            return static[typepath]
        end
        return JLD2.default_typemap(f, typepath, params)
    end
end

function _is_rdeenv_typepath(typepath::String)
    return typepath == "RDE_Env.RDEEnv" || typepath == "RDEEnv"
end

function _is_rdeproblem_typepath(typepath::String)
    return typepath == "RDE.RDEProblem"
end

function _is_odeproblem_typepath(typepath::String)
    return endswith(typepath, ".ODEProblem") || typepath == "ODEProblem"
end

function _odeproblem_needs_dummy(params::Vector)
    for p in params
        if p isa DataType && p <: RDE.RDEProblem && length(p.parameters) == 4 &&
                _is_legacy_control_shift(p.parameters[4])
            return true
        end
    end
    return false
end

"""
    create_injection_migration_typemap()

JLD2 typemap for pre-injection-profile checkpoints: `RDEEnv` / `RDEProblem` last
parameter used to be a control-shift type, and `FVCache` stored live `u_p`/`s` on
the cache. Also upgrades legacy DRL_RDE_utils / PPO / Agent blobs via
`create_migration_typemap`.
"""
function create_injection_migration_typemap()
    structs = create_migration_typemap()
    return function injection_migration_typemap(f::JLD2.JLDFile, typepath::String, params::Vector)
        mapped_path = remap_dril_typepath(typepath)
        if _is_rdeenv_typepath(mapped_path) &&
                length(params) == 10 &&
                all(p -> p isa Type, params) &&
                _is_legacy_control_shift(params[10])
            return _upgrade_legacy_rdeenv(params)
        end
        if _is_rdeproblem_typepath(mapped_path) &&
                length(params) == 4 &&
                all(p -> p isa Type, params) &&
                _is_legacy_control_shift(params[4])
            return _upgrade_legacy_rdeproblem(params)
        end
        if mapped_path == "RDE.FVCache" && length(params) == 1 && params[1] isa Type
            return JLD2.Upgrade(RDE.FVCache{params[1]})
        end
        if _is_odeproblem_typepath(mapped_path) && _odeproblem_needs_dummy(params)
            return JLD2.Upgrade(MigratedODEProblem)
        end
        return structs(f, mapped_path, params)
    end
end

const DRIL_EXTENSION_TYPEPATH_RENAMES = (
    "DRiL_TensorBoardLoggerExt." => "Drill_TensorBoardLoggerExt.",
    "DRiL_DearDiaryExt." => "Drill_DearDiaryExt.",
    "DRiL_WandbExt." => "Drill_WandbExt.",
)

function remap_dril_typepath(typepath::String)
    for (old_prefix, new_prefix) in DRIL_EXTENSION_TYPEPATH_RENAMES
        if startswith(typepath, old_prefix)
            suffix = typepath[(ncodeunits(old_prefix) + 1):end]
            return new_prefix * suffix
        end
    end

    # Replace DRiL with Drill anywhere in the path (handles DRiL., DRiLConfig, etc.)
    if occursin("DRiL", typepath)
        return replace(typepath, "DRiL" => "Drill")
    end

    return typepath
end

function create_dril_rename_typemap(; on_remap::Function = (old_typepath, new_typepath) -> nothing)
    typemap = function (f::JLD2.JLDFile, typepath::String, params::Vector)
        mapped_typepath = remap_dril_typepath(typepath)
        if mapped_typepath != typepath
            on_remap(typepath, mapped_typepath)
            return JLD2.default_typemap(f, mapped_typepath, params)
        end
        return JLD2.default_typemap(f, typepath, params)
    end
    return typemap
end

"""
    reevaluate_and_save(experiment_name; reward_strategy = TransitionBasedReward(StabilityReward(1.0f0), reward_threshold= 0.9f0))

Reevaluate the results using transitionbasedreward with reward_strategy and save the new results.
Results are saved in a new experiment directory with the name "experiment_name_reevaluated".
Logs are copied from the old experiment directory.
"""
function reevaluate_and_save(experiment_name; reward_strategy = TransitionBasedReward(StabilityReward(1.0f0), reward_threshold = 0.9f0))
    results_dir = datadir("studies", experiment_name, "results")
    models_dir = datadir("studies", experiment_name, "models")
    old_logs_dir = datadir("studies", experiment_name, "logs")
    new_experiment_name = experiment_name * "_reevaluated"
    new_results_dir = datadir("studies", new_experiment_name, "results")
    new_models_dir = datadir("studies", new_experiment_name, "models")
    logs_dir = datadir("studies", new_experiment_name, "logs")
    mkpath(new_results_dir)
    # mkpath(new_models_dir)
    # mkpath(logs_dir)
    cp(models_dir, new_models_dir, force = true)
    cp(old_logs_dir, logs_dir, force = true)
    @showprogress showspeed = true desc = "reevaluating and saving" @threads for file in readdir(results_dir)
        result = load(joinpath(results_dir, file))
        old_eval_rew_mean = result["eval/ep_rew_mean"]
        old_eval_rew_std = result["eval/ep_rew_std"]
        old_eval_len_mean = result["eval/ep_len_mean"]
        old_eval_len_std = result["eval/ep_len_std"]
        experiment = result["experiment"]
        old_reward_strategy = experiment.eval_config.env_config.reward_strat
        new_reward_strategy = if old_reward_strategy isa TransitionBasedReward
            reward_strategy
        elseif old_reward_strategy isa ScalarToVectorReward
            @assert old_reward_strategy.wrapped_reward isa TransitionBasedReward "Wrapped reward should be a TransitionBasedReward"
            ScalarToVectorReward(reward_strategy, old_reward_strategy.n)
        else
            error("Invalid reward strategy: $old_reward_strategy")
        end
        @reset experiment.eval_config.env_config.reward_strat = new_reward_strategy
        result["experiment"] = experiment
        agent = get_agent(experiment, joinpath(models_dir, basename(result["model_path"])))
        train_env = get_train_env(experiment, joinpath(models_dir, basename(result["model_path"])))
        eval_env = make_eval_env(experiment.eval_config.env_config, experiment.dril_config, train_env)
        mean_reward, std_reward, mean_length, std_length = Drill.evaluate_agent(agent, eval_env; n_eval_episodes = experiment.eval_config.n_eval_episodes)
        result["eval/ep_rew_mean"] = mean_reward
        result["eval/ep_rew_std"] = std_reward
        result["eval/ep_len_mean"] = mean_length
        result["eval/ep_len_std"] = std_length
        result["eval/ep_rew_mean_old"] = old_eval_rew_mean
        result["eval/ep_rew_std_old"] = old_eval_rew_std
        result["eval/ep_len_mean_old"] = old_eval_len_mean
        result["eval/ep_len_std_old"] = old_eval_len_std
        save(joinpath(new_results_dir, file), result)
    end
    return nothing
end

function shiftreset_typemap(f::JLD2.JLDFile, typepath::String, params::Vector)
    if occursin("ShiftReset", typepath)
        return JLD2.Upgrade(RDE.ShiftReset)
    else
        return JLD2.default_typemap(f, typepath, params)
    end
end
"""
    fix_shiftreset_loading(experiment_name)

Fix the shiftreset loading by converting the results and models to the new types.
"""
function fix_shiftreset_loading(experiment_name)
    results_dir = datadir("studies", experiment_name, "results")
    models_dir = datadir("studies", experiment_name, "models")
    old_results_dir = datadir("studies", experiment_name, "results_old")
    old_models_dir = datadir("studies", experiment_name, "models_old")
    mv(results_dir, old_results_dir, force = true)
    mv(models_dir, old_models_dir, force = true)

    @showprogress desc = "converting results" @threads for file in readdir(old_results_dir)
        l = load(joinpath(old_results_dir, file), typemap = shiftreset_typemap)
        safesave(joinpath(results_dir, file), l)
    end

    @showprogress desc = "converting models" @threads for file in readdir(old_models_dir)
        l = load(joinpath(old_models_dir, file), typemap = shiftreset_typemap)
        safesave(joinpath(models_dir, file), l)
    end
    return nothing
end

"""
    separate_random_fixed_results(experiment_name)

Separate the results into random and fixed target results.
"""
function separate_random_fixed_results(experiment_name)
    results_dir = datadir("studies", experiment_name, "results")
    models_dir = datadir("studies", experiment_name, "models")
    logs_dir = datadir("studies", experiment_name, "logs")
    random_target_experiment_name = experiment_name * "_random"
    fixed_target_experiment_name = experiment_name * "_fixed"
    random_target_results_dir = datadir("studies", random_target_experiment_name, "results")
    fixed_target_results_dir = datadir("studies", fixed_target_experiment_name, "results")
    random_target_models_dir = datadir("studies", random_target_experiment_name, "models")
    fixed_target_models_dir = datadir("studies", fixed_target_experiment_name, "models")
    random_target_logs_dir = datadir("studies", random_target_experiment_name, "logs")
    fixed_target_logs_dir = datadir("studies", fixed_target_experiment_name, "logs")

    @info "Creating directories for random/fixed goals"
    mkpath(random_target_results_dir)
    mkpath(fixed_target_results_dir)
    mkpath(random_target_models_dir)
    mkpath(fixed_target_models_dir)
    mkpath(random_target_logs_dir)
    mkpath(fixed_target_logs_dir)

    @showprogress desc = "separating random/fixed goals" @threads for file in readdir(results_dir)
        if occursin("context_strat=random", file)
            tgt_results_dir = random_target_results_dir
            tgt_models_dir = random_target_models_dir
            tgt_logs_dir = random_target_logs_dir
        elseif occursin("context_strat=fixed", file)
            tgt_results_dir = fixed_target_results_dir
            tgt_models_dir = fixed_target_models_dir
            tgt_logs_dir = fixed_target_logs_dir
        else
            error("Invalid context_strat in file: $file")
        end

        # Copy result and model files
        cp(joinpath(results_dir, file), joinpath(tgt_results_dir, file), force = true)
        cp(joinpath(models_dir, file), joinpath(tgt_models_dir, file), force = true)

        # Handle logs: strip ".jld2" to get log folder name, then copy folder recursively
        log_folder = replace(file, r"\.jld2$" => "")
        src_log_folder = joinpath(logs_dir, log_folder)
        dest_log_folder = joinpath(tgt_logs_dir, log_folder)
        if isdir(src_log_folder)
            cp(src_log_folder, dest_log_folder; force = true)
        else
            @warn "Log folder $src_log_folder does not exist for file $file"
        end
    end
    return nothing
end
