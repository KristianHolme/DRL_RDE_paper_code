const DEFAULT_CFG = Dict(
    "movingframe" => false,
    "context_strat" => "fixed",
    "reward_strat" => "stabilitytarget",
    "action_strat" => "linear",
    "tmax" => 500.0f0,
    "normalize_env" => true,
    "alg" => "PPO",
    "n_envs" => 8,
    "log_std_init" => 0.0f0,
    "ent_coef" => 0.0f0,
    "init_ent_coef" => 0.0f0,
    "layer_size" => 128,
)
const MANDATORY_CFG_KEYS = [
    "n_sections",
    "n_agents",
    "experiment_name",
    "experiment_seed",
    "dt",
]
const EVAL_REPETITIONS_PER_CONFIG = 4

function get_or_default(cfg::Dict, key::String)
    if haskey(cfg, key)
        return cfg[key]
    else
        return DEFAULT_CFG[key]
    end
end

"""
    experiment_savename(c, suffix::AbstractString = ""; kwargs...)

Return a short name for saving (e.g. paths, filenames), matching DrWatson's `savename`.
If `c` contains a non-empty `"name_keys"` or `:name_keys` entry, only those keys are
used for the name (via DrWatson's `accesses`), giving shorter, filesystem-safe paths.
Otherwise behaves as `savename(c, suffix; kwargs...)`. The full config is still
stored inside saved files; this only affects the path string.
"""
function experiment_savename(c, suffix::AbstractString = ""; kwargs...)
    name_keys = get(c, "name_keys", get(c, :name_keys, nothing))
    if name_keys !== nothing && !isempty(name_keys)
        key_type = keytype(c)
        accesses = key_type === Symbol ? Tuple(Symbol.(name_keys)) : Tuple(string.(name_keys))
        return savename(c, suffix; accesses = accesses, kwargs...)
    else
        return savename(c, suffix; kwargs...)
    end
end

function verify_cfg(cfg::Dict)
    for key in MANDATORY_CFG_KEYS
        @assert haskey(cfg, key) "$key is required"
    end
    @assert cfg["n_sections"] > 0 "n_sections must be greater than 0"
    @assert cfg["n_agents"] > 0 "n_agents must be greater than 0"
    if cfg["n_sections"] != cfg["n_agents"]
        @assert cfg["n_agents"] == 1 "n_agents must be 1 if n_sections is not equal to n_agents"
    end
    return true
end

function get_abbreviation(cfg::Dict)
    if cfg["n_sections"] > 1 && cfg["n_agents"] > 1
        base = "MA"
    elseif cfg["n_sections"] > 1 && cfg["n_agents"] == 1
        base = "SAS"
    elseif cfg["n_sections"] == 1 && cfg["n_agents"] == 1
        base = "SAU"
    else
        error("Invalid cfg: $cfg")
    end
    if cfg["movingframe"]
        base = base * "MF"
    end
    return base
end

function get_action_strat_SAU(cfg::Dict)
    action_strat = get_or_default(cfg, "action_strat")
    if action_strat == "linear"
        return LinearScalarPressureAction()
    elseif action_strat == "pid"
        return PIDAction()
    elseif action_strat == "centered"
        return ScalarPressureAction()
    elseif action_strat == "linear-dt"
        return LinearScalarPressureWithDtAction(;
            dt_min = get(cfg, "dt_min", 0.1f0),
            dt_max = get(cfg, "dt_max", 20.0f0),
        )
    elseif action_strat == "multistep"
        return MultiStepPressureAction(;
            n_sections = 1,
            n_steps = Int(get(cfg, "n_multisteps", 4)),
        )
    elseif action_strat == "linear_multistep"
        return LinearMultiStepPressureAction(;
            n_sections = 1,
            n_steps = Int(get(cfg, "n_multisteps", 4)),
        )
    elseif action_strat == "target_linear"
        return TargetSafeLinearScalarPressureAction()
    elseif action_strat == "target_linear_multistep"
        return TargetSafeLinearMultiStepPressureAction(;
            n_sections = 1,
            n_steps = Int(get(cfg, "n_multisteps", 2)),
        )
    else
        error("Invalid action_strat for SAU: $action_strat")
    end
end
function get_action_strat_SAS(cfg::Dict)
    act_strat = get_or_default(cfg, "action_strat")
    if act_strat == "linear"
        return LinearVectorPressureAction(n_sections = cfg["n_sections"])
    elseif act_strat == "centered"
        return VectorPressureAction(n_sections = cfg["n_sections"])
    else
        error("Invalid action_strat for SAS: $act_strat")
    end
end
function get_action_strat_MA(cfg::Dict)
    act_strat = get_or_default(cfg, "action_strat")
    if act_strat == "linear"
        return LinearVectorPressureAction(n_sections = cfg["n_sections"])
    elseif act_strat == "centered"
        return VectorPressureAction(n_sections = cfg["n_sections"])
    else
        error("Invalid action_strat for MA: $act_strat")
    end
end

function get_obs_strat_SAU(cfg::Dict)
    obs = get(cfg, "obs", nothing)
    if obs == "sectioned_mf"
        return SectionedStateMovingFrameObservation()
    elseif obs == "fourier8"
        return FourierMagnitudeObservation(n_modes = 8)
    elseif obs == "fourier8_up"
        return FourierMagnitudePressureShockObservation(n_modes = 8)
    elseif obs == "fourier8_acthist"
        return FourierMagnitudeActionHistoryObservation(n_modes = 8, history_length = 8)
    elseif obs == "up_hist_rew"
        return PressureHistoryWaveRewardObservation(history_length = 8)
    elseif obs !== nothing
        error("Invalid obs for SAU: $obs")
    elseif get(cfg, "p_hist", false)
        return SectionedStateWithPressureHistoryObservation()
    else
        return SectionedStateObservation()
    end
end


function get_obs_strat_SAS(cfg::Dict)
    movingframe = cfg["movingframe"]
    if movingframe
        return SectionedStateMovingFrameObservation()
    else
        return SectionedStateObservation()
    end
end

function get_obs_strat_MA(cfg::Dict)
    movingframe = cfg["movingframe"]
    n_agents = cfg["n_agents"]
    use_pe = get(cfg, "positional_encoding", false)
    base = if movingframe
        MultiCenteredMovingFrameObservation(n_sections = n_agents)
    else
        MultiCenteredObservation(n_sections = n_agents)
    end
    if use_pe
        scale = get(cfg, "pe_scale", 0.1f0)
        return MultiAgentObservationWithSinusoidalPE(inner = base, scale = scale)
    else
        return base
    end
end

function get_context_strat(cfg::Dict)
    context_strat = get_or_default(cfg, "context_strat")
    if context_strat == "fixed"
        @assert haskey(cfg, "target_shock_count") "target_shock_count is required for fixed goal strategy"
        return FixedTargetShockCountStrategy(cfg["target_shock_count"])
    elseif context_strat == "random"
        return RandomTargetShockCountStrategy()
    else
        error("Invalid context_strat: $context_strat")
    end
end

function get_reward_strat_SAU(cfg::Dict)
    reward_strat = get_or_default(cfg, "reward_strat")
    if reward_strat == "stabilitytarget"
        return StabilityTargetReward()
    elseif reward_strat == "stabilitytarget_tv"
        return MultiplicativeReward(StabilityTargetReward(), TargetGatedPressureTVReward())
    elseif reward_strat == "stabilitytarget_cost"
        return StepCostReward(
            StabilityTargetReward();
            cost = Float32(get(cfg, "step_cost", 0.5f0)),
        )
    elseif reward_strat == "stabilitytarget_tv_cost"
        return StepCostReward(
            MultiplicativeReward(StabilityTargetReward(), TargetGatedPressureTVReward());
            cost = Float32(get(cfg, "step_cost", 0.5f0)),
        )
    elseif reward_strat == "stabilitytarget_lock"
        return DenseLockReward(
            StabilityTargetReward();
            bonus = Float32(get(cfg, "lock_bonus", 150.0f0)),
        )
    elseif reward_strat == "stabilitytarget_tv_lock"
        return DenseLockReward(
            MultiplicativeReward(StabilityTargetReward(), TargetGatedPressureTVReward());
            bonus = Float32(get(cfg, "lock_bonus", 150.0f0)),
        )
    elseif reward_strat == "target_lock"
        return DenseLockReward(
            TargetMatchReward();
            bonus = Float32(get(cfg, "lock_bonus", 150.0f0)),
        )
    elseif reward_strat == "target_stab_lock"
        return DenseLockReward(
            MultiplicativeReward(TargetMatchReward(), StabilityReward(1.0f0));
            bonus = Float32(get(cfg, "lock_bonus", 150.0f0)),
        )
    elseif reward_strat == "stabilitytarget_dt"
        return DtWeightedReward(StabilityTargetReward())
    elseif reward_strat == "transition"
        return TransitionBasedReward()
    elseif reward_strat == "periodminimum"
        return PeriodMinimumReward()
    else
        error("Invalid reward_strat: $reward_strat")
    end
end

get_reward_strat_SAS(cfg::Dict) = get_reward_strat_SAU(cfg)

function get_reward_strat_MA(cfg::Dict)
    reward_strat = get_or_default(cfg, "reward_strat")
    n_sections = cfg["n_sections"]
    if reward_strat == "periodminimum"
        reward = MultiSectionPeriodMinimumReward(n_sections = n_sections)
    elseif reward_strat == "transition"
        reward = ScalarToVectorReward(TransitionBasedReward(), n_sections)
    elseif reward_strat == "stabilitytarget"
        reward = ScalarToVectorReward(StabilityTargetReward(), n_sections)
    else
        error("Invalid reward_strat: $reward_strat")
    end
    return reward
end

function get_strats(cfg::Dict)
    abbreviation = get_abbreviation(cfg)
    cfg["abbreviation"] = abbreviation
    if occursin("SAU", abbreviation)
        action_strat = get_action_strat_SAU(cfg)
        observation_strat = get_obs_strat_SAU(cfg)
        reward_strat = get_reward_strat_SAU(cfg)
    elseif occursin("SAS", abbreviation)
        action_strat = get_action_strat_SAS(cfg)
        observation_strat = get_obs_strat_SAS(cfg)
        reward_strat = get_reward_strat_SAS(cfg)
    elseif occursin("MA", abbreviation)
        action_strat = get_action_strat_MA(cfg)
        observation_strat = get_obs_strat_MA(cfg)
        reward_strat = get_reward_strat_MA(cfg)
    end
    context_strat = get_context_strat(cfg)
    return (; action_strat, observation_strat, context_strat, reward_strat)
end

function get_alg_kwargs(cfg::Dict, alg::Type{<:Drill.AbstractAlgorithm})
    fns = fieldnames(alg)
    cfg_keys = keys(cfg) .|> Symbol
    ppo_keys = intersect(fns, cfg_keys)
    kwargs = (; (key => get(cfg, String(key), nothing) for key in ppo_keys)...)
    # Drill.PPO takes `target_kl`, but the stored field is `target_kl_strategy`.
    if alg <: Drill.PPO && haskey(cfg, "target_kl")
        kwargs = merge(kwargs, (; target_kl = cfg["target_kl"]))
    end
    return kwargs
end

function get_alg_kwargs(cfg::Dict, algorithm::String)
    if algorithm == "PPO"
        return get_alg_kwargs(cfg, Drill.PPO)
    elseif algorithm == "SAC"
        return get_alg_kwargs(cfg, Drill.SAC)
    else
        return error("Invalid algorithm: $algorithm")
    end
end

"""
    get_train_reset_strategy(cfg)

Training start-shock mix. Does not encode a `u_p` schedule.

- `"random"` (default): `RandomShock` (includes already-on-target).
- `"eval_starts"`: cycle the off-target starts used at eval.
- `"start1_heavy"`: same off-target starts, but 1-shock three times as often.
"""
function get_train_reset_strategy(cfg::Dict)
    reset_strat = String(get(cfg, "train_reset", "random"))
    target = Int(get(cfg, "target_shock_count", 2))
    if reset_strat == "random"
        return ShiftReset(RandomShock())
    elseif reset_strat == "eval_starts"
        return ShiftReset(CycleShockReset(shocks = setdiff(1:4, target)))
    elseif reset_strat == "start1_heavy"
        target == 1 && error("start1_heavy requires target_shock_count ≠ 1")
        others = filter(!=(1), setdiff(1:4, target))
        shocks = vcat(fill(1, 3), others)
        return ShiftReset(CycleShockReset(shocks = shocks))
    else
        error("Invalid train_reset: $reset_strat")
    end
end

function get_eval_config(env_config::RDEEnvConfig, cfg::Dict)
    context_strat = env_config.context_strat
    if context_strat isa FixedTargetShockCountStrategy
        target_shock_count = context_strat.target_shock_count
        eval_env_config = @set env_config.reset_strategy = ShiftReset(
            CycleShockReset(shocks = setdiff(1:4, target_shock_count))
        )
        n_eval_episodes = 3 * EVAL_REPETITIONS_PER_CONFIG * cfg["n_agents"]
    elseif context_strat isa RandomTargetShockCountStrategy
        eval_env_config = @set env_config.reset_strategy = ShiftReset(
            EvalCycleShockReset(EVAL_REPETITIONS_PER_CONFIG)
        )
        eval_context_strat = EvalCycleTargetShockCountStrategy(; repetitions_per_config = EVAL_REPETITIONS_PER_CONFIG)
        eval_env_config = @set eval_env_config.context_strat = eval_context_strat
        n_eval_episodes = 3 * EVAL_REPETITIONS_PER_CONFIG * 4 * cfg["n_agents"]
    else
        error("Invalid context_strat: $context_strat")
    end
    eval_reward_strat = TransitionBasedReward(StabilityReward(1.0f0))
    if cfg["n_agents"] > 1
        eval_reward_strat = ScalarToVectorReward(eval_reward_strat, cfg["n_agents"])
    end
    eval_env_config = @set eval_env_config.reward_strat = eval_reward_strat
    eval_config = RDEEvalConfig(; env_config = eval_env_config, n_eval_episodes)
    return eval_config
end

function setup_experiments(
        sample_cfgs::Function,
        n_samples::Int,
        max_steps::Int,
        experiment_suffix::String = "",
        n_runs::Int = 1,
        sampling_seed::Int = 227,
    )
    rng = Random.default_rng()
    Random.seed!(rng, sampling_seed)
    cfg_list = sample_cfgs(rng, n_samples)
    if length(cfg_list) > n_samples
        cfg_list = cfg_list[1:n_samples]
    end
    @assert all(verify_cfg, cfg_list) "Invalid configs"

    experiments = Tuple{DrillExperiment, Dict}[]
    for cfg in cfg_list
        cfg["max_steps"] = max_steps
        dt = cfg["dt"]
        tmax = get_or_default(cfg, "tmax")
        normalize_env = get_or_default(cfg, "normalize_env")
        algorithm = get_or_default(cfg, "alg")
        n_envs = get_or_default(cfg, "n_envs")
        log_std_init = get_or_default(cfg, "log_std_init")


        cfg["ent_coef"] = if algorithm == "PPO"
            get_or_default(cfg, "ent_coef")
        elseif algorithm == "SAC"
            get_or_default(cfg, "init_ent_coef")
        end

        if !haskey(cfg, "n_steps")
            cfg["n_steps"] = 2048 ÷ (n_envs * cfg["n_agents"])
        end

        alg_kwargs = get_alg_kwargs(cfg, algorithm)
        strats = get_strats(cfg)


        env_config = RDEEnvConfig(
            dt = dt,
            tmax = tmax,
            reset_strategy = get_train_reset_strategy(cfg),
            observation_strat = strats.observation_strat,
            action_strat = strats.action_strat,
            reward_strat = strats.reward_strat,
            context_strat = strats.context_strat
        )

        eval_config = get_eval_config(env_config, cfg)
        learn_config = DrillLearnConfig(max_timesteps = max_steps, verbose = 0)

        hidden_dims = fill(get_or_default(cfg, "layer_size"), 2)
        @assert length(hidden_dims) == 2 "hidden_dims should now have 2 elements"
        dril_config = setup_dril_config(
            algorithm,
            n_envs,
            normalize_env,
            learn_config;
            alg_kwargs,
            layer_kwargs = (; log_std_init, hidden_dims)
        )

        for run in 1:n_runs
            seed = cfg["experiment_seed"] + run - 1
            cfg["seed"] = seed
            cfg["run"] = run
            experiment = DrillExperiment(;
                study = cfg["experiment_name"] * experiment_suffix,
                run = run,
                seed = seed,
                env_config = deepcopy(env_config),
                eval_config = deepcopy(eval_config),
                dril_config = deepcopy(dril_config)
            )
            push!(experiments, (experiment, deepcopy(cfg)))
        end
    end
    return experiments
end
