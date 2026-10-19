abstract type AbstractExperiment end

abstract type AbstractRLConfig end

abstract type LearnConfig end

abstract type EnvConfig end
abstract type EvalConfig end

@kwdef struct RDEEnvConfig{
        T <: AbstractFloat,
        RE <: AbstractReset,
        O <: AbstractObservationStrategy,
        A <: AbstractActionStrategy,
        RW <: AbstractRewardStrategy,
        CS <: AbstractShockCountContextStrategy,
    } <: EnvConfig
    dt::T = 1.0f0
    tmax::T = 200.0f0
    reset_strategy::RE = ShiftReset(RandomShock())
    observation_strat::O = SectionedStateObservation()
    action_strat::A = ScalarPressureAction()
    reward_strat::RW = PeriodMinimumReward()
    context_strat::CS = FixedTargetShockCountStrategy(2)
end

@kwdef struct RDEEvalConfig{T <: AbstractFloat} <: EvalConfig
    env_config::RDEEnvConfig{T} = RDEEnvConfig()
    n_eval_episodes::Int = 60
end

@kwdef struct BasicLearnConfig <: LearnConfig
    total_timesteps::Int = 100_000
    progress_bar::Bool = true
    log_interval::Int = 1
    reset_num_timesteps::Bool = false
end

@kwdef struct RLExperiment <: AbstractExperiment
    study::String
    run::Int
    seed::Int
    env_config::EnvConfig
    eval_config::EvalConfig
    sb_config::AbstractRLConfig
end

@kwdef struct DrillLearnConfig <: LearnConfig
    max_timesteps::Int = 100_000
    verbose::Int = 1 #1 is simple progress bar
end

abstract type PolicyConfig end

struct LayerKwargs <: PolicyConfig
    kwargs::NamedTuple
end

@kwdef struct DrillConfig{A <: Drill.AbstractAlgorithm, P <: Drill.AbstractParallelEnv} <: AbstractRLConfig
    algorithm::A = Drill.PPO()
    policy_config::PolicyConfig = LayerKwargs((;))
    n_envs::Int = 16
    normalize_env::Bool = true
    parallel_type::Type{P} = BroadcastedParallelEnv #maybe better with broadcasted parallel env for parallel runs
    learn_config::DrillLearnConfig = DrillLearnConfig()
end

@kwdef struct DrillExperiment <: AbstractExperiment
    study::String
    run::Int
    seed::Int
    env_config::EnvConfig
    eval_config::EvalConfig
    dril_config::DrillConfig
end
