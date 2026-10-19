@testitem "Single Agent Mode Training" begin
    using DRL_RDE
    using Accessors
    using Random
    using Dates
    using Drill
    using DrillInterface
    using Enzyme

    # Create a minimal single agent configuration
    study = "test_single_agent"
    run_num = 1
    seed = 123
    target_shock_count = 2

    # Simple reward configuration
    periodminreward = PeriodMinimumReward(
        lowest_action_magnitude_reward = 0.5f0,
        weights = [1.0f0, 1.0f0, 1.0f0, 1.0f0]
    )

    # Environment config for single agent
    env_config = RDEEnvConfig(
        dt = 1.0f0,
        tmax = 50.0f0,  # Short for testing
        reset_strategy = ShiftReset(RandomShock()),
        observation_strat = SectionedStateObservation(
            minisections = 8,  # Smaller for testing
        ),
        action_strat = ScalarPressureAction(),
        reward_strat = periodminreward
    )

    # Eval config
    eval_env_config = @set env_config.reset_strategy = ShiftReset(CycleShockReset(shocks = setdiff(1:4, target_shock_count)))
    eval_reward_strat = TransitionBasedReward(PeriodicityReward())
    eval_env_config = @set eval_env_config.reward_strat = eval_reward_strat
    eval_config = RDEEvalConfig(env_config = eval_env_config, n_eval_episodes = 6)  # Minimal evaluation

    # Learning config with minimal timesteps
    learn_config = DrillLearnConfig(
        max_timesteps = 1000,  # Very short for testing
        verbose = 0,
    )

    # Drill config
    dril_config = DrillConfig(
        algorithm = Drill.PPO(; n_steps = 32, batch_size = 16, epochs = 2, learning_rate = 3.0f-4),
        n_envs = 2,  # Minimal envs
        policy_config = LayerKwargs((; log_std_init = 0.0f0)),
        normalize_env = true,
        parallel_type = BroadcastedParallelEnv,
        learn_config = learn_config
    )

    # Create experiment
    experiment = DrillExperiment(
        study = study,
        run = run_num,
        seed = seed,
        env_config = env_config,
        eval_config = eval_config,
        dril_config = dril_config
    )

    # Create config dict
    config = Dict(
        "weights" => [1.0f0, 1.0f0, 1.0f0, 1.0f0],
        "run" => run_num,
        "lowest_action_magnitude_reward" => 0.5f0,
        "seed" => seed,
        "target_shock_count" => target_shock_count,
        "normalize_env" => true,
        "reward" => "periodminimum"
    )

    result = single_run(config, experiment; ignore_existing = true, save = false, logging = false)
    @test result isa Dict
    @test haskey(result, "eval/ep_rew_mean")
    @test haskey(result, "eval/ep_rew_std")
    @test haskey(result, "training_time")
    @test result["framework"] == "Drill"
end

@testitem "Multi-Agent Mode Training" begin
    using DRL_RDE
    using Accessors
    using Random
    using Dates
    using Drill
    using DrillInterface
    using Enzyme
    # Create a minimal multi-agent configuration
    study = "test_multi_agent"
    run_num = 1
    seed = 456
    target_shock_count = 2
    n_agents = 4  # Smaller for testing

    # Multi-section reward
    periodmin_reward = MultiSectionPeriodMinimumReward(
        n_sections = n_agents,
        lowest_action_magnitude_reward = 0.5f0,
        weights = [1.0f0, 1.0f0, 1.0f0, 1.0f0]
    )

    # Environment config for multi-agent
    env_config = RDEEnvConfig(
        dt = 1.0f0,
        tmax = 50.0f0,  # Short for testing
        reset_strategy = ShiftReset(RandomShock()),
        observation_strat = MultiCenteredObservation(
            n_sections = n_agents,
            minisections = 8,  # Smaller for testing
        ),
        action_strat = VectorPressureAction(n_sections = n_agents),
        reward_strat = periodmin_reward,
        context_strat = FixedTargetShockCountStrategy(target_shock_count)
    )

    # Eval config
    eval_env_config = @set env_config.reset_strategy = ShiftReset(CycleShockReset(shocks = setdiff(1:4, target_shock_count)))
    eval_reward_strat = TransitionBasedReward(PeriodicityReward())
    eval_reward_strat = ScalarToVectorReward(eval_reward_strat, n_agents)
    eval_env_config = @set eval_env_config.reward_strat = eval_reward_strat
    eval_config = RDEEvalConfig(env_config = eval_env_config, n_eval_episodes = 6)  # Minimal evaluation

    # Learning config
    learn_config = DrillLearnConfig(
        max_timesteps = 1000,  # Very short for testing
        verbose = 0,
    )

    # Drill config
    dril_config = DrillConfig(
        algorithm = Drill.PPO(; n_steps = 32, batch_size = 16, epochs = 2, learning_rate = 3.0f-4),
        n_envs = 2,  # Minimal envs
        policy_config = LayerKwargs((; log_std_init = 0.0f0)),
        normalize_env = true,
        parallel_type = BroadcastedParallelEnv,
        learn_config = learn_config
    )

    # Create experiment
    experiment = DrillExperiment(
        study = study,
        run = run_num,
        seed = seed,
        env_config = env_config,
        eval_config = eval_config,
        dril_config = dril_config
    )

    # Create config dict
    config = Dict(
        "weights" => [1.0f0, 1.0f0, 1.0f0, 1.0f0],
        "run" => run_num,
        "lowest_action_magnitude_reward" => 0.5f0,
        "seed" => seed,
        "target_shock_count" => target_shock_count,
        "normalize_env" => true,
        "reward" => "periodminimum"
    )

    result = single_run(config, experiment; ignore_existing = true, save = false, logging = false)
    @test result isa Dict
    @test haskey(result, "eval/ep_rew_mean")
    @test haskey(result, "eval/ep_rew_std")
    @test haskey(result, "training_time")
    @test result["framework"] == "Drill"
end

@testitem "Single Agent Vector Action Mode Training" begin
    using DRL_RDE
    using Accessors
    using Random
    using Dates
    using Drill
    using DrillInterface
    using Enzyme
    # Create a minimal single agent vector action configuration
    study = "test_single_agent_vector"
    run_num = 1
    seed = 789
    target_shock_count = 2
    n_sections = 4  # Smaller for testing

    # Single agent reward (scalar)
    periodminreward = PeriodMinimumReward(
        lowest_action_magnitude_reward = 0.5f0,
        weights = [1.0f0, 1.0f0, 1.0f0, 1.0f0]
    )

    # Environment config for single agent with vector actions
    env_config = RDEEnvConfig(
        dt = 1.0f0,
        tmax = 50.0f0,  # Short for testing
        reset_strategy = ShiftReset(RandomShock()),
        observation_strat = SectionedStateObservation(
            minisections = 8,  # Smaller for testing
        ),
        action_strat = VectorPressureAction(n_sections = n_sections),  # Vector actions
        reward_strat = periodminreward,
        context_strat = FixedTargetShockCountStrategy(target_shock_count)
    )

    # Eval config
    eval_env_config = @set env_config.reset_strategy = ShiftReset(CycleShockReset(shocks = setdiff(1:4, target_shock_count)))
    eval_reward_strat = TransitionBasedReward(PeriodicityReward())
    eval_env_config = @set eval_env_config.reward_strat = eval_reward_strat
    eval_config = RDEEvalConfig(env_config = eval_env_config, n_eval_episodes = 6)  # Minimal evaluation

    # Learning config
    learn_config = DrillLearnConfig(
        max_timesteps = 1000,  # Very short for testing
        verbose = 0,
    )

    # Drill config
    dril_config = DrillConfig(
        algorithm = Drill.PPO(; n_steps = 32, batch_size = 16, epochs = 2, learning_rate = 3.0f-4),
        n_envs = 2,  # Minimal envs
        policy_config = LayerKwargs((; log_std_init = 0.0f0)),
        normalize_env = true,
        parallel_type = BroadcastedParallelEnv,
        learn_config = learn_config
    )

    # Create experiment
    experiment = DrillExperiment(
        study = study,
        run = run_num,
        seed = seed,
        env_config = env_config,
        eval_config = eval_config,
        dril_config = dril_config
    )

    # Create config dict
    config = Dict(
        "weights" => [1.0f0, 1.0f0, 1.0f0, 1.0f0],
        "run" => run_num,
        "lowest_action_magnitude_reward" => 0.5f0,
        "seed" => seed,
        "target_shock_count" => target_shock_count,
        "normalize_env" => true,
        "reward" => "periodminimum"
    )

    result = single_run(config, experiment; ignore_existing = true, save = false, logging = false)
    @test result isa Dict
    @test haskey(result, "eval/ep_rew_mean")
    @test haskey(result, "eval/ep_rew_std")
    @test haskey(result, "training_time")
    @test result["framework"] == "Drill"
end
