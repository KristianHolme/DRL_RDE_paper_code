@testitem "ShockSpanReward" begin
    using DrillInterface
    using RDE_Env
    # using DRL_RDE.RDE_Env_Strategies
    # Create environment with ShockSpanReward
    env = RDEEnv(;
        dt = 1.0,
        reward_strat = ShockSpanReward(
            span_scale = 4.0f0,
            shock_weight = 5.0f0
        ),
        context_strat = FixedTargetShockCountStrategy(3)
    )

    # Test initial reward
    reset!(env)
    @test env.reward isa Float32
    @test !isnan(env.reward)
    @test !isinf(env.reward)

    # Test reward with no shocks
    env.state .= 1.0  # Constant state = no shocks
    set_reward!(env, env.reward_strat)
    @test env.reward < 0  # Should be negative with no shocks

    # Test reward with target number of shocks
    # Create a state with 3 shocks
    N = env.prob.params.N
    env.state[1:N] .= 1.0
    env.state[N ÷ 4] = 2.0  # First shock
    env.state[N ÷ 2] = 2.0  # Second shock
    env.state[3N ÷ 4] = 2.0  # Third shock
    set_reward!(env, env.reward_strat)
    @test env.reward > 0  # Should be positive with target shocks
end

@testitem "ShockPreservingReward" begin
    using DrillInterface
    using RDE_Env
    target_shock_count = 3
    # Create environment with ShockPreservingReward
    params = RDEParam(tmax = 1.0)
    env = RDEEnv(
        params;
        dt = 0.1,
        τ_smooth = 0.05,
        reward_strat = ShockPreservingReward(
            abscence_limit = 0.01f0
        ),
        context_strat = FixedTargetShockCountStrategy(target_shock_count)
    )
    set_target_shock_count!(env, target_shock_count)

    # Test initial reward
    reset!(env)
    @test env.reward isa Float32
    @test !isnan(env.reward)
    @test !isinf(env.reward)

    # Test truncation with wrong number of shocks
    env.state .= 1.0  # Constant state = no shocks
    set_reward!(env, env.reward_strat)
    act!(env, 0.0f0)
    act!(env, 0.0f0) #act twice to outrun abscence limit

    @test env.terminated  # Should be terminated with wrong shock count
    @test env.reward ≈ -2.0  # Should get penalty reward

    # Test reward with correct number of shocks
    env.terminated = false  # Reset termination flag
    N = env.prob.params.N
    env.state[1:N] .= 1.0
    # Create 3 evenly spaced shocks
    env.state[N ÷ 3] = 2.0
    env.state[2N ÷ 3] = 2.0
    env.state[N] = 2.0
    set_reward!(env, env.reward_strat)
    @test !env.terminated  # Should not be terminated
    @test env.reward > 0  # Should get positive reward for evenly spaced shocks
    @test env.reward ≤ 1.0  # Reward should be normalized
end

@testitem "Policy Reward Behavior" begin
    using DrillInterface
    using RDE_Env

    # Test parameters
    tmax = 1.0
    dt = 0.1
    n_steps = Int(tmax / dt)

    @testset "Constant Policy with ShockSpanReward" begin
        env = RDEEnv(;
            dt = dt,
            τ_smooth = dt / 2,
            params = RDEParam(tmax = tmax),
            reward_strat = ShockSpanReward(
                span_scale = 4.0f0,
                shock_weight = 5.0f0
            ),
            context_strat = FixedTargetShockCountStrategy(3)
        )
        policy = ConstantRDEPolicy(env)

        # Run policy and collect rewards
        _reset!(env)
        rewards = Float32[]
        for _ in 1:n_steps
            action = _predict_action(policy, _observe(env))
            push!(rewards, _act!(env, action))
            @test !isnan(env.reward)
            @test !isinf(env.reward)
        end
        @test length(rewards) == n_steps
        @test !any(isnan.(rewards))
        @test !any(isinf.(rewards))
    end

    @testset "Random Policy with ShockPreservingReward" begin
        env = RDEEnv(;
            dt = dt,
            params = RDEParam(tmax = tmax),
            reward_strat = ShockPreservingReward(),
            context_strat = FixedTargetShockCountStrategy(3)
        )
        policy = RandomRDEPolicy(env)

        # Run policy and collect rewards
        reset!(env)
        rewards = Float32[]
        terminations = Int[]
        for step in 1:n_steps
            if env.done
                break
            end
            action = _predict_action(policy, _observe(env))
            push!(rewards, _act!(env, action))
            if env.terminated
                push!(terminations, step)
            end
            @test !isnan(env.reward)
            @test !isinf(env.reward)
        end

        # Test reward properties
        @test !isempty(rewards)
        @test !any(isnan.(rewards))
        @test !any(isinf.(rewards))

        # Test truncation behavior
        if !isempty(terminations)
            @test all(rewards[terminations] .≈ -2.0)  # All truncated steps should have penalty reward
        end
    end

    @testset "Policy Run Data Collection" begin
        env = RDEEnv(;
            dt = dt,
            params = RDEParam(tmax = tmax),
            reward_strat = ShockSpanReward(
                span_scale = 4.0f0,
                shock_weight = 5.0f0
            ),
            context_strat = FixedTargetShockCountStrategy(3)
        )
        policy = ConstantRDEPolicy(env)

        # Run policy and collect data
        data = run_policy(policy, env)

        # Test data properties
        @test !isempty(data.rewards)
        @test !any(isnan.(data.rewards))
        @test !any(isinf.(data.rewards))
        @test length(data.action_ts) == length(data.rewards)
        @test all(diff(data.action_ts) .≈ dt)  # Time steps should be consistent
    end
end

@testitem "TimeAggCompositeReward" begin
    using DrillInterface
    using RDE_Env

    # Create environment with TimeAggCompositeReward
    env = RDEEnv(;
        dt = 1.0,
        reward_strat = TimeAggCompositeReward(
            aggregation = TimeMin(),
            lowest_action_magnitude_reward = 0.5f0,
            weights = [0.25f0, 0.25f0, 0.25f0, 0.25f0]
        ),
        context_strat = FixedTargetShockCountStrategy(3)
    )

    # Test initial reward
    reset!(env)
    @test env.reward isa Float32
    @test !isnan(env.reward)
    @test !isinf(env.reward)

    # Test reward with no shocks
    env.state .= 1.0  # Constant state = no shocks
    set_reward!(env, env.reward_strat)
    @test env.reward ≈ 0  # Should be zero with no shocks

    # Test reward with target number of shocks
    N = env.prob.params.N
    env.state[1:N] .= 1.0
    env.state[N ÷ 4] = 2.0  # First shock
    env.state[N ÷ 2] = 2.0  # Second shock
    env.state[3N ÷ 4] = 2.0  # Third shock
    set_reward!(env, env.reward_strat)
    @test env.reward > 0  # Should be positive with target shocks

    # Test different aggregation methods
    for agg in [TimeMin(), TimeMax(), TimeAvg(), TimeSum(), TimeProd()]
        env.reward_strat = TimeAggCompositeReward(
            aggregation = agg,
            lowest_action_magnitude_reward = 0.5f0,
            weights = [0.25f0, 0.25f0, 0.25f0, 0.25f0]
        )
        set_reward!(env, env.reward_strat)
        @test !isnan(env.reward)
        @test !isinf(env.reward)
    end
end

@testitem "TimeAggMultiSectionReward" begin
    using DrillInterface
    using RDE_Env
    # Create environment with TimeAggMultiSectionReward
    env = RDEEnv(;
        dt = 1.0,
        reward_strat = TimeAggMultiSectionReward(
            aggregation = TimeMin(),
            n_sections = 4,
            lowest_action_magnitude_reward = 0.5f0,
            weights = [1.0f0, 1.0f0, 5.0f0, 1.0f0]
        ),
        context_strat = FixedTargetShockCountStrategy(3)
    )

    # Test initial reward
    reset!(env)
    @test env.reward isa Vector{Float32}
    @test length(env.reward) == 4  # One reward per section
    @test !any(isnan.(env.reward))
    @test !any(isinf.(env.reward))

    # Test reward with no shocks
    env.state .= 1.0  # Constant state = no shocks
    set_reward!(env, env.reward_strat)
    @test all(env.reward .≤ 0)  # All sections should have negative reward with no shocks

    # Test reward with target number of shocks
    N = env.prob.params.N
    env.state[1:N] .= 1.0
    env.state[N ÷ 4] = 2.0  # First shock
    env.state[N ÷ 2] = 2.0  # Second shock
    env.state[3N ÷ 4] = 2.0  # Third shock
    set_reward!(env, env.reward_strat)
    @test all(env.reward .≥ 0)  # All sections should have positive reward with target shocks

    # Test different aggregation methods
    for agg in [TimeMin(), TimeMax(), TimeAvg(), TimeSum(), TimeProd()]
        env.reward_strat = TimeAggMultiSectionReward(
            aggregation = agg,
            n_sections = 4,
            lowest_action_magnitude_reward = 0.5f0,
            weights = [1.0f0, 1.0f0, 5.0f0, 1.0f0]
        )
        set_reward!(env, env.reward_strat)
        @test !any(isnan.(env.reward))
        @test !any(isinf.(env.reward))
        @test length(env.reward) == 4  # Should maintain section count
    end
end

@testitem "MultiplicativeReward" begin
    using DrillInterface
    using RDE_Env

    @testset "Scalar rewards multiplication" begin
        # Create two scalar rewards
        reward1 = ShockSpanReward(
            span_scale = 4.0f0,
            shock_weight = 0.5f0
        )

        reward2 = TimeDiffNormReward(
            threshold = 10.0f0,
            threshold_reward = 0.1f0
        )

        # Create environment with MultiplicativeReward
        env = RDEEnv(;
            dt = 1.0,
            reward_strat = MultiplicativeReward(reward1, reward2),
            context_strat = FixedTargetShockCountStrategy(3)
        )

        # Test initial reward
        _reset!(env)
        @test env.reward isa Float32
        @test !isnan(env.reward)
        @test !isinf(env.reward)

        # Compare with individual rewards
        r1 = RDE_Env.compute_reward(env, reward1, env.cache.reward_cache.caches[1], env.cache.context)
        r2 = RDE_Env.compute_reward(env, reward2, env.cache.reward_cache.caches[2], env.cache.context)
        @test env.reward ≈ r1 * r2

        # Test with different state
        N = env.prob.params.N
        env.state[1:N] .= 1.0
        env.state[N ÷ 4] = 2.0  # First shock
        env.state[N ÷ 2] = 2.0  # Second shock
        env.state[3N ÷ 4] = 2.0  # Third shock

        set_reward!(env, env.reward_strat)
        r1 = RDE_Env.compute_reward(env, reward1, env.cache.reward_cache.caches[1], env.cache.context)
        r2 = RDE_Env.compute_reward(env, reward2, env.cache.reward_cache.caches[2], env.cache.context)
        @test env.reward ≈ r1 * r2
    end

    @testset "Vector rewards multiplication" begin
        # Create two vector rewards
        reward1 = TimeAggMultiSectionReward(
            aggregation = TimeMin(),
            n_sections = 4,
            lowest_action_magnitude_reward = 0.5f0,
            weights = [1.0f0, 1.0f0, 5.0f0, 1.0f0]
        )

        reward2 = MultiSectionReward(
            n_sections = 4,
            lowest_action_magnitude_reward = 0.8f0
        )

        # Create environment with MultiplicativeReward
        env = RDEEnv(;
            dt = 1.0,
            reward_strat = MultiplicativeReward(reward1, reward2),
            context_strat = FixedTargetShockCountStrategy(3)
        )

        # Test initial reward
        reset!(env)
        @test env.reward isa Vector{Float32}
        @test length(env.reward) == 4  # Should maintain section count
        @test !any(isnan.(env.reward))
        @test !any(isinf.(env.reward))

        # Compare with individual rewards
        r1 = RDE_Env.compute_reward(env, reward1, env.cache.reward_cache.caches[1], env.cache.context)
        r2 = RDE_Env.compute_reward(env, reward2, env.cache.reward_cache.caches[2], env.cache.context)
        @test all(env.reward .≈ r1 .* r2)

        # Test with different state
        N = env.prob.params.N
        env.state[1:N] .= 1.0
        env.state[N ÷ 4] = 2.0  # First shock
        env.state[N ÷ 2] = 2.0  # Second shock
        env.state[3N ÷ 4] = 2.0  # Third shock

        set_reward!(env, env.reward_strat)
        r1 = RDE_Env.compute_reward(env, reward1, env.cache.reward_cache.caches[1], env.cache.context)
        r2 = RDE_Env.compute_reward(env, reward2, env.cache.reward_cache.caches[2], env.cache.context)
        @test all(env.reward .≈ r1 .* r2)
    end

    @testset "Mixed scalar and vector rewards" begin
        # Create a scalar reward
        scalar_reward = ShockSpanReward(
            span_scale = 4.0f0,
            shock_weight = 0.5f0
        )

        # Create a vector reward
        vector_reward = MultiSectionReward(
            n_sections = 4,
            lowest_action_magnitude_reward = 0.8f0
        )

        # Create environment with MultiplicativeReward
        env = RDEEnv(;
            dt = 1.0,
            reward_strat = MultiplicativeReward(scalar_reward, vector_reward),
            context_strat = FixedTargetShockCountStrategy(3)
        )

        # Test initial reward
        reset!(env)
        @test env.reward isa Vector{Float32}
        @test length(env.reward) == 4  # Should maintain section count from vector reward
        @test !any(isnan.(env.reward))
        @test !any(isinf.(env.reward))

        # Compare with individual rewards
        r1 = RDE_Env.compute_reward(env, scalar_reward, env.cache.reward_cache.caches[1], env.cache.context)
        r2 = RDE_Env.compute_reward(env, vector_reward, env.cache.reward_cache.caches[2], env.cache.context)
        @test all(env.reward .≈ r1 .* r2)

        # Test scalar reward first vs. vector reward first
        env2 = RDEEnv(;
            dt = 1.0,
            reward_strat = MultiplicativeReward(vector_reward, scalar_reward),
            context_strat = FixedTargetShockCountStrategy(3)
        )
        reset!(env2)
        set_reward!(env2, env2.reward_strat)
        @test all(env2.reward .≈ env.reward)  # Order shouldn't matter

        # Test with multiple scalar rewards
        scalar_reward2 = TimeDiffNormReward(threshold = 10.0f0)
        env3 = RDEEnv(;
            dt = 1.0,
            reward_strat = MultiplicativeReward(scalar_reward, vector_reward, scalar_reward2),
            context_strat = FixedTargetShockCountStrategy(3)
        )
        reset!(env3)
        set_reward!(env3, env3.reward_strat)
        r3 = RDE_Env.compute_reward(env3, scalar_reward2, env3.cache.reward_cache.caches[3], env3.cache.context)
        @test all(env3.reward .≈ r1 .* r2 .* r3)
    end

    @testset "Policy with MultiplicativeReward" begin
        using DrillInterface
        using RDE_Env
        # Test parameters
        tmax = 1.0
        dt = 0.1
        n_steps = Int(tmax / dt)

        # Create a multiplicative reward with mixed types
        scalar_reward = ShockSpanReward()
        vector_reward = MultiSectionReward(n_sections = 4)

        env = RDEEnv(;
            dt = dt,
            τ_smooth = dt / 2,
            params = RDEParam(tmax = tmax),
            reward_strat = MultiplicativeReward(scalar_reward, vector_reward),
            context_strat = FixedTargetShockCountStrategy(3)
        )
        policy = ConstantRDEPolicy(env)

        # Run policy and collect rewards
        reset!(env)
        for _ in 1:n_steps
            action = _predict_action(policy, _observe(env))
            act!(env, action)
            @test !any(isnan.(env.reward))
            @test !any(isinf.(env.reward))
        end

        # Run policy and collect data
        data = run_policy(policy, env)

        # Test data properties
        @test !isempty(data.rewards)
        @test all(x -> x isa Vector{Float32}, data.rewards)
        @test all(x -> !any(isnan.(x)), data.rewards)
        @test all(x -> !any(isinf.(x)), data.rewards)
        @test length(data.action_ts) == length(data.rewards)
    end
end

@testitem "StabilityTargetReward Constructor" begin
    # Test default constructor
    reward = StabilityTargetReward()
    @test reward isa StabilityTargetReward{Float32}
    @test reward.stability_weight == 0.5f0
    @test reward.target_weight == 0.5f0
    @test reward.stability_reward isa StabilityReward{Float32}

    # Test custom weights
    reward2 = StabilityTargetReward(stability_weight = 0.8f0, target_weight = 0.2f0)
    @test reward2.stability_weight == 0.8f0
    @test reward2.target_weight == 0.2f0

    # Test passing kwargs to StabilityReward
    reward3 = StabilityTargetReward(variation_scaling = 5.0f0)
    @test reward3.stability_reward.variation_scaling == 5.0f0

    # Test type consistency - should error with inconsistent types
    @test_throws MethodError StabilityTargetReward(stability_weight = 0.8f0, target_weight = 0.2)  # Float32 vs Float64
    @test_throws MethodError StabilityTargetReward(stability_weight = 0.8, target_weight = 0.2f0)  # Float64 vs Float32

    # Test type consistency with kwargs gets converted
    @test StabilityTargetReward(variation_scaling = 5.0) isa StabilityTargetReward{Float32}

    # Test typed constructor with consistent types
    reward4 = StabilityTargetReward(stability_weight = 0.8, target_weight = 0.2, variation_scaling = 5.0)
    @test reward4 isa StabilityTargetReward{Float64}
    @test reward4.stability_weight isa Float64
    @test reward4.target_weight isa Float64
    @test reward4.stability_reward.variation_scaling isa Float64
end

@testitem "StabilityTargetReward Cache" begin
    using RDE_Env
    reward = StabilityTargetReward()
    N = 512
    cache = RDE_Env.initialize_cache(reward, N, Float32)

    @test cache isa WrappedRewardCache
    @test cache.inner_cache isa RewardShiftBufferCache{Float32}
    @test cache.outer_cache isa NoCache
    @test length(cache.inner_cache.shift_buffer) == N
end

@testitem "StabilityTargetReward Basic Computation" begin
    using RDE_Env
    # Create environment with StabilityTargetReward
    env = RDEEnv(;
        dt = 1.0,
        reward_strat = StabilityTargetReward(),
        context_strat = FixedTargetShockCountStrategy(2)
    )

    # Test initial reward
    _reset!(env)
    @test env.reward isa Float32
    @test !isnan(env.reward)
    @test !isinf(env.reward)
    @test env.reward >= 0  # Should be non-negative initially

    # Get target shock count
    target_count = get_target_shock_count(env)
    @test target_count == 2  # Default target
end

@testitem "StabilityTargetReward Target Matching" begin
    using DrillInterface
    using RDE_Env
    target_shock_count = 1
    action = 0.65f0
    # Create environment
    env = RDEEnv(;
        dt = 1.0,
        reward_strat = StabilityTargetReward(stability_weight = 0.5f0, target_weight = 0.5f0),
        context_strat = FixedTargetShockCountStrategy(target_shock_count)
    )
    set_target_shock_count!(env, target_shock_count)
    reset!(env)

    # Get initial reward with correct number of shocks
    for _ in 1:50
        act!(env, action)
    end

    initial_reward = act!(env, action)

    # Create state with wrong number of shocks (0 shocks)
    N = env.prob.params.N
    env.state[1:N] .= 1.0  # Constant state = no shocks
    set_reward!(env, env.reward_strat)
    no_shock_reward = env.reward

    # Should get lower reward with wrong shock count
    # (no target bonus, only stability component)
    @test no_shock_reward < initial_reward
end

@testitem "StabilityTargetReward Weight Variation" begin
    using DrillInterface
    using RDE_Env
    # Test different weight combinations
    weights = [
        (0.9f0, 0.1f0),  # Heavily weighted toward stability
        (0.5f0, 0.5f0),  # Equal weighting
        (0.3f0, 0.7f0),  # Heavily weighted toward target
    ]

    for (stab_w, targ_w) in weights
        env = RDEEnv(;
            dt = 1.0,
            reward_strat = StabilityTargetReward(
                stability_weight = stab_w,
                target_weight = targ_w
            ),
            context_strat = FixedTargetShockCountStrategy(2)
        )
        reset!(env)
        @test env.reward isa Float32
        @test !isnan(env.reward)
        @test !isinf(env.reward)
        @test env.reward_strat.stability_weight == stab_w
        @test env.reward_strat.target_weight == targ_w
    end
end

@testitem "StabilityTargetReward Policy Run" begin
    using DrillInterface
    using RDE_Env
    # Test parameters
    tmax = 1.0
    dt = 0.1
    n_steps = Int(tmax / dt)

    # Test with ConstantRDEPolicy
    env = RDEEnv(;
        dt = dt,
        τ_smooth = dt / 2,
        params = RDEParam(tmax = tmax),
        reward_strat = StabilityTargetReward(),
        context_strat = FixedTargetShockCountStrategy(2)
    )
    policy = ConstantRDEPolicy(env)

    # Run policy and collect rewards
    reset!(env)
    rewards = Float32[]
    for _ in 1:n_steps
        action = _predict_action(policy, _observe(env))
        push!(rewards, act!(env, action))
        @test !isnan(env.reward)
        @test !isinf(env.reward)
    end

    @test length(rewards) == n_steps
    @test !any(isnan.(rewards))
    @test !any(isinf.(rewards))
    @test all(rewards .>= 0)  # Should be non-negative

    # Run full policy data collection
    data = run_policy(policy, env)
    @test !isempty(data.rewards)
    @test !any(isnan.(data.rewards))
    @test !any(isinf.(data.rewards))
    @test length(data.action_ts) == length(data.rewards)
end

@testitem "StabilityTargetReward Comparison" begin
    using DrillInterface
    using RDE_Env
    # Create two environments: one with StabilityReward, one with StabilityTargetReward
    env_stability = RDEEnv(;
        dt = 1.0,
        reward_strat = StabilityReward(),
        context_strat = FixedTargetShockCountStrategy(2)

    )

    env_target = RDEEnv(;
        dt = 1.0,
        reward_strat = StabilityTargetReward(stability_weight = 1.0f0, target_weight = 0.0f0),
        context_strat = FixedTargetShockCountStrategy(2)
    )

    # Reset both with same seed
    reset!(env_stability)
    reset!(env_target)

    # When target_weight=0, should behave like pure StabilityReward
    @test env_stability.reward ≈ env_target.reward rtol = 1.0e-5
end

@testitem "DtWeightedReward scales with env.dt" begin
    using DrillInterface
    using RDE_Env

    env_wrapped = RDEEnv(;
        dt = 1.0f0,
        reward_strat = DtWeightedReward(StabilityTargetReward()),
        action_strat = LinearScalarPressureAction(),
        context_strat = FixedTargetShockCountStrategy(2),
    )
    reset!(env_wrapped)
    act!(env_wrapped, 0.0f0)
    r1 = env_wrapped.reward
    env_wrapped.dt = 2.0f0
    set_reward!(env_wrapped, env_wrapped.reward_strat)
    @test env_wrapped.reward ≈ 2 * r1

    env_plain = RDEEnv(;
        dt = 1.0f0,
        reward_strat = StabilityTargetReward(),
        action_strat = LinearScalarPressureAction(),
        context_strat = FixedTargetShockCountStrategy(2),
    )
    reset!(env_plain)
    act!(env_plain, 0.0f0)
    r_plain = env_plain.reward
    env_plain.dt = 2.0f0
    set_reward!(env_plain, env_plain.reward_strat)
    @test env_plain.reward ≈ r_plain
end
