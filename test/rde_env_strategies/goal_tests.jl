@testitem "FixedTargetShockCountStrategy Basic Functionality" begin
    context_strat = FixedTargetShockCountStrategy(3)
    env = RDEEnv(RDEParam{Float32}(N = 32), context_strat = context_strat)
    @test context_strat.target_shock_count == 3

    # Test get_target_shock_count
    @test get_target_shock_count(env) == 3
end

@testitem "RandomTargetShockCountStrategy Basic Functionality" begin
    context_strat = RandomTargetShockCountStrategy(min_value = 1, max_value = 4)
    @test context_strat.min_value == 1
    @test context_strat.max_value == 4

    # Test cache initialization
    cache = RDE_Env.initialize_cache(context_strat, 32, Float32)
    @test cache isa TargetShockCountContext
    @test 1 ≤ cache.target_shock_count ≤ 4
end

@testitem "EvalCycleTargetShockCountStrategy Initialization" begin
    context_strat = EvalCycleTargetShockCountStrategy(repetitions_per_config = 4)
    @test context_strat.repetitions_per_config == 4

    # Test cache initialization
    cache = RDE_Env.initialize_cache(context_strat, 32, Float32)
    @test cache isa EvalCycleTargetShockCountContext
    @test cache.current_config == 1
    @test length(cache.target_shocks) == 48  # 4 goals × 3 non-goal shocks × 4 repetitions

    # Verify the pattern: 12×1, 12×2, 12×3, 12×4
    expected_pattern = repeat(1:4, inner = 12)
    @test cache.target_shocks == expected_pattern
end

@testitem "EvalCycleTargetShockCountStrategy Custom Repetitions" begin
    context_strat = EvalCycleTargetShockCountStrategy(repetitions_per_config = 2)
    cache = RDE_Env.initialize_cache(context_strat, 32, Float32)

    @test length(cache.target_shocks) == 24  # 4 goals × 3 non-goal shocks × 2 repetitions

    # Verify the pattern: 6×1, 6×2, 6×3, 6×4
    expected_pattern = vcat([repeat([i], 6) for i in 1:4]...)
    @test cache.target_shocks == expected_pattern
end

@testitem "EvalCycleTargetShockCountStrategy Goal Sequence" begin
    context_strat = EvalCycleTargetShockCountStrategy(repetitions_per_config = 4)
    env = RDEEnv(RDEParam{Float32}(N = 32), context_strat = context_strat)
    # Test first 12 resets should all be goal 1
    expected_targets = repeat(1:4, inner = 12)

    context = env.cache.context
    @test context.target_shocks == expected_targets

    in_syncs = Bool[]
    for i in 1:48
        push!(in_syncs, context.current_config == i)
        _reset!(env)
    end

    #should have wrapped
    @test context.current_config == 1
end

@testitem "EvalCycleTargetShockCountStrategy Wrap Around" begin
    context_strat = EvalCycleTargetShockCountStrategy(repetitions_per_config = 2)
    env = RDEEnv(RDEParam{Float32}(N = 32), context_strat = context_strat)
    context = env.cache.context

    # Go through all 24 configurations
    for i in 1:24
        RDE_Env.on_reset!(context, context_strat, env)
    end

    # Should have wrapped around to config 1 (goal 1)
    @test context.current_config == 1
    @test get_target_shock_count(env) == 1
end

@testitem "EvalCycleTargetShockCountStrategy and EvalCycleShockReset Synchronization" begin
    # Create both strategies with same repetitions
    context_strat = EvalCycleTargetShockCountStrategy(repetitions_per_config = 4)
    reset_strat = EvalCycleShockReset(4)

    env = RDEEnv(; context_strat, reset_strategy = reset_strat)
    context = env.cache.context
    ok_steps = Bool[]
    for i in 1:48
        step_ok = reset_strat.current_config == i && context.current_config == i
        push!(ok_steps, step_ok)
        _reset!(env)
    end
    @test all(ok_steps)

    # Should have wrapped around to the beginning
    @test context.current_config == 1
    @test reset_strat.current_config == 1
    @test get_target_shock_count(env) == 1
end

@testitem "EvalCycleTargetShockCountStrategy and EvalCycleShockReset Synchronization" begin
    reset_strat = EvalCycleShockReset(4)
    context_strat = EvalCycleTargetShockCountStrategy(repetitions_per_config = 4)
    env = RDEEnv(context_strat = context_strat, reset_strategy = reset_strat)
    # Test synchronized behavior over 48 resets
    expected_inits = vcat([repeat(setdiff(1:4, i), inner = 4) for i in 1:4]...)
    expected_targets = repeat(1:4, inner = 12)

    @test reset_strat.current_config == 1


    in_sync = [true]
    for i in 1:48
        reset_strat_current = reset_strat.current_config
        if reset_strat_current != i
            @error "Reset strategy current config $reset_strat_current does not match expected $i"
            in_sync[1] = false
        end
        if get_target_shock_count(env) != expected_targets[i]
            @error "i = $i, Target shock count $(get_target_shock_count(env)) does not match expected $(expected_targets[i])"
            in_sync[1] = false
        end
        if reset_strat.init_shocks[i] != expected_inits[i]
            @error "i = $i, Init shock $(reset_strat.init_shocks[i]) does not match expected $(expected_inits[i])"
            in_sync[1] = false
        end
        _reset!(env)
    end
    @test in_sync[1]
    @test env.cache.context.current_config == 1
    @test reset_strat.current_config == 1
    @test get_target_shock_count(env) == 1
end
