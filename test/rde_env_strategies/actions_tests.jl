@testitem "LinearScalarPressureAction" begin
    using DRL_RDE.RDE
    using Accessors
    using DRL_RDE.RDE_Env

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(params, action_strat = LinearScalarPressureAction())
    _reset!(env)

    # Test basic action application with scalar input
    u_p_initial = mean_u_p(env.prob.injection)
    action = 0.5f0
    RDE_Env.apply_action!(env, action, env.action_strat, env.cache.action_cache, env.cache.context)
    expected = linear_action_to_control(action, u_p_initial, env.u_pmax, RDE_Env.momentum(env.action_strat))
    @test current_u_p(env.prob.injection) ≈ expected

    # Test with array input
    _reset!(env)
    RDE_Env.apply_action!(env, [0.3f0], env.action_strat, env.cache.action_cache, env.cache.context)
    @test true  # Should not throw

    # Test momentum: α=0 gives direct mapping
    env_no_momentum = RDEEnv(params, action_strat = LinearScalarPressureAction())
    _reset!(env_no_momentum)
    @reset env_no_momentum.action_strat.momentum = 0.0f0
    RDE_Env.apply_action!(env_no_momentum, 0.0f0, env_no_momentum.action_strat, env_no_momentum.cache.action_cache, env_no_momentum.cache.context)
    @test mean_u_p(env_no_momentum.prob.injection) ≈ 0.5f0 * env_no_momentum.u_pmax

    # Test bounds checking
    @test_logs (:warn,) RDE_Env.apply_action!(env, 1.5f0, env.action_strat, env.cache.action_cache, env.cache.context)
    @test_logs (:warn,) RDE_Env.apply_action!(env, -1.1f0, env.action_strat, env.cache.action_cache, env.cache.context)
    @test_logs RDE_Env.apply_action!(env, 1.0f0, env.action_strat, env.cache.action_cache, env.cache.context)
    @test_logs RDE_Env.apply_action!(env, -1.0f0, env.action_strat, env.cache.action_cache, env.cache.context)
end

@testitem "LinearVectorPressureAction" begin

    params = RDEParam(N = 512, tmax = 0.01)

    # Test single section behaves like scalar
    env1 = RDEEnv(params, action_strat = LinearVectorPressureAction(n_sections = 1))
    _reset!(env1)
    RDE_Env.apply_action!(env1, [0.5f0], env1.action_strat, env1.cache.action_cache, env1.cache.context)
    @test true  # Should not throw

    # Test multiple independent sections (inference uses section midpoints)
    env4 = RDEEnv(params, action_strat = LinearVectorPressureAction(n_sections = 4))
    _reset!(env4)
    action = [-1.0f0, -0.3f0, 0.3f0, 1.0f0]
    midpoint_indices = RDE_Env.section_midpoint_indices(512, 4)
    u_now = current_u_p(env4.prob.injection)
    u_p_initial = if u_now isa AbstractVector
        [u_now[midpoint_indices[i]] for i in 1:4]
    else
        fill(u_now, 4)
    end

    RDE_Env.apply_action!(env4, action, env4.action_strat, env4.cache.action_cache, env4.cache.context)

    # Verify each section has correct independent value at section midpoint
    for i in 1:4
        sample_idx = midpoint_indices[i]
        expected = linear_action_to_control(action[i], u_p_initial[i], env4.u_pmax, RDE_Env.momentum(env4.action_strat))
        @test current_u_p(env4.prob.injection)[sample_idx] ≈ expected
    end

    # Test validation for wrong action length
    @test_throws AssertionError RDE_Env.apply_action!(env4, [0.5f0, 0.5f0], env4.action_strat, env4.cache.action_cache, env4.cache.context)
    @test_throws AssertionError RDE_Env.apply_action!(env4, [0.1f0, 0.2f0, 0.3f0, 0.4f0, 0.5f0], env4.action_strat, env4.cache.action_cache, env4.cache.context)
end

@testitem "Linear actions type stability" begin
    using RDE

    # Test Float32
    params32 = RDEParam{Float32}(tmax = 0.01f0)
    env32 = RDEEnv(params32, action_strat = LinearScalarPressureAction())
    _reset!(env32)
    RDE_Env.apply_action!(env32, 0.5f0, env32.action_strat, env32.cache.action_cache, env32.cache.context)
    @test typeof(current_u_p(env32.prob.injection)) == Float32

    # Test Float64
    params64 = RDEParam{Float64}(tmax = 0.01)
    env64 = RDEEnv(params64, action_strat = LinearScalarPressureAction(momentum = 0.0))
    _reset!(env64)
    RDE_Env.apply_action!(env64, 0.5, env64.action_strat, env64.cache.action_cache, env64.cache.context)
    @test typeof(current_u_p(env64.prob.injection)) == Float64

    # Test vector action type stability
    env32_vec = RDEEnv(params32, action_strat = LinearVectorPressureAction(n_sections = 2))
    _reset!(env32_vec)
    RDE_Env.apply_action!(env32_vec, [0.3f0, 0.7f0], env32_vec.action_strat, env32_vec.cache.action_cache, env32_vec.cache.context)
    @test eltype(current_u_p(env32_vec.prob.injection)) == Float32
end

@testitem "TargetSafeLinearPressureAction map" begin
    using DRL_RDE.RDE

    u_pmax = 1.0f0
    for t in 1:4
        u_safe = Float32(RDE.SHOCK_PRESSURES[t])
        @test target_safe_control_target(0.0f0, u_safe, u_pmax) ≈ u_safe
        @test target_safe_control_target(-1.0f0, u_safe, u_pmax) ≈ 0.0f0
        @test target_safe_control_target(1.0f0, u_safe, u_pmax) ≈ u_pmax
        @test control_to_target_safe_action(u_safe, u_safe, u_pmax) ≈ 0.0f0
        @test control_to_target_safe_action(0.0f0, u_safe, u_pmax) ≈ -1.0f0
        @test control_to_target_safe_action(u_pmax, u_safe, u_pmax) ≈ 1.0f0
        a = -0.4f0
        u = target_safe_control_target(a, u_safe, u_pmax)
        @test control_to_target_safe_action(u, u_safe, u_pmax) ≈ a
        a = 0.6f0
        u = target_safe_control_target(a, u_safe, u_pmax)
        @test control_to_target_safe_action(u, u_safe, u_pmax) ≈ a
    end
    @test RDE.SHOCK_PRESSURES[2] ≈ 0.65
end

@testitem "TargetSafeLinearScalarPressureAction apply" begin
    using DRL_RDE.RDE
    using DRL_RDE.RDE_Env

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(
        params;
        action_strat = TargetSafeLinearScalarPressureAction(),
        context_strat = FixedTargetShockCountStrategy(2),
    )
    _reset!(env)
    u_safe = Float32(RDE.SHOCK_PRESSURES[2])
    @test target_safe_pressure(env) ≈ u_safe

    RDE_Env.apply_action!(env, 0.0f0, env.action_strat, env.cache.action_cache, env.cache.context)
    @test mean_u_p(env.prob.injection) ≈ u_safe

    RDE_Env.apply_action!(env, -1.0f0, env.action_strat, env.cache.action_cache, env.cache.context)
    @test mean_u_p(env.prob.injection) ≈ 0.0f0

    RDE_Env.apply_action!(env, 1.0f0, env.action_strat, env.cache.action_cache, env.cache.context)
    @test mean_u_p(env.prob.injection) ≈ env.u_pmax

    RDE_Env.apply_action!(env, [0.0f0], env.action_strat, env.cache.action_cache, env.cache.context)
    @test mean_u_p(env.prob.injection) ≈ u_safe
end

@testitem "TargetSafeLinearMultiStepPressureAction apply" begin
    using DRL_RDE.RDE
    using DRL_RDE.RDE_Env

    params = RDEParam(N = 512, tmax = 0.01)
    env = RDEEnv(
        params;
        dt = 10.0f0,
        action_strat = TargetSafeLinearMultiStepPressureAction(; n_sections = 1, n_steps = 2),
        context_strat = FixedTargetShockCountStrategy(2),
    )
    _reset!(env)
    u_safe = Float32(RDE.SHOCK_PRESSURES[2])
    RDE_Env.apply_action!(
        env,
        [0.0f0, 0.0f0],
        env.action_strat,
        env.cache.action_cache,
        env.cache.context,
    )
    @test env.cache.action_cache.section_controls ≈ [u_safe, u_safe]
    @test env.cache.action_cache.times ≈ [env.t + 5.0f0, env.t + 10.0f0]
end
