using RDE_Env
using DRL_RDE.RDE_Env_Strategies

@testitem "RDEEnv Initialization" begin
    using RDE

    @test begin
        params = RDEParam(; N = 16, tmax = 0.01)
        prob = RDEProblem(params)
        sum(abs.(prob.u0)) > 0.01
    end

    @test begin
        env = RDEEnv()
        true
    end
end

@testitem "RDEEnv Policies" begin
    using RDE

    @test begin
        env = RDEEnv(RDEParam(; N = 512, tmax = 0.1), reset_strategy = NShock(1))
        ConstPolicy = ConstantRDEPolicy(env)
        data = run_policy(ConstPolicy, env)
        data isa PolicyRunData
    end
end

@testitem "Fourier Observation" begin
    using RDE

    N = 16  # Number of spatial points for testing
    params = RDEParam(; N = N, tmax = 0.1)

    fft_terms = 4
    env = RDEEnv(params = params, observation_strat = FourierObservation(fft_terms))
    _reset!(env)
    # Test initialization
    @test length(env.observation) == 2fft_terms + 2  # 2 * fft_terms + s_scaled + u_p_scaled

    # Test observation
    obs = _observe(env)
    @test length(obs) == 2fft_terms + 2
    @test all(-1 .<= obs[1:2fft_terms] .<= 1)  # FFT coefficients should be normalized
    @test -1 <= obs[2fft_terms + 1] <= 1  # Normalized s_scaled
    @test -1 <= obs[2fft_terms + 2] <= 1  # Normalized u_p_scaled
end

@testitem "Fourier Magnitude Observation" begin
    using RDE

    N = 16
    n_modes = 8
    params = RDEParam(; N = N, tmax = 0.1)
    env = RDEEnv(params = params, observation_strat = FourierMagnitudeObservation(n_modes = n_modes))
    _reset!(env)
    @test length(env.observation) == 2 * n_modes
    obs = _observe(env)
    @test length(obs) == 16
    @test all(obs .>= 0)
    @test all(isfinite, obs)
end

@testitem "Fourier Magnitude Action History Observation" begin
    using DrillInterface
    using RDE
    using Statistics: mean

    N = 16
    n_modes = 8
    history_length = 8
    params = RDEParam(; N = N, tmax = 1.0)
    env = RDEEnv(
        params;
        dt = 0.5f0,
        observation_strat = FourierMagnitudeActionHistoryObservation(; n_modes, history_length),
        action_strat = LinearScalarPressureWithDtAction(),
        context_strat = FixedTargetShockCountStrategy(2),
    )
    _reset!(env)
    obs0 = copy(_observe(env))
    @test length(obs0) == 2 * n_modes + 2 * history_length + 2
    @test length(env.observation) == 34
    @test all(obs0[1:(2 * n_modes)] .>= 0)
    @test all(isfinite, obs0)
    @test 0 <= obs0[end - 1] <= 1
    @test 0 <= obs0[end] <= 1
    @test obs0[end] ≈ 2 / 4

    pressure_hist0 = @view obs0[(2 * n_modes + 1):(2 * n_modes + history_length)]
    dt_hist0 = @view obs0[(2 * n_modes + history_length + 1):(2 * n_modes + 2 * history_length)]
    u_p0 = mean_u_p(env.prob.injection) / env.u_pmax
    @test all(pressure_hist0 .≈ u_p0)
    @test all(dt_hist0 .≈ env.dt / env.action_strat.dt_max)

    state_u = copy(@view env.state[1:N])
    state_λ = copy(@view env.state[(N + 1):end])
    circshift!(state_u, 3)
    circshift!(state_λ, 3)
    env.state[1:N] .= state_u
    env.state[(N + 1):end] .= state_λ
    obs_shift = copy(_observe(env))
    @test obs_shift[1:(2 * n_modes)] ≈ obs0[1:(2 * n_modes)]

    _reset!(env)
    act!(env, Float32[1.0, 1.0])
    obs1 = env.observation
    u_p1 = mean_u_p(env.prob.injection) / env.u_pmax
    @test obs1[2 * n_modes + history_length] ≈ u_p1
    @test obs1[2 * n_modes + 2 * history_length] ≈ env.dt / env.action_strat.dt_max
    @test env.dt ≈ env.action_strat.dt_max
end

@testitem "Fourier Magnitude Pressure Shock Observation and MultiStep action" begin
    using DrillInterface
    using RDE

    N = 16
    n_modes = 8
    n_multisteps = 4
    dt = 1.0f0
    params = RDEParam(; N = N, tmax = 2.0f0)
    env = RDEEnv(
        params;
        dt = dt,
        observation_strat = FourierMagnitudePressureShockObservation(; n_modes),
        action_strat = MultiStepPressureAction(; n_sections = 1, n_steps = n_multisteps),
        context_strat = FixedTargetShockCountStrategy(2),
        reset_strategy = NShock(2),
        verbose = false,
    )
    _reset!(env)
    obs = copy(_observe(env))
    @test length(obs) == 2 * n_modes + 3
    @test length(env.observation) == 19
    @test all(obs[1:(2 * n_modes)] .>= 0)
    @test all(isfinite, obs)
    @test 0 <= obs[end - 2] <= 1
    @test 0 <= obs[end - 1] <= 1
    @test 0 <= obs[end] <= 1
    @test obs[end] ≈ 2 / 4
    @test obs[end - 2] ≈ mean_u_p(env.prob.injection) / env.u_pmax

    aspace = action_space(env)
    @test length(aspace.low) == n_multisteps
    @test length(aspace.high) == n_multisteps

    t0 = env.t
    action = fill(0.6f0, n_multisteps)
    act!(env, action)
    times = env.prob.injection.times
    @test length(times) == n_multisteps + 1
    @test times[1] ≈ t0
    step_dt = dt / n_multisteps
    for k in 1:n_multisteps
        @test times[k + 1] ≈ t0 + k * step_dt
    end
    @test env.prob.injection isa UniformMultiStepPressureProfile
    @test env.prob.injection.smoother isa CinftySmoother
end

@testitem "Pressure History Wave Reward Observation" begin
    using DrillInterface
    using RDE

    N = 16
    history_length = 8
    params = RDEParam(; N = N, tmax = 0.5)
    env = RDEEnv(
        params;
        dt = 0.1,
        observation_strat = PressureHistoryWaveRewardObservation(history_length = history_length),
        action_strat = LinearScalarPressureAction(),
        context_strat = FixedTargetShockCountStrategy(2),
    )
    _reset!(env)
    obs0 = _observe(env)
    @test length(obs0) == history_length + 2
    @test length(env.observation) == 10
    @test obs0[end] == 0
    @test all(obs0[2:(history_length + 1)] .≈ obs0[2])
    r_reset = env.reward
    act!(env, 1.0f0)
    obs1 = _observe(env)
    @test length(obs1) == 10
    @test obs1[end] ≈ r_reset
    @test all(obs1[2:history_length] .≈ obs0[2])
end

@testitem "State Observation" begin
    using RDE

    N = 16  # Number of spatial points for testing
    params = RDEParam(; N = N, tmax = 0.1)

    env = RDEEnv(params = params, observation_strat = StateObservation())
    _reset!(env)
    # Test initialization
    @test length(env.observation) == 2N + 2  # u and λ states + s_scaled + u_p_scaled

    # Test observation
    obs = _observe(env)
    @test length(obs) == 2N + 2
    @test all(-1 .<= obs[1:(end - 2)] .<= 1)  # State components should be normalized
    @test 0 <= obs[end - 1] <= 1  # Normalized s
    @test 0 <= obs[end] <= 1  # Normalized u_p

    # Test that first N components are normalized u and next N are normalized λ
    u_obs = obs[1:N]
    λ_obs = obs[(N + 1):2N]
    @test length(u_obs) == length(λ_obs) == N
end

@testitem "Sampled Observation" begin
    using RDE

    N = 16  # Number of spatial points for testing
    params = RDEParam(; N = N, tmax = 0.1)

    n_samples = 8
    env = RDEEnv(params = params, observation_strat = SampledStateObservation(n_samples))
    _reset!(env)
    # Test initialization
    @test length(env.observation) == 2n_samples + 1  # sampled u and λ + time

    # Test observation
    obs = _observe(env)
    @test length(obs) == 2n_samples + 1
    @test all(-1 .<= obs[1:(end - 1)] .<= 1)  # Sampled values should be normalized
    @test 0 <= obs[end] <= 1  # Normalized time

    # Test sampling points
    u_samples = obs[1:n_samples]
    λ_samples = obs[(n_samples + 1):2n_samples]
    @test length(u_samples) == length(λ_samples) == n_samples
end

@testitem "Observation Consistency" begin
    using RDE

    N = 16  # Number of spatial points for testing
    params = RDEParam(; N = N, tmax = 0.1)

    # Test that observations remain consistent after reset
    for strategy in [
            FourierObservation(4),
            StateObservation(),
            SampledStateObservation(8),
        ]
        env = RDEEnv(params = params, observation_strat = strategy)
        _reset!(env)
        obs1 = _observe(env)
        _reset!(env)
        obs2 = _observe(env)
        @test length(obs1) == length(obs2)
        @test all(isfinite.(obs1))
        @test all(isfinite.(obs2))
    end
end

@testitem "SectionedStateWithPressureHistoryObservation" begin
    using RDE

    N = 16  # Number of spatial points for testing
    params = RDEParam(; N = N, tmax = 0.1)

    history_length = 5
    minisections = 8
    env = RDEEnv(
        params = params,
        observation_strat = SectionedStateWithPressureHistoryObservation(; minisections, history_length),
        action_strat = DirectScalarPressureAction(),
        context_strat = FixedTargetShockCountStrategy(3)
    )
    _reset!(env)

    # Test initialization
    @test length(env.observation) == 2 * minisections + history_length + 2  # 2 * 8 + 5 + 2 = 23

    # Test first observation - should fill history with current pressure
    obs1 = _observe(env)
    @test length(obs1) == 2 * minisections + history_length + 2

    # Check that pressure history is filled with same value (first observation)
    pressure_history = obs1[(2 * minisections + 1):(2 * minisections + history_length)]
    @test all(pressure_history .≈ pressure_history[1])  # All values should be the same

    # Test subsequent observations with different pressures
    # Step 1: Set pressure to 0.2
    action1 = 0.2f0
    _act!(env, action1)
    obs2 = _observe(env)
    pressure_history2 = obs2[(2 * minisections + 1):(2 * minisections + history_length)]

    # Step 2: Set pressure to 2.0
    action2 = 0.5f0
    _act!(env, action2)
    obs3 = _observe(env)
    pressure_history3 = obs3[(2 * minisections + 1):(2 * minisections + history_length)]

    # Step 3: Set pressure to 2.5
    action3 = 0.8f0
    _act!(env, action3)
    obs4 = _observe(env)
    pressure_history4 = obs4[(2 * minisections + 1):(2 * minisections + history_length)]

    # Test that history shifts correctly
    # The newest pressure should be at the end, oldest should drop off
    @test pressure_history4[end] ≈ 0.8 / env.u_pmax  # Newest pressure (normalized)
    @test pressure_history4[end - 1] ≈ 0.5 / env.u_pmax  # Previous pressure
    @test pressure_history4[end - 2] ≈ 0.2 / env.u_pmax  # Earlier pressure

    # Test reset behavior
    _reset!(env)
    obs_reset = _observe(env)
    pressure_history_reset = obs_reset[(2 * minisections + 1):(2 * minisections + history_length)]

    # After reset, history should be filled with current pressure again
    @test all(pressure_history_reset .≈ pressure_history_reset[1])  # All values should be the same
end

@testitem "Target gated pressure TV reward" begin
    using RDE
    using RDE_Env
    using DrillInterface
    using DRL_RDE.RDE_Env_Strategies

    n_multisteps = 4
    params = RDEParam(; N = 512, tmax = 1.0f0)
    rew = TargetGatedPressureTVReward()
    cache = NoCache()
    meanΔ = DRL_RDE.RDE_Env_Strategies.mean_adjacent_pressure_delta
    knotsΔ = DRL_RDE.RDE_Env_Strategies.committed_knot_pressures

    @test meanΔ(Float32[0.0, 1.2, 0.0, 1.2]) ≈ 1.2f0
    @test meanΔ(Float32[0.6, 0.6, 0.6]) ≈ 0.0f0

    function make_env(start_shocks)
        env = RDEEnv(
            params;
            dt = 1.0f0,
            observation_strat = FourierMagnitudePressureShockObservation(n_modes = 8),
            action_strat = LinearMultiStepPressureAction(; n_sections = 1, n_steps = n_multisteps),
            reward_strat = rew,
            context_strat = FixedTargetShockCountStrategy(2),
            reset_strategy = NShock(start_shocks),
            verbose = false,
        )
        _reset!(env)
        return env
    end

    env_hold = make_env(2)
    u_hold = mean_u_p(env_hold.prob.injection)
    a_hold = 2.0f0 * u_hold / env_hold.u_pmax - 1.0f0
    act!(env_hold, fill(a_hold, n_multisteps))
    @test meanΔ(knotsΔ(env_hold.prob.injection)) < 0.02f0 * env_hold.u_pmax
    dx = RDE.get_dx(env_hold.prob)
    n_hold = RDE.count_shocks(@view(env_hold.state[1:params.N]), dx)
    if n_hold == 2
        @test RDE_Env.compute_reward(env_hold, rew, cache, env_hold.cache.context) ≈ 1.0f0 atol = 0.02f0
    end

    env_bang = make_env(2)
    act!(env_bang, Float32[-1, 1, -1, 1])
    @test meanΔ(knotsΔ(env_bang.prob.injection)) > 0.5f0 * env_bang.u_pmax
    n_bang = RDE.count_shocks(@view(env_bang.state[1:params.N]), RDE.get_dx(env_bang.prob))
    if n_bang == 2
        @test RDE_Env.compute_reward(env_bang, rew, cache, env_bang.cache.context) < 0.5f0
    end

    env_off = make_env(1)
    act!(env_off, Float32[-1, 1, -1, 1])
    n_off = RDE.count_shocks(@view(env_off.state[1:params.N]), RDE.get_dx(env_off.prob))
    @test n_off != 2
    @test RDE_Env.compute_reward(env_off, rew, cache, env_off.cache.context) == 1.0f0
end

@testitem "Target gated pressure TV reward, linear action" begin
    using RDE
    using RDE_Env
    using DrillInterface
    using DRL_RDE.RDE_Env_Strategies

    params = RDEParam(; N = 512, tmax = 1.0f0)
    rew = TargetGatedPressureTVReward()

    function make_linear_env(start_shocks)
        env = RDEEnv(
            params;
            dt = 1.0f0,
            observation_strat = FourierMagnitudePressureShockObservation(n_modes = 8),
            action_strat = LinearScalarPressureAction(),
            reward_strat = rew,
            context_strat = FixedTargetShockCountStrategy(2),
            reset_strategy = NShock(start_shocks),
            verbose = false,
        )
        _reset!(env)
        return env
    end

    env_hold = make_linear_env(2)
    cache = env_hold.cache.reward_cache
    @test cache isa PressureStepTVCache
    u_hold = mean_u_p(env_hold.prob.injection)
    a_hold = 2.0f0 * u_hold / env_hold.u_pmax - 1.0f0
    act!(env_hold, a_hold)
    act!(env_hold, a_hold)
    dx = RDE.get_dx(env_hold.prob)
    n_hold = RDE.count_shocks(@view(env_hold.state[1:params.N]), dx)
    if n_hold == 2
        @test env_hold.reward ≈ 1.0f0 atol = 0.02f0
    end

    env_bang = make_linear_env(2)
    act!(env_bang, -1.0f0)
    up_low = mean_u_p(env_bang.prob.injection)
    act!(env_bang, 1.0f0)
    n_bang = RDE.count_shocks(@view(env_bang.state[1:params.N]), RDE.get_dx(env_bang.prob))
    if n_bang == 2
        probe = PressureStepTVCache{Float32}(up_low)
        @test RDE_Env.compute_reward(env_bang, rew, probe, env_bang.cache.context) < 0.5f0
    end

    env_off = make_linear_env(1)
    act!(env_off, -1.0f0)
    act!(env_off, 1.0f0)
    n_off = RDE.count_shocks(@view(env_off.state[1:params.N]), RDE.get_dx(env_off.prob))
    @test n_off != 2
    @test RDE_Env.compute_reward(env_off, rew, env_off.cache.reward_cache, env_off.cache.context) == 1.0f0
end

@testitem "Step cost reward subtracts a constant" begin
    using RDE
    using RDE_Env
    using DrillInterface
    using DRL_RDE.RDE_Env_Strategies

    inner = TargetGatedPressureTVReward()
    rew = StepCostReward(inner; cost = 0.5f0)
    params = RDEParam(; N = 512, tmax = 1.0f0)
    env = RDEEnv(
        params;
        dt = 1.0f0,
        observation_strat = FourierMagnitudePressureShockObservation(n_modes = 8),
        action_strat = LinearScalarPressureAction(),
        reward_strat = rew,
        context_strat = FixedTargetShockCountStrategy(2),
        reset_strategy = NShock(1),
        verbose = false,
    )
    _reset!(env)
    act!(env, 0.0f0)
    @test env.reward ≈ RDE_Env.compute_reward(
        env, inner, NoCache(), env.cache.context
    ) - 0.5f0
end

@testitem "Target match reward is 0 off-target and 1 on-target" begin
    using RDE
    using RDE_Env
    using DrillInterface
    using DRL_RDE.RDE_Env_Strategies

    params = RDEParam(; N = 512, tmax = 1.0f0)
    rew = TargetMatchReward()

    function make_env(start_shocks)
        env = RDEEnv(
            params;
            dt = 1.0f0,
            observation_strat = FourierMagnitudePressureShockObservation(n_modes = 8),
            action_strat = LinearScalarPressureAction(),
            reward_strat = rew,
            context_strat = FixedTargetShockCountStrategy(2),
            reset_strategy = NShock(start_shocks),
            verbose = false,
        )
        _reset!(env)
        return env
    end

    env_on = make_env(2)
    @test env_on.reward == 1.0f0
    env_off = make_env(1)
    @test env_off.reward == 0.0f0

    locked = DenseLockReward(rew; bonus = 150.0f0)
    env_wrap = RDEEnv(
        params;
        dt = 1.0f0,
        observation_strat = FourierMagnitudePressureShockObservation(n_modes = 8),
        action_strat = LinearScalarPressureAction(),
        reward_strat = locked,
        context_strat = FixedTargetShockCountStrategy(2),
        reset_strategy = NShock(1),
        verbose = false,
    )
    _reset!(env_wrap)
    act!(env_wrap, 0.0f0)
    @test env_wrap.reward == 0.0f0
    @test !env_wrap.terminated
end

@testitem "Observation Type Consistency" begin
    using RDE

    # Test for Float32
    env = RDEEnv(params = RDEParam{Float32}())
    _reset!(env)
    obs = _observe(env)
    @test eltype(obs) == Float32

    # Test for Float64
    env = RDEEnv(RDEParam{Float64}())
    _reset!(env)
    obs = _observe(env)
    @test eltype(obs) == Float64

    # Test type consistency across different observation strategies
    for T in [Float32, Float64]
        for strategy in [
                FourierObservation(4),
                StateObservation(),
                SampledStateObservation(8),
            ]
            env1 = RDEEnv(RDEParam{T}(), observation_strat = strategy)
            _reset!(env1)
            obs1 = _observe(env1)
            @test eltype(obs1) == T
        end
    end
end
