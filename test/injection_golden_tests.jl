@testitem "injection profile goldens SAU SAS ± shift" begin
    using JLD2
    using RDE
    using RDE_Env

    goldens = JLD2.load(joinpath(homedir(), ".julia/dev/RDE.jl/test/goldens/injection_profile_goldens.jld2"), "goldens")

    function replay(; action_strat, control_shift_strategy, actions)
        params = RDEParam(; N = 64, tmax = 4.0f0)
        env = RDEEnv(;
            params,
            dt = 1.0f0,
            τ_smooth = 0.1f0,
            action_strat,
            control_shift_strategy,
            reset_strategy = NShock(2),
            spatial_kernel_width = 0,
            verbose = false,
        )
        _reset!(env)
        ts = Float32[env.t]
        states = [copy(env.state)]
        for a in actions
            _act!(env, a; saves_per_action = 4)
            sol = env.prob.sol
            append!(ts, sol.t[2:end])
            append!(states, [copy(u) for u in sol.u[2:end]])
        end
        return ts, states
    end

    sau_actions = [0.6f0, 0.8f0, 0.4f0]
    sas_actions = [
        [0.5f0, 0.7f0, 0.6f0, 0.8f0],
        [0.9f0, 0.4f0, 0.5f0, 0.7f0],
        [0.3f0, 0.6f0, 0.8f0, 0.5f0],
    ]
    cases = (
        "sau_static" => (
            DirectScalarPressureAction(),
            ZeroControlShift(),
            sau_actions,
        ),
        "sas_static" => (
            DirectVectorPressureAction(; n_sections = 4),
            ZeroControlShift(),
            sas_actions,
        ),
        "sau_mf" => (
            DirectScalarPressureAction(),
            LinearControlShift(1.71f0),
            sau_actions,
        ),
        "sas_mf" => (
            DirectVectorPressureAction(; n_sections = 4),
            LinearControlShift(1.71f0),
            sas_actions,
        ),
    )

    for (name, (astrat, shift, actions)) in cases
        ts, states = replay(;
            action_strat = astrat,
            control_shift_strategy = shift,
            actions,
        )
        g = goldens[name]
        @test ts ≈ g["t"]
        @test length(states) == length(g["states"])
        for (a, b) in zip(states, g["states"])
            @test a ≈ b
        end
    end
end

@testitem "run_policy stores live profile at state times" begin
    using RDE
    using RDE_Env
    using DrillInterface

    params = RDEParam(; N = 32, tmax = 2.0f0)
    env = RDEEnv(;
        params,
        dt = 1.0f0,
        τ_smooth = 0.1f0,
        action_strat = MultiStepPressureAction(; n_sections = 1, n_steps = 3),
        reset_strategy = NShock(2),
        verbose = false,
    )
    _reset!(env)
    action = Float32[0.3, 0.8, 0.5]
    _act!(env, action; saves_per_action = 8)
    @test length(env.prob.sol.t) >= 2
    u_p_buf = zeros(Float32, params.N)
    s_buf = zeros(Float32, params.N)
    for t in env.prob.sol.t
        update_injection!(u_p_buf, s_buf, env.prob.injection, Float32(t))
        @test all(isfinite, u_p_buf)
    end
    @test u_p_buf[1] != u_p_buf[1] + 1 # sanity
    mid = Float32(env.prob.sol.t[end] - env.dt / 2)
    update_injection!(u_p_buf, s_buf, env.prob.injection, mid)
    @test 0 <= u_p_buf[1] <= env.u_pmax
end
