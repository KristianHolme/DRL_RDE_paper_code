@kwdef mutable struct ScalarPressureAction{T <: AbstractFloat} <: AbstractScalarActionStrategy
    momentum::T = 0.0f0
end


@kwdef mutable struct ScalarAreaScalarPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    momentum::T = 0.0f0
end


@kwdef mutable struct VectorPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1 #number of sections
    momentum::T = 0.0f0
end
struct VectorActionCache{T <: AbstractFloat} <: AbstractCache
    section_controls::Vector{T}
    field::Vector{T}
end

mutable struct PIDActionCache{T <: AbstractFloat} <: AbstractCache
    integral::T
    previous_error::T
end

function RDE_Env.reset_cache!(cache::PIDActionCache{T}) where {T}
    cache.integral = zero(T)
    cache.previous_error = zero(T)
    return nothing
end

"""
    LinearScalarPressureAction <: AbstractActionStrategy

Action type where a single scalar action in `[-1, 1]` directly maps linearly
to pressure control in `[0, u_pmax]` via: `control = 0.5 * u_pmax * (action + 1)`.
Unlike `ScalarPressureAction`, the mapping is constant and independent of
the previous control state.
"""
@kwdef struct LinearScalarPressureAction{T <: AbstractFloat} <: AbstractScalarActionStrategy
    momentum::T = 0.0f0
end


"""
    LinearVectorPressureAction <: AbstractActionStrategy

Vector action type where each section's action in `[-1, 1]` directly maps linearly
to pressure control in `[0, u_pmax]` via: `control = 0.5 * u_pmax * (action + 1)`.
Unlike `VectorPressureAction`, the mapping is constant and independent of
the previous control state.

# Fields
- `n_sections::Int`: Number of sections to control independently (default: 1)
"""
@kwdef struct LinearVectorPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1
    momentum::T = 0.0f0
end

"""
    LinearMultiStepPressureAction <: AbstractActionStrategy

Multi-step action type where each knot's action in `[-1, 1]` maps linearly to
pressure control in `[0, u_pmax]` via: `control = 0.5 * u_pmax * (action + 1)`.

Unlike `RDE_Env.MultiStepPressureAction`, which takes `u_p` directly on
`[0, u_pmax]`, the action space here is centred: a zero-mean policy starts at
`u_pmax / 2` instead of at the lower boundary, and a Gaussian of unit width
covers the box without saturating on the clamp.

Actions are laid out as `n_steps` consecutive blocks of `n_sections` entries.

# Fields
- `n_sections::Int`: Number of sections to control independently (default: 1)
- `n_steps::Int`: Number of injection knots per environment step (default: 1)
"""
@kwdef struct LinearMultiStepPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1
    n_steps::Int = 1
    momentum::T = 0.0f0
end

"""
    LinearScalarPressureWithDtAction{T} <: AbstractVectorActionStrategy

Action type that combines linear pressure control with timestep (dt) control.
Two action elements: [pressure_action, dt_action] ∈ [-1,1]²

- pressure_action: Maps linearly to pressure control in [0, u_pmax]
- dt_action: Maps linearly to dt in [dt_min, dt_max]

# Fields
- `dt_min::T`: Minimum allowed timestep (default: 0.1)
- `dt_max::T`: Maximum allowed timestep (default: 20.0)
- `momentum::T`: Optional momentum parameter for pressure control (default: 0.0)
"""
@kwdef struct LinearScalarPressureWithDtAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    dt_min::T = 0.1f0
    dt_max::T = 20.0f0
    momentum::T = 0.0f0
end

"""
    TargetSafeLinearScalarPressureAction <: AbstractActionStrategy

Scalar action in `[-1, 1]` mapped piecewise through the mode-locked pressure
for the current target shock count (`RDE.SHOCK_PRESSURES[t]`):

- `a = 0` → `u_safe`
- `a ∈ [-1, 0]` → `[0, u_safe]`
- `a ∈ [0, 1]` → `[u_safe, u_pmax]`

This is a hold-at-target prior, not a two-step schedule.
"""
@kwdef struct TargetSafeLinearScalarPressureAction{T <: AbstractFloat} <: AbstractScalarActionStrategy
    momentum::T = 0.0f0
end

"""
    TargetSafeLinearMultiStepPressureAction <: AbstractActionStrategy

Same piecewise map as [`TargetSafeLinearScalarPressureAction`](@ref) on each
of `n_steps` knots. Layout is `n_steps` consecutive blocks of `n_sections`.
"""
@kwdef struct TargetSafeLinearMultiStepPressureAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    n_sections::Int = 1
    n_steps::Int = 2
    momentum::T = 0.0f0
end


"""
    PIDAction <: AbstractScalarActionStrategy

Action type where the agent supplies PID gains `[Kp, Ki, Kd]` and the
environment produces a scalar pressure action in `[-1, 1]` using a
built-in PID computation against a target value for the injection pressure.

State for PID accumulators (integral, previous_error) is stored in PIDActionCache.
Target value is derived from RDE.SHOCK_PRESSURES[target_shock_count] where
target_shock_count comes from env.cache.goal.

# Fields
- `momentum::T`: Optional momentum parameter (default: 0.0), can be used for smoothing
"""
@kwdef struct PIDAction{T <: AbstractFloat} <: AbstractVectorActionStrategy
    momentum::T = 0.0f0
end

function RDE_Env._action_space(env::RDEEnv, action_strat::ScalarAreaScalarPressureAction)
    return Drill.Box([-1.0f0, -1.0f0], [1.0f0, 1.0f0])
end

function RDE_Env._action_space(env::RDEEnv, action_strat::VectorPressureAction)
    return Drill.Box(fill(-1.0f0, action_strat.n_sections), fill(1.0f0, action_strat.n_sections))
end

function RDE_Env._action_space(env::RDEEnv, action_strat::PIDAction)
    return Drill.Box([-1.0f0, -1.0f0, -1.0f0], [1.0f0, 1.0f0, 1.0f0])
end

function RDE_Env.get_init_control_data(env::RDEEnv{T}, action_strat::PIDAction, max_steps::Int) where {T}
    return Vector{T}(undef, max_steps), Vector{T}(undef, max_steps)
end

function RDE_Env._action_space(env::RDEEnv, action_strat::LinearVectorPressureAction)
    return Drill.Box(fill(-1.0f0, action_strat.n_sections), fill(1.0f0, action_strat.n_sections))
end

function RDE_Env._action_space(::RDEEnv, action_strat::LinearMultiStepPressureAction)
    n = action_strat.n_sections * action_strat.n_steps
    return Drill.Box(fill(-1.0f0, n), fill(1.0f0, n))
end

function RDE_Env._action_space(::RDEEnv, ::TargetSafeLinearScalarPressureAction)
    return Drill.Box([-1.0f0], [1.0f0])
end

function RDE_Env._action_space(::RDEEnv, action_strat::TargetSafeLinearMultiStepPressureAction)
    n = action_strat.n_sections * action_strat.n_steps
    return Drill.Box(fill(-1.0f0, n), fill(1.0f0, n))
end

function RDE_Env._action_space(env::RDEEnv, action_strat::LinearScalarPressureWithDtAction)
    return Drill.Box([-1.0f0, -1.0f0], [1.0f0, 1.0f0])
end

function RDE_Env._action_space(::RDEEnv, ::LinearScalarPressureAction)
    return Drill.Box([-1.0f0], [1.0f0])
end

function RDE_Env._action_space(::RDEEnv, ::ScalarPressureAction)
    return Drill.Box([-1.0f0], [1.0f0])
end

# Scalar-action overloads (avoid building action vectors)
@inline function control_target(a::T, c_prev::T, c_max::T) where {T <: AbstractFloat}
    target = ifelse(a < 0, c_prev * (a + one(T)), c_prev + (c_max - c_prev) * a)::T
    return target
end

#linear function from [-1,1] to [0, u_p_max]
@inline function linear_control_target(a::T, c_max::T) where {T <: AbstractFloat}
    target = T(0.5) * c_max * (a + T(1))
    return target
end

# a = 0 → u_safe; a = -1 → 0; a = 1 → u_pmax
@inline function target_safe_control_target(a::T, u_safe::T, u_pmax::T) where {T <: AbstractFloat}
    a_clamped = clamp(a, -one(T), one(T))
    if a_clamped >= zero(T)
        return u_safe + a_clamped * (u_pmax - u_safe)
    else
        return u_safe + a_clamped * u_safe
    end
end

function target_safe_action_to_control(a::T, c_prev::T, u_safe::T, u_pmax::T, momentum::T) where {T <: AbstractFloat}
    return momentum_target(target_safe_control_target(a, u_safe, u_pmax), c_prev, momentum)
end

function control_to_target_safe_action(u::T, u_safe::T, u_pmax::T) where {T <: AbstractFloat}
    if u >= u_safe
        denom = u_pmax - u_safe
        return denom > zero(T) ? clamp((u - u_safe) / denom, zero(T), one(T)) : zero(T)
    else
        return u_safe > zero(T) ? clamp((u - u_safe) / u_safe, -one(T), zero(T)) : -one(T)
    end
end

function target_safe_pressure(env::RDEEnv{T}) where {T <: AbstractFloat}
    return T(RDE.SHOCK_PRESSURES[get_target_shock_count(env)])
end

function momentum_target(control_target::T, previous_target::T, momentum::T) where {T <: AbstractFloat}
    return momentum * previous_target + (one(T) - momentum) * control_target
end

# translate action ∈ [-1,1] to a target injection pressure, using a piecewise linear function, where 0 maps to the current control, -1 to 0 and 1 to the max allowed control
@inline function action_to_control(a::T, c_prev::T, c_max::T, momentum::T) where {T <: AbstractFloat}
    target::T = control_target(a, c_prev, c_max)
    return momentum * c_prev + (one(T) - momentum) * target
end

function linear_action_to_control(a::T, c_prev::T, c_max::T, momentum::T) where {T <: AbstractFloat}
    control_target = linear_control_target(a, c_max)
    return momentum_target(control_target, c_prev, momentum)
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::AbstractVector{T},
        action_strat::VectorPressureAction,
        action_cache::VectorActionCache{T},
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: VectorPressureAction, O, RW, G, V, OBS, M, RS, C}
    N = env.prob.params.N
    @assert N > 0 "Action type N not set"
    @assert length(action) == action_strat.n_sections "Action length ($(length(action))) must match n_sections ($(action_strat.n_sections))"
    @assert N % action_strat.n_sections == 0 "N ($(N)) must be divisible by n_sections ($(action_strat.n_sections))"

    if any(abs.(action) .> one(T))
        @warn "action out of bounds [-1,1]"
    end

    current = current_u_p(env.prob.injection)
    if current isa AbstractVector
        current_section_controls = RDE_Env.section_midpoint_values(current, action_strat.n_sections)
    else
        current_section_controls = fill(current, action_strat.n_sections)
    end
    section_controls = action_cache.section_controls
    section_controls .= action_to_control.(action, current_section_controls, env.u_pmax, RDE_Env.momentum(action_strat))
    if action_strat.n_sections == 1
        RDE_Env.commit_uniform_u_p!(env, section_controls[1])
    else
        RDE_Env.commit_section_u_p!(env, section_controls, action_cache.field)
    end
    return nothing
end

# Specialization: ScalarPressureAction — scalar action updates only u_p channel
function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::T,
        action_strat::ScalarPressureAction,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: ScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    if abs(action) > one(T)
        @warn "action (u_p) out of bounds [-1,1]"
    end
    prev = current_u_p(env.prob.injection)
    prev_s = prev isa AbstractVector ? prev[1] : prev
    target = action_to_control(action, prev_s, env.u_pmax, RDE_Env.momentum(action_strat))
    RDE_Env.commit_uniform_u_p!(env, target)
    return nothing
end

# Convenience: accept array-like scalar for ScalarPressureAction
function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::AbstractArray{T},
        action_strat::ScalarPressureAction,
        action_cache::AbstractCache,
        context::AbstractCache,
    ) where {T <: AbstractFloat, A <: ScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    return RDE_Env.apply_action!(env, action[1], action_strat, action_cache, context)
end

# Specialization: ScalarAreaScalarPressureAction — two scalar actions (s, u_p)
function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::AbstractVector{T},
        action_strat::ScalarAreaScalarPressureAction,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: ScalarAreaScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 2
    a_s, a_up = action[1], action[2]
    if abs(a_s) > one(T) || abs(a_up) > one(T)
        @warn "action (s/u_p) out of bounds [-1,1]"
    end
    prev_u = current_u_p(env.prob.injection)
    prev_u_s = prev_u isa AbstractVector ? prev_u[1] : prev_u
    prev_s = current_s(env.prob.injection)
    s_target = action_to_control(a_s, prev_s, env.smax, RDE_Env.momentum(action_strat))
    u_target = action_to_control(a_up, prev_u_s, env.u_pmax, RDE_Env.momentum(action_strat))
    RDE_Env.commit_uniform_u_p!(env, u_target; s = s_target)
    return nothing
end

# function apply_action!(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, action::NTuple{2, T}) where {T <: AbstractFloat, A <: ScalarAreaScalarPressureAction, O, RW, V, OBS, M, RS, C}
#     return apply_action!(env, collect(action))
# end

# Specialization: PIDAction — gains => scalar u_p set via PID, s unchanged
function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C},
        gains::AbstractVector{T},
        action_strat::PIDAction,
        action_cache::PIDActionCache{T},
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: PIDAction, O, RW, CS <: AbstractShockCountContextStrategy, V, OBS, M, RS, C}
    @assert length(gains) == 3 "PIDAction expects [Kp, Ki, Kd]"
    Kp, Ki, Kd = gains

    u_p_mean::T = mean_u_p(env.prob.injection)

    # Get cache and target from goal
    target::T = RDE.SHOCK_PRESSURES[get_target_shock_count(env)]

    # PID terms (keep in T)
    err::T = target - u_p_mean
    action_cache.integral += err * env.dt
    deriv::T = (err - action_cache.previous_error) / env.dt
    u_p_action::T = clamp(Kp * err + Ki * action_cache.integral + Kd * deriv, -one(T), one(T))
    action_cache.previous_error = err

    prev = current_u_p(env.prob.injection)
    prev_s = prev isa AbstractVector ? prev[1] : prev
    u_target = action_to_control(u_p_action, prev_s, env.u_pmax, RDE_Env.momentum(action_strat))
    RDE_Env.commit_uniform_u_p!(env, u_target)
    return nothing
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::Vector{T},
        action_strat::LinearScalarPressureAction,
        action_cache::AbstractCache,
        context::AbstractCache,
    ) where {T <: AbstractFloat, A <: LinearScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 1 "LinearScalarPressureAction expects a single action"
    RDE_Env.apply_action!(env, action[1], action_strat, action_cache, context)
    return nothing
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::T,
        action_strat::LinearScalarPressureAction,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: LinearScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    if abs(action) > one(T)
        @warn "action (u_p) out of bounds [-1,1]"
    end
    prev = current_u_p(env.prob.injection)
    prev_s = prev isa AbstractVector ? prev[1] : prev
    target = linear_action_to_control(action, prev_s, env.u_pmax, RDE_Env.momentum(action_strat))
    RDE_Env.commit_uniform_u_p!(env, target)
    return nothing
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::AbstractVector{T},
        action_strat::LinearVectorPressureAction,
        action_cache::VectorActionCache{T},
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: LinearVectorPressureAction, O, RW, G, V, OBS, M, RS, C}
    N = env.prob.params.N
    @assert N > 0 "Action type N not set"
    @assert length(action) == action_strat.n_sections "Action length ($(length(action))) must match n_sections ($(action_strat.n_sections))"
    @assert N % action_strat.n_sections == 0 "N ($(N)) must be divisible by n_sections ($(action_strat.n_sections))"

    if any(abs.(action) .> one(T))
        @warn "action out of bounds [-1,1]"
    end

    current = current_u_p(env.prob.injection)
    if current isa AbstractVector
        current_section_controls = RDE_Env.section_midpoint_values(current, action_strat.n_sections)
    else
        current_section_controls = fill(current, action_strat.n_sections)
    end
    section_controls = action_cache.section_controls
    section_controls .= linear_action_to_control.(action, current_section_controls, env.u_pmax, RDE_Env.momentum(action_strat))
    if action_strat.n_sections == 1
        RDE_Env.commit_uniform_u_p!(env, section_controls[1])
    else
        RDE_Env.commit_section_u_p!(env, section_controls, action_cache.field)
    end
    return nothing
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::AbstractVector{T},
        action_strat::LinearMultiStepPressureAction,
        action_cache::RDE_Env.MultiStepActionCache{T},
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: LinearMultiStepPressureAction, O, RW, G, V, OBS, M, RS, C}
    n_sec = action_strat.n_sections
    n_steps = action_strat.n_steps
    expected = n_sec * n_steps
    @assert length(action) == expected "Action length ($(length(action))) must match n_sections * n_steps ($(expected))"

    if any(abs.(action) .> one(T))
        @warn "action out of bounds [-1,1]"
    end

    t0 = env.t
    step_dt = env.dt / T(n_steps)
    for k in 1:n_steps
        action_cache.times[k] = t0 + T(k) * step_dt
    end

    u_pmax = env.u_pmax
    mom = RDE_Env.momentum(action_strat)
    if n_sec == 1
        prev = current_u_p(env.prob.injection)
        prev_s = prev isa AbstractVector ? prev[1] : prev
        targets = action_cache.section_controls
        resize!(targets, n_steps)
        for k in 1:n_steps
            targets[k] = linear_action_to_control(action[k], prev_s, u_pmax, mom)
            prev_s = targets[k]
        end
        RDE.commit_schedule!(env.prob.injection, t0, action_cache.times, targets)
    else
        N = env.prob.params.N
        @assert N % n_sec == 0 "N ($(N)) must be divisible by n_sections ($(n_sec))"
        current = current_u_p(env.prob.injection)
        if current isa AbstractVector
            section_prev = RDE_Env.section_midpoint_values(current, n_sec)
        else
            section_prev = fill(current, n_sec)
        end
        targets = action_cache.fields
        section_controls = view(action_cache.section_controls, 1:n_sec)
        for k in 1:n_steps
            lo = (k - 1) * n_sec + 1
            chunk = view(action, lo:(k * n_sec))
            section_controls .= linear_action_to_control.(chunk, section_prev, u_pmax, mom)
            copyto!(section_prev, section_controls)
            RDE_Env.fill_sections!(targets[k], section_controls)
        end
        RDE.commit_schedule!(env.prob.injection, t0, action_cache.times, targets)
    end
    return nothing
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::AbstractVector{T},
        action_strat::LinearScalarPressureWithDtAction,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: LinearScalarPressureWithDtAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 2 "LinearScalarPressureWithDtAction expects two actions: [pressure_action, dt_action]"
    a_pressure, a_dt = action[1], action[2]

    if abs(a_pressure) > one(T)
        @warn "pressure action out of bounds [-1,1]"
    end
    if abs(a_dt) > one(T)
        @warn "dt action out of bounds [-1,1]"
    end

    prev = current_u_p(env.prob.injection)
    prev_s = prev isa AbstractVector ? prev[1] : prev
    u_target = linear_action_to_control(a_pressure, prev_s, env.u_pmax, RDE_Env.momentum(action_strat))
    RDE_Env.commit_uniform_u_p!(env, u_target)

    dt_range = action_strat.dt_max - action_strat.dt_min
    env.dt = action_strat.dt_min + T(0.5) * dt_range * (a_dt + one(T))
    return nothing
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::Vector{T},
        action_strat::TargetSafeLinearScalarPressureAction,
        action_cache::AbstractCache,
        context::AbstractCache,
    ) where {T <: AbstractFloat, A <: TargetSafeLinearScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    @assert length(action) == 1 "TargetSafeLinearScalarPressureAction expects a single action"
    RDE_Env.apply_action!(env, action[1], action_strat, action_cache, context)
    return nothing
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::T,
        action_strat::TargetSafeLinearScalarPressureAction,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: TargetSafeLinearScalarPressureAction, O, RW, G, V, OBS, M, RS, C}
    if abs(action) > one(T)
        @warn "action (u_p) out of bounds [-1,1]"
    end
    prev = current_u_p(env.prob.injection)
    prev_s = prev isa AbstractVector ? prev[1] : prev
    u_safe = target_safe_pressure(env)
    target = target_safe_action_to_control(action, prev_s, u_safe, env.u_pmax, RDE_Env.momentum(action_strat))
    RDE_Env.commit_uniform_u_p!(env, target)
    return nothing
end

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        action::AbstractVector{T},
        action_strat::TargetSafeLinearMultiStepPressureAction,
        action_cache::RDE_Env.MultiStepActionCache{T},
        ::AbstractCache,
    ) where {T <: AbstractFloat, A <: TargetSafeLinearMultiStepPressureAction, O, RW, G, V, OBS, M, RS, C}
    n_sec = action_strat.n_sections
    n_steps = action_strat.n_steps
    expected = n_sec * n_steps
    @assert length(action) == expected "Action length ($(length(action))) must match n_sections * n_steps ($(expected))"
    @assert n_sec == 1 "TargetSafeLinearMultiStepPressureAction is wired for n_sections=1"

    if any(abs.(action) .> one(T))
        @warn "action out of bounds [-1,1]"
    end

    t0 = env.t
    step_dt = env.dt / T(n_steps)
    for k in 1:n_steps
        action_cache.times[k] = t0 + T(k) * step_dt
    end

    prev = current_u_p(env.prob.injection)
    prev_s = prev isa AbstractVector ? prev[1] : prev
    u_safe = target_safe_pressure(env)
    u_pmax = env.u_pmax
    mom = RDE_Env.momentum(action_strat)
    targets = action_cache.section_controls
    resize!(targets, n_steps)
    for k in 1:n_steps
        targets[k] = target_safe_action_to_control(action[k], prev_s, u_safe, u_pmax, mom)
        prev_s = targets[k]
    end
    RDE.commit_schedule!(env.prob.injection, t0, action_cache.times, targets)
    return nothing
end

# Action cache initializers for vector action types
RDE_Env.initialize_cache(at::VectorPressureAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    VectorActionCache{T}(Vector{T}(undef, at.n_sections), Vector{T}(undef, N))

RDE_Env.initialize_cache(at::LinearVectorPressureAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    VectorActionCache{T}(Vector{T}(undef, at.n_sections), Vector{T}(undef, N))

function RDE_Env.initialize_cache(
        at::LinearMultiStepPressureAction{T}, N::Int, ::Type{T}
    ) where {T <: AbstractFloat}
    n_steps = at.n_steps
    return RDE_Env.MultiStepActionCache{T}(
        Vector{T}(undef, max(at.n_sections, n_steps)),
        [zeros(T, N) for _ in 1:n_steps],
        Vector{T}(undef, n_steps),
    )
end

function RDE_Env.initialize_cache(
        at::TargetSafeLinearMultiStepPressureAction{T}, N::Int, ::Type{T}
    ) where {T <: AbstractFloat}
    n_steps = at.n_steps
    return RDE_Env.MultiStepActionCache{T}(
        Vector{T}(undef, max(at.n_sections, n_steps)),
        [zeros(T, N) for _ in 1:n_steps],
        Vector{T}(undef, n_steps),
    )
end

RDE_Env.initialize_cache(::PIDAction{T}, N::Int, ::Type{T}) where {T <: AbstractFloat} =
    PIDActionCache{T}(zero(T), zero(T))
