"""
    ConstantRDEPolicy <: AbstractRDEPolicy

Policy that maintains constant control values.

# Fields
- `env::RDEEnv`: RDE environment

# Notes
Returns [0.0, 0.0] for ScalarAreaScalarPressureAction
Returns 0.0 for ScalarPressureAction
"""
struct ConstantRDEPolicy <: AbstractRDEPolicy
    env::RDEEnv
end

get_env(π::ConstantRDEPolicy) = π.env

function RDE_Env._predict_action(π::ConstantRDEPolicy, s::Vector{T}) where {T <: AbstractFloat}
    if π.env.action_strat isa ScalarAreaScalarPressureAction
        return zeros(T, 2)
    elseif π.env.action_strat isa ScalarPressureAction
        return zero(T)
    elseif π.env.action_strat isa VectorPressureAction
        return zeros(T, π.env.action_strat.n_sections)
    elseif π.env.action_strat isa DirectScalarPressureAction
        u_p = mean_u_p(π.env.prob.injection)
        return clamp(u_p, zero(T), π.env.u_pmax)
    elseif π.env.action_strat isa DirectVectorPressureAction
        u_p_current = current_u_p(π.env.prob.injection)
        controls = if u_p_current isa AbstractVector
            RDE_Env.section_midpoint_values(u_p_current, π.env.action_strat.n_sections)
        else
            fill(u_p_current, π.env.action_strat.n_sections)
        end
        return clamp.(controls, zero(T), π.env.u_pmax)
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for ConstantRDEPolicy"
    end
end
Base.show(io::IO, π::ConstantRDEPolicy) = print(io, "ConstantRDEPolicy()")
Base.show(io::IO, ::MIME"text/plain", π::ConstantRDEPolicy) = println(io, "ConstantRDEPolicy()")
"""
    SinusoidalRDEPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

Policy that applies sinusoidal control signals.

# Fields
- `env::RDEEnv{T}`: RDE environment
- `w_1::T`: Phase speed parameter for first action
- `w_2::T`: Phase speed parameter for second action

# Constructor
```julia
SinusoidalRDEPolicy(env::RDEEnv{T}; w_1::T=1.0, w_2::T=2.0) where {T<:AbstractFloat}
```
"""
struct SinusoidalRDEPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    w_1::T
    w_2::T
    scale::T
    function SinusoidalRDEPolicy(env::RDEEnv{T}; w_1::T = 1.0f0, w_2::T = 2.0f0, scale = 0.2f0) where {T <: AbstractFloat}
        return new{T}(env, w_1, w_2, scale)
    end
end

get_env(π::SinusoidalRDEPolicy) = π.env

function RDE_Env._predict_action(π::SinusoidalRDEPolicy, s::AbstractVector{T}) where {T <: AbstractFloat}
    t = π.env.t
    action1 = T(sin(π.w_1 * t)) * π.scale
    action2 = T(sin(π.w_2 * t)) * π.scale
    if π.env.action_strat isa ScalarAreaScalarPressureAction
        return [action1, action2]
    elseif π.env.action_strat isa ScalarPressureAction
        return action2
    elseif π.env.action_strat isa DirectScalarPressureAction
        base = mean_u_p(π.env.prob.injection)
        delta = π.scale * π.env.u_pmax * T(sin(π.w_2 * t))
        return clamp(base + delta, zero(T), π.env.u_pmax)
    elseif π.env.action_strat isa DirectVectorPressureAction
        base = mean_u_p(π.env.prob.injection)
        delta = π.scale * π.env.u_pmax * T(sin(π.w_2 * t))
        action_value = clamp(base + delta, zero(T), π.env.u_pmax)
        return fill(action_value, π.env.action_strat.n_sections)
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for SinusoidalRDEPolicy"
    end
end

function Base.show(io::IO, π::SinusoidalRDEPolicy)
    return print(io, "SinusoidalRDEPolicy()")
end

function Base.show(io::IO, ::MIME"text/plain", π::SinusoidalRDEPolicy)
    println(io, "SinusoidalRDEPolicy:")
    println(io, "  w₁: $(π.w_1)")
    println(io, "  w₂: $(π.w_2)")
    println(io, "  scale: $(π.scale)")
    return println(io, "  env: $(typeof(π.env))")
end

"""
    StepwiseRDEPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

Policy that applies predefined control values at specified times.

# Fields
- `env::RDEEnv{T}`: RDE environment
- `ts::Vector{T}`: Vector of time steps
- `c::Vector{Vector{T}}`: Vector of control actions

# Notes
- Only supports ScalarAreaScalarPressureAction
- Requires sorted time steps
- Each control action must have 2 elements
"""
struct StepwiseRDEPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    ts::Vector{T}  # Vector of time steps
    c::Union{Vector{Vector{T}}, Vector{T}}  # Vector of control actions

    function StepwiseRDEPolicy(env::RDEEnv{T}, ts::Vector{T}, c::Union{Vector{Vector{T}}, Vector{T}}) where {T <: AbstractFloat}
        @assert env.action_strat isa DirectScalarPressureAction || env.action_strat isa DirectVectorPressureAction "StepwiseRDEPolicy requires direct action types"
        @assert length(ts) == length(c) "Length of time steps and control actions must be equal"
        @assert issorted(ts) "Time steps must be in ascending order"
        if env.action_strat isa DirectVectorPressureAction
            @assert eltype(c) <: Vector{T} "Control actions must be a vector of vectors for DirectVectorPressureAction"
            @assert all(length(action) == env.action_strat.n_sections for action in c) "Each control action must match n_sections"
        else
            @assert eltype(c) <: T "Control actions must be a vector of scalars for DirectScalarPressureAction"
        end
        return new{T}(env, ts, c)
    end
end

get_env(π::StepwiseRDEPolicy) = π.env

function RDE_Env._predict_action(π::StepwiseRDEPolicy, s::AbstractVector{T}) where {T <: AbstractFloat}
    t = π.env.t
    past = π.ts .≤ t
    idx = findlast(past)

    # If before first time step, return current pressure (maintain status quo)
    if isnothing(idx)
        u_now = current_u_p(π.env.prob.injection)
        if π.env.action_strat isa DirectVectorPressureAction
            if u_now isa AbstractVector
                return RDE_Env.section_midpoint_values(u_now, π.env.action_strat.n_sections)
            else
                return fill(u_now, π.env.action_strat.n_sections)
            end
        elseif π.env.action_strat isa DirectScalarPressureAction
            return u_now isa AbstractVector ? u_now[1] : u_now
        end
    end

    if π.env.action_strat isa DirectVectorPressureAction
        u_pmax = π.env.u_pmax
        return clamp.(π.c[idx], zero(T), u_pmax)
    elseif π.env.action_strat isa DirectScalarPressureAction
        u_pmax = π.env.u_pmax
        return clamp(π.c[idx], zero(T), u_pmax)
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for StepwiseRDEPolicy"
    end
end

function Base.show(io::IO, π::StepwiseRDEPolicy)
    return print(io, "StepwiseRDEPolicy($(length(π.ts)) steps)")
end

function Base.show(io::IO, ::MIME"text/plain", π::StepwiseRDEPolicy)
    println(io, "StepwiseRDEPolicy:")
    println(io, "  steps: $(length(π.ts))")
    println(io, "  env: $(typeof(π.env))")
    return println(io, "  time range: [$(minimum(π.ts)), $(maximum(π.ts))]")
end

"""
    RandomRDEPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

Policy that applies random control values.

# Fields
- `env::RDEEnv{T}`: RDE environment

# Notes
Generates random values in [-1, 1] for each control dimension
"""
struct RandomRDEPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    scale::T
    function RandomRDEPolicy(env::RDEEnv{T}, scale::T = 1.0f0) where {T <: AbstractFloat}
        return new{T}(env, scale)
    end
end

get_env(π::RandomRDEPolicy) = π.env

function RDE_Env._predict_action(π::RandomRDEPolicy, state::AbstractVector{T}) where {T <: AbstractFloat}
    action1 = π.scale * (2 * rand(T) - 1)
    action2 = π.scale * (2 * rand(T) - 1)
    if π.env.action_strat isa ScalarAreaScalarPressureAction
        return [action1, action2]
    elseif π.env.action_strat isa ScalarPressureAction
        return action2
    elseif π.env.action_strat isa DirectScalarPressureAction
        return rand(T) * π.env.u_pmax
    elseif π.env.action_strat isa DirectVectorPressureAction
        return rand(T, π.env.action_strat.n_sections) .* π.env.u_pmax
    else
        @error "Unknown action type $(typeof(π.env.action_strat)) for RandomRDEPolicy"
    end
end

function Base.show(io::IO, π::RandomRDEPolicy)
    return print(io, "RandomRDEPolicy(scale=$(π.scale))")
end

function Base.show(io::IO, ::MIME"text/plain", π::RandomRDEPolicy)
    println(io, "RandomRDEPolicy:")
    return println(io, "  env: $(typeof(π.env))")
    return println(io, "  scale: $(π.scale)")
end

"""
    DelayedPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

A policy wrapper that delays the start of another policy until a specified time.
Before the start time, returns zero actions.

# Fields
- `policy::Policy`: The policy to be delayed
- `start_time::T`: Time at which to start using the wrapped policy
- `env::RDEEnv{T}`: RDE environment (needed to access current time)

# Example
```julia
base_policy = RandomRDEPolicy(env)
delayed_policy = DelayedPolicy(base_policy, 100.0f0, env)
```
"""
struct DelayedPolicy{T <: AbstractFloat, P <: Drill.AbstractPolicy} <: AbstractRDEPolicy
    policy::P
    start_time::T
    env::RDEEnv{T}
end

get_env(π::DelayedPolicy) = π.env

function RDE_Env._predict_action(π::DelayedPolicy, s::Union{AbstractVector{T}, Matrix{T}}) where {T <: AbstractFloat}
    t = π.env.t
    if t < π.start_time
        if π.env.action_strat isa ScalarAreaScalarPressureAction
            return [zeros(T, 2)]
        elseif π.env.action_strat isa ScalarPressureAction
            return zero(T)
        elseif π.env.action_strat isa VectorPressureAction
            return zeros(T, π.env.action_strat.n_sections)
        elseif π.env.action_strat isa DirectScalarPressureAction
            u_p = mean_u_p(π.env.prob.injection)
            return clamp(u_p, zero(T), π.env.u_pmax)
        elseif π.env.action_strat isa DirectVectorPressureAction
            u_p_current = current_u_p(π.env.prob.injection)
            controls = if u_p_current isa AbstractVector
                RDE_Env.section_midpoint_values(u_p_current, π.env.action_strat.n_sections)
            else
                fill(u_p_current, π.env.action_strat.n_sections)
            end
            return clamp.(controls, zero(T), π.env.u_pmax)
        elseif π.env.action_strat isa TargetSafeLinearScalarPressureAction
            u_p = mean_u_p(π.env.prob.injection)
            u_safe = RDE_Env_Strategies.target_safe_pressure(π.env)
            action = RDE_Env_Strategies.control_to_target_safe_action(u_p, u_safe, π.env.u_pmax)
            return [action]
        elseif π.env.action_strat isa TargetSafeLinearMultiStepPressureAction
            u_p = mean_u_p(π.env.prob.injection)
            u_safe = RDE_Env_Strategies.target_safe_pressure(π.env)
            action = RDE_Env_Strategies.control_to_target_safe_action(u_p, u_safe, π.env.u_pmax)
            n = π.env.action_strat.n_sections * π.env.action_strat.n_steps
            return fill(action, n)
        elseif π.env.action_strat isa LinearScalarPressureAction ||
                π.env.action_strat isa LinearVectorPressureAction ||
                π.env.action_strat isa LinearScalarPressureWithDtAction
            u_p = mean_u_p(π.env.prob.injection)
            u_pmax = π.env.u_pmax
            action = 2 * u_p / u_pmax - 1
            control_from_action = RDE_Env_Strategies.linear_control_target(action, u_pmax)
            @assert abs(u_p - control_from_action) < 1.0e-6 "action $(action) and control_from_action $(control_from_action) are not close"
            if π.env.action_strat isa LinearScalarPressureAction
                return [action]
            elseif π.env.action_strat isa LinearVectorPressureAction
                return fill(action, π.env.action_strat.n_sections)
            elseif π.env.action_strat isa LinearScalarPressureWithDtAction
                dt_max = π.env.action_strat.dt_max
                dt_min = π.env.action_strat.dt_min
                dt = π.env.dt
                progress = (dt - dt_min) / (dt_max - dt_min)
                time_action = 2progress - 1
                return [action, time_action]
            end
        elseif π.env.action_strat isa PIDAction
            return zeros(T, 3)
        elseif π.env.action_strat isa MultiStepPressureAction
            u_p = mean_u_p(π.env.prob.injection)
            held = clamp(u_p, zero(T), π.env.u_pmax)
            n = π.env.action_strat.n_sections * π.env.action_strat.n_steps
            return fill(held, n)
        elseif π.env.action_strat isa LinearMultiStepPressureAction
            u_p = mean_u_p(π.env.prob.injection)
            u_pmax = π.env.u_pmax
            action = clamp(2 * u_p / u_pmax - 1, -one(T), one(T))
            n = π.env.action_strat.n_sections * π.env.action_strat.n_steps
            return fill(action, n)
        else
            @error "Unknown action type $(typeof(π.env.action_strat)) for DelayedPolicy"
        end
    else
        return _predict_action(π.policy, s)
    end
end

function Base.show(io::IO, π::DelayedPolicy)
    return print(io, "DelayedPolicy(t>$(π.start_time))")
end

function Base.show(io::IO, ::MIME"text/plain", π::DelayedPolicy)
    println(io, "DelayedPolicy:")
    println(io, "  start_time: $(π.start_time)")
    println(io, "  policy: $(π.policy)")
    return println(io, "  env: $(typeof(π.env))")
end

"""
    HeldActionPolicy(policy, env, period)

Call `policy` once every `period` time units and reuse that action until the
next switch. Times lock detection at a finer `env.dt` than the controller step
without changing the closed-loop input.
"""
mutable struct HeldActionPolicy{P, E, T <: AbstractFloat} <: AbstractRDEPolicy
    policy::P
    env::E
    period::T
    last_switch::T
    last_action::Any
end

function HeldActionPolicy(
        policy,
        env::RDEEnv{T},
        period::Real,
    ) where {T <: AbstractFloat}
    p = T(period)
    return HeldActionPolicy{typeof(policy), typeof(env), T}(policy, env, p, -p, nothing)
end

get_env(π::HeldActionPolicy) = π.env

function RDE_Env._predict_action(π::HeldActionPolicy, s)
    t = π.env.t
    if π.last_action === nothing || t + eps(typeof(t)) >= π.last_switch + π.period
        π.last_action = RDE_Env._predict_action(π.policy, s)
        π.last_switch = t
    end
    return π.last_action
end

function Base.show(io::IO, π::HeldActionPolicy)
    return print(io, "HeldActionPolicy(period=$(π.period))")
end

function Base.show(io::IO, ::MIME"text/plain", π::HeldActionPolicy)
    println(io, "HeldActionPolicy:")
    println(io, "  period: $(π.period)")
    return println(io, "  policy: $(π.policy)")
end

struct ScaledPolicy{T <: AbstractFloat, P <: AbstractRDEPolicy} <: AbstractRDEPolicy
    policy::P
    scale::T
end

function get_env(π::ScaledPolicy)
    return get_env(π.policy)
end

function RDE_Env._predict_action(π::ScaledPolicy, s::AbstractVector{T}) where {T <: AbstractFloat}
    return π.scale::T .* _predict_action(π.policy, s)
end

function Base.show(io::IO, π::ScaledPolicy)
    return print(io, "ScaledPolicy(scale=$(π.scale))")
end

function Base.show(io::IO, ::MIME"text/plain", π::ScaledPolicy)
    println(io, "ScaledPolicy:")
    println(io, "  scale: $(π.scale)")
    return println(io, "  policy: $(π.policy)")
end

struct LinearPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    ts::Vector{T}  # Vector of time points
    c::Union{Vector{Vector{T}}, Vector{T}}  # Vector of control values at each time point

    function LinearPolicy(env::RDEEnv{T}, ts::Vector{T}, c::Union{Vector{Vector{T}}, Vector{T}}) where {T <: AbstractFloat}
        @assert env.action_strat isa DirectScalarPressureAction || env.action_strat isa DirectVectorPressureAction "LinearPolicy requires direct action types"
        @assert length(ts) == length(c) "Length of time points and control values must be equal"
        @assert issorted(ts) "Time points must be in ascending order"
        @assert length(ts) >= 2 "At least two time points are required for linear interpolation"
        if env.action_strat isa DirectVectorPressureAction
            @assert eltype(c) <: Vector{T} "Control values must be a vector of vectors for DirectVectorPressureAction"
            @assert all(length(action) == env.action_strat.n_sections for action in c) "Each control value must match n_sections"
        else
            @assert eltype(c) <: T "Control values must be a vector of scalars for DirectScalarPressureAction"
        end
        return new{T}(env, ts, c)
    end
end

get_env(π::LinearPolicy) = π.env

function RDE_Env._predict_action(π::LinearPolicy, s::AbstractVector)
    t = π.env.t
    cache = π.env.prob.method.cache

    # Find the time points to interpolate between
    past = π.ts .≤ t
    idx = findlast(past)

    if isnothing(idx)
        target_value = π.c[1]
    elseif idx == length(π.ts)
        target_value = π.c[end]
    else
        t1, t2 = π.ts[idx], π.ts[idx + 1]
        c1, c2 = π.c[idx], π.c[idx + 1]
        if π.env.action_strat isa DirectVectorPressureAction
            target_value = c1 .+ (c2 .- c1) .* (t - t1) / (t2 - t1)
        elseif π.env.action_strat isa DirectScalarPressureAction
            target_value = c1 + (c2 - c1) * (t - t1) / (t2 - t1)
        else
            error("Unknown action type $(typeof(π.env.action_strat)) for LinearPolicy")
        end
    end

    if π.env.action_strat isa DirectVectorPressureAction
        u_pmax = π.env.u_pmax
        return clamp.(target_value, zero(eltype(target_value)), u_pmax)
    elseif π.env.action_strat isa DirectScalarPressureAction
        u_pmax = π.env.u_pmax
        return clamp(target_value, zero(typeof(target_value)), u_pmax)
    else
        error("Unknown action type $(typeof(π.env.action_strat)) for LinearPolicy")
    end
end

function Base.show(io::IO, π::LinearPolicy)
    return print(io, "LinearPolicy($(length(π.ts)) points)")
end

function Base.show(io::IO, ::MIME"text/plain", π::LinearPolicy)
    println(io, "LinearPolicy:")
    println(io, "  points: $(length(π.ts))")
    println(io, "  env: $(typeof(π.env))")
    return println(io, "  time range: [$(minimum(π.ts)), $(maximum(π.ts))]")
end

"""
    SawtoothPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

Policy that implements a sawtooth control pattern for pressure control, periodically increasing from min to max value.

# Fields
- `env::RDEEnv{T}`: RDE environment
- `timescale::T`: Time period for one complete sawtooth cycle
- `max_value::T`: Maximum pressure value
- `min_value::T`: Minimum pressure value (value to drop to)

# Notes
- Only compatible with ScalarPressureAction
- Starts from the current injection pressure and gradually increases
"""
struct SawtoothPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    env::RDEEnv{T}
    timescale::T
    max_value::T
    min_value::T

    function SawtoothPolicy(env::RDEEnv{T}, timescale::T, max_value::T, min_value::T) where {T <: AbstractFloat}
        @assert env.action_strat isa DirectScalarPressureAction "SawtoothPolicy requires direct scalar action"
        @assert timescale > 0 "Timescale must be positive"
        @assert max_value > min_value "Max value must be greater than min value"
        return new{T}(env, timescale, max_value, min_value)
    end
end

get_env(π::SawtoothPolicy) = π.env

function RDE_Env._predict_action(π::SawtoothPolicy, s::AbstractVector)
    t = π.env.t
    cache = π.env.prob.method.cache

    # Calculate the current phase in the sawtooth cycle
    phase = mod(t, π.timescale)
    period_number = t ÷ π.timescale + 1
    # Calculate target value based on phase
    # Linear increase from min to max
    max_value = π.max_value
    min_value = period_number == 1 ? π.env.prob.params.u_p : π.min_value
    target_value = min_value + (max_value - min_value) * phase / π.timescale

    # Direct control target (clamped)
    return clamp(target_value, zero(typeof(target_value)), π.env.u_pmax)
end

function Base.show(io::IO, π::SawtoothPolicy)
    return print(io, "SawtoothPolicy(timescale=$(π.timescale))")
end

function Base.show(io::IO, ::MIME"text/plain", π::SawtoothPolicy)
    println(io, "SawtoothPolicy:")
    println(io, "  timescale: $(π.timescale)")
    println(io, "  max_value: $(π.max_value)")
    println(io, "  min_value: $(π.min_value)")
    return println(io, "  env: $(typeof(π.env))")
end
"""
    PIDControllerPolicy{T<:AbstractFloat} <: AbstractRDEPolicy

A policy that outputs constant PID gains for use with PIDAction environments.
The environment handles all PID computation (error, integral, derivative).

# Requirements
- Must be used with an environment that has PIDAction action type
- Environment derives target from RDE.SHOCK_PRESSURES[target_shock_count]

# Fields
- `Kp::T`: Proportional gain
- `Ki::T`: Integral gain
- `Kd::T`: Derivative gain

# Notes
- Policy is stateless - all PID state managed by environment's PIDActionCache
- To change target, use set_target_shock_count!(env, n)
"""
struct PIDControllerPolicy{T <: AbstractFloat} <: AbstractRDEPolicy
    Kp::T
    Ki::T
    Kd::T
end

# Convenience constructor with default Float32
PIDControllerPolicy(; Kp::Real, Ki::Real, Kd::Real) =
    PIDControllerPolicy{Float32}(Float32(Kp), Float32(Ki), Float32(Kd))

function RDE_Env._predict_action(π::PIDControllerPolicy, o::AbstractVector)
    # Simply return the PID gains - environment does the computation
    return [π.Kp, π.Ki, π.Kd]
end

function Base.show(io::IO, π::PIDControllerPolicy)
    return print(io, "PIDControllerPolicy(Kp=$(π.Kp), Ki=$(π.Ki), Kd=$(π.Kd))")
end

function Base.show(io::IO, ::MIME"text/plain", π::PIDControllerPolicy)
    println(io, "PIDControllerPolicy:")
    println(io, "  Kp: $(π.Kp)")
    println(io, "  Ki: $(π.Ki)")
    return println(io, "  Kd: $(π.Kd)")
end

"""
    PIDPressurePolicy{E,T} <: AbstractRDEPolicy

Pressure-outputting PID for use with `DirectScalarPressureAction` and
`HeldActionPolicy`. Integral / derivative use `control_dt` on each sample, so
wrapping with `HeldActionPolicy(period=control_dt)` and `env.dt = lock_dt`
matches SU fair-clock (coarse control, fine detection).

Returns absolute `u_p` (same mapping as `PIDAction` via `action_to_control`).
"""
mutable struct PIDPressurePolicy{E, T <: AbstractFloat} <: AbstractRDEPolicy
    env::E
    Kp::T
    Ki::T
    Kd::T
    control_dt::T
    integral::T
    previous_error::T
end

function PIDPressurePolicy(
        env;
        Kp::Real,
        Ki::Real,
        Kd::Real,
        control_dt::Real,
    )
    T = Float32
    return PIDPressurePolicy{typeof(env), T}(
        env,
        T(Kp), T(Ki), T(Kd), T(control_dt),
        zero(T), zero(T),
    )
end

get_env(π::PIDPressurePolicy) = π.env

function reset_pid_pressure_policy!(π::PIDPressurePolicy{E, T}) where {E, T}
    π.integral = zero(T)
    π.previous_error = zero(T)
    return nothing
end

function RDE_Env._predict_action(π::PIDPressurePolicy{E, T}, ::AbstractVector) where {E, T}
    @assert π.env.action_strat isa DirectScalarPressureAction "PIDPressurePolicy requires DirectScalarPressureAction"
    target = T(RDE.SHOCK_PRESSURES[get_target_shock_count(π.env)])
    u_p_mean = T(mean_u_p(π.env.prob.injection))
    err = target - u_p_mean
    π.integral += err * π.control_dt
    deriv = (err - π.previous_error) / π.control_dt
    π.previous_error = err
    a = clamp(π.Kp * err + π.Ki * π.integral + π.Kd * deriv, -one(T), one(T))
    prev = current_u_p(π.env.prob.injection)
    prev_s = prev isa AbstractVector ? T(prev[1]) : T(prev)
    # Same piecewise map as PIDAction / action_to_control with momentum = 0
    u_target = ifelse(a < 0, prev_s * (a + one(T)), prev_s + (π.env.u_pmax - prev_s) * a)
    return clamp(u_target, zero(T), π.env.u_pmax)
end

function Base.show(io::IO, π::PIDPressurePolicy)
    return print(io, "PIDPressurePolicy(Kp=$(π.Kp), Ki=$(π.Ki), Kd=$(π.Kd), dt=$(π.control_dt))")
end

function Base.show(io::IO, ::MIME"text/plain", π::PIDPressurePolicy)
    println(io, "PIDPressurePolicy:")
    println(io, "  Kp: $(π.Kp)")
    println(io, "  Ki: $(π.Ki)")
    println(io, "  Kd: $(π.Kd)")
    println(io, "  control_dt: $(π.control_dt)")
    return println(io, "  integral: $(π.integral)")
end

get_env(π::AbstractRDEPolicy) = nothing
