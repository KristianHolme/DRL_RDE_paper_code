# ============================================================================
# Cache Types
# ============================================================================

struct RewardShiftBufferCache{T <: AbstractFloat} <: AbstractCache
    shift_buffer::Vector{T}
end

struct CompositeRewardCache{T <: Tuple} <: AbstractCache
    caches::T
end

function RDE_Env.reset_cache!(cache::CompositeRewardCache)
    for c in cache.caches
        RDE_Env.reset_cache!(c)
    end
    return nothing
end

struct WrappedRewardCache{IC <: AbstractCache, OC <: AbstractCache} <: AbstractCache
    inner_cache::IC  # wrapped reward's cache
    outer_cache::OC  # wrapper's own state
end

function RDE_Env.reset_cache!(cache::WrappedRewardCache)
    RDE_Env.reset_cache!(cache.inner_cache)
    RDE_Env.reset_cache!(cache.outer_cache)
    return nothing
end

mutable struct ExponentialAverageCache{T <: AbstractFloat} <: AbstractCache
    average::Union{Nothing, T, Vector{T}}
end

function RDE_Env.reset_cache!(cache::ExponentialAverageCache)
    cache.average = nothing
    return nothing
end

mutable struct TransitionBasedCache{T <: AbstractFloat} <: AbstractCache
    past_rewards::Vector{T}
    past_shock_counts::Vector{Int}
    transition_found::Bool
end

function RDE_Env.reset_cache!(cache::TransitionBasedCache)
    empty!(cache.past_rewards)
    empty!(cache.past_shock_counts)
    cache.transition_found = false
    return nothing
end

abstract type CachedCompositeReward <: AbstractScalarRewardStrategy end
abstract type MultiAgentCachedCompositeReward <: AbstractVectorRewardStrategy end

const CompositeRewardStrategy = Union{
    CachedCompositeReward,
    MultiAgentCachedCompositeReward,
}

# ============================================================================
# Helper Functions
# ============================================================================

function action_magnitude_factor(lowest_action_magnitude_reward::AbstractFloat, action_magnitudes)
    α = lowest_action_magnitude_reward
    action_magnitude_inv = 1 .- action_magnitudes
    return α .+ (1 - α) .* action_magnitude_inv
end

function span_variation_reward(values::Vector{T}, α::T; threshold::T = T(4.0f-3)) where {T <: AbstractFloat}
    v_min, v_max = RDE.turbo_extrema(values)
    span = v_max - v_min
    span = max(span - threshold, zero(T))
    variation = span / mean(abs.(values))
    return exp.(-α * variation)
end

"""
    compute_section_control_efforts(env::RDEEnv{T}, n_sections::Int) where {T}

Compute per-section control efforts based on Δu_p normalized by u_pmax.
Returns a vector of length `n_sections` where each element is the maximum absolute
difference in control pressure within that section.
"""
function compute_section_control_efforts(env::RDEEnv{T}, n_sections::Int) where {T}
    N = env.prob.params.N
    cur = current_u_p(env.prob.injection)
    prev = previous_u_p(env.prob.injection)
    points_per_section = N ÷ n_sections
    efforts = zeros(T, n_sections)
    if cur isa AbstractVector
        for i in 1:n_sections
            s = (i - 1) * points_per_section + 1
            e = i * points_per_section
            efforts[i] = RDE.turbo_maximum_abs_diff(@view(cur[s:e]), @view(prev[s:e])) / env.u_pmax
        end
    else
        fill!(efforts, abs(cur - prev) / env.u_pmax)
    end
    return efforts
end

function calculate_periodicity_reward(u::AbstractVector{T}, N::Int, target_shock_count::Int, cache::AbstractVector{T}) where {T}
    inv_sqrt_N = 1 / sqrt(N)
    if target_shock_count > 1
        errs = zeros(T, target_shock_count - 1)
        shift_steps = N ÷ target_shock_count
        for i in 1:(target_shock_count - 1)
            # cache .= u #maybe not necessary?
            circshift!(cache, u, -shift_steps * i)
            errs[i] = RDE.turbo_diff_norm(u, cache) * inv_sqrt_N
        end
        maxerr = RDE.turbo_maximum(errs)
        periodicity_reward = one(T) - (max(maxerr - T(0.08), zero(T)) / sqrt(T(3)))
        periodicity_reward = sigmoid_to_linear(periodicity_reward)
        return periodicity_reward
    end
    return one(T)
end

function calculate_shock_reward(shocks::T, target_shock_count::Int, max_shocks::Int) where {T <: AbstractFloat}
    safe_max_shocks = T(max_shocks) + T(1.0e-6)
    shock_reward = max(min(shocks / target_shock_count, (shocks - safe_max_shocks) / (target_shock_count - safe_max_shocks)), zero(T))
    return sigmoid_to_linear(shock_reward)
end

#TODO: dont use shock_indices(performance), use shock_locations instead
function calculate_shock_rewards(u::AbstractVector{T}, dx::T, L::T, N::Int, target_shock_count::Int) where {T <: AbstractFloat}

    shock_inds = RDE.shock_indices(u, dx)
    shocks = T(length(shock_inds))
    @logmsg LogLevel(-10000) "shocks: $shocks"
    max_shocks = 4
    shock_reward = calculate_shock_reward(shocks, target_shock_count, max_shocks)

    if shocks > 1
        optimal_spacing = L / target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = one(T) - RDE.turbo_maximum_abs((shock_spacing .- optimal_spacing) ./ optimal_spacing)
    elseif shocks == 1 && target_shock_count == 1
        shock_spacing_reward = one(T)
    else
        shock_spacing_reward = zero(T)
    end
    shock_spacing_reward = sigmoid_to_linear(shock_spacing_reward)
    return shock_reward, shock_spacing_reward, shocks
end

function calculate_span_rewards(u::AbstractVector{T}, shocks::T) where {T <: AbstractFloat}
    mn, mx = RDE.turbo_extrema(u)
    span = mx - mn
    abs_span_punishment_threshold = T(0.08)
    target_span = T(2) - T(0.3) * shocks
    span_reward = span / target_span
    # span_reward = linear_to_sigmoid(span_reward)
    low_span_punishment = RDE.smooth_g(span / abs_span_punishment_threshold)
    return span_reward, low_span_punishment
end

function global_rewards(
        u::AbstractVector{T}, L::T, dx::T, rew_strat::CompositeRewardStrategy,
        cache::AbstractVector{T}, target_shock_count::Int
    ) where {T <: AbstractFloat}
    N = length(u)

    periodicity_reward = calculate_periodicity_reward(u, N, target_shock_count, cache)
    shock_reward, shock_spacing_reward, shocks = calculate_shock_rewards(u, dx, L, N, target_shock_count)
    span_reward, low_span_punishment = calculate_span_rewards(u, shocks)

    @logmsg LogLevel(-10000) "low_span_punishment: $low_span_punishment"
    @logmsg LogLevel(-10000) "span_reward: $span_reward"
    @logmsg LogLevel(-10000) "periodicity_reward: $periodicity_reward"
    @logmsg LogLevel(-10000) "shock_reward: $shock_reward"
    @logmsg LogLevel(-10000) "shock_spacing_reward: $shock_spacing_reward"
    return span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment
end

function global_reward(
        u::AbstractVector{T}, L::T, dx::T, rew_strat::CompositeRewardStrategy,
        cache::AbstractVector{T}, target_shock_count::Int
    ) where {T <: AbstractFloat}
    span_reward, periodicity_reward, shock_reward, shock_spacing_reward, low_span_punishment = global_rewards(u, L, dx, rew_strat, cache, target_shock_count)
    weighted_rewards = [span_reward, periodicity_reward, shock_reward, shock_spacing_reward]' * rew_strat.weights / sum(rew_strat.weights)
    global_reward = low_span_punishment * sum(weighted_rewards)
    @logmsg LogLevel(-10000) "global_reward: $global_reward"
    return global_reward
end

function global_reward(
        env::RDEEnv{T}, rew_strat::CompositeRewardStrategy,
        cache::RewardShiftBufferCache{T}, context::TargetShockCountContext
    ) where {T}
    N = env.prob.params.N
    dx::T = RDE.get_dx(env.prob)
    L = env.prob.params.L
    u = env.state[1:N]
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(context)

    return global_reward(u, L, dx, rew_strat, shift_buffer, target_shock_count)
end

# ============================================================================
# Reward Types and Implementations
# ============================================================================

# ----------------------------------------------------------------------------
# ShockSpanReward
# ----------------------------------------------------------------------------

@kwdef struct ShockSpanReward <: AbstractScalarRewardStrategy
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::ShockSpanReward,
        ::NoCache, context::TargetShockCountContext
    ) where {T}
    target_shock_count = get_target_shock_count(context)
    max_span = rew_strat.span_scale
    λ = rew_strat.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.params.L / env.prob.params.N
    shocks = T(RDE.count_shocks(u, dx))
    u_min, u_max = RDE.turbo_extrema(u)
    span = u_max - u_min
    span_reward = span / max_span
    if shocks >= target_shock_count
        shock_reward = one(T)
    elseif shocks > 0
        shock_reward = shocks / (2 * target_shock_count)
    else
        shock_reward = T(-1)
    end

    return λ * shock_reward + (1 - λ) * span_reward
end
# ----------------------------------------------------------------------------
# ShockPreservingReward
# ----------------------------------------------------------------------------

@kwdef mutable struct ShockPreservingReward <: AbstractScalarRewardStrategy
    span_scale::Float32 = 4.0f0
    shock_weight::Float32 = 0.8f0
    abscence_limit::Float32 = 5.0f0
    abscence_start::Union{Float32, Nothing} = nothing
end

function RDE_Env.compute_reward(env::RDEEnv{T}, rew_strat::ShockPreservingReward, ::NoCache, context::TargetShockCountContext) where {T}
    target_shock_count = get_target_shock_count(context)
    max_span = rew_strat.span_scale
    λ = rew_strat.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx::T = RDE.get_dx(env.prob)
    N = env.prob.params.N
    L = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)

    u_min, u_max = RDE.turbo_extrema(u)
    span = u_max - u_min
    span_reward = span / max_span

    if length(shock_inds) != target_shock_count
        if isnothing(rew_strat.abscence_start)
            rew_strat.abscence_start = env.t
        elseif env.t - rew_strat.abscence_start > rew_strat.abscence_limit
            env.terminated = true
            return T(-2.0)
        end
        shock_reward = T(-1.0)
    else
        optimal_spacing = L / target_shock_count
        shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_reward = T(1.0) - mean(abs.((shock_spacing .- optimal_spacing) ./ optimal_spacing))
    end
    return λ * shock_reward + (1 - λ) * span_reward
end

# ----------------------------------------------------------------------------
# ShockPreservingSymmetryReward
# ----------------------------------------------------------------------------

mutable struct ShockPreservingSymmetryReward <: AbstractScalarRewardStrategy end

RDE_Env.initialize_cache(::ShockPreservingSymmetryReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function RDE_Env.compute_reward(env::RDEEnv{T}, rew_strat::ShockPreservingSymmetryReward, cache::RewardShiftBufferCache{T}, context::TargetShockCountContext) where {T}
    target_shock_count = get_target_shock_count(context)
    N = env.prob.params.N
    u = env.state[1:N]

    errs = zeros(T, target_shock_count - 1)
    shift_buffer = cache.shift_buffer
    shift_steps = N ÷ target_shock_count
    for i in 1:(target_shock_count - 1)
        shift_buffer .= u
        circshift!(shift_buffer, u, -shift_steps * i)
        errs[i] = RDE.turbo_diff_norm(u, shift_buffer) / sqrt(T(N))
    end
    maxerr = RDE.turbo_maximum(errs)
    return T(1) - (maxerr - T(0.1)) / T(0.5)
end

# ----------------------------------------------------------------------------
# PeriodicityReward
# ----------------------------------------------------------------------------

struct PeriodicityReward <: AbstractScalarRewardStrategy end

RDE_Env.initialize_cache(::PeriodicityReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function RDE_Env.compute_reward(env::RDEEnv{T}, ::PeriodicityReward, cache::RewardShiftBufferCache{T}, ::AbstractCache) where {T <: AbstractFloat}
    u, = RDE.split_sol_view(env.state)
    N::Int = env.prob.params.N
    dx::T = RDE.get_dx(env.prob)
    L::T = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)
    shocks::Int = length(shock_inds)

    shift_buffer = cache.shift_buffer
    if shocks > 1
        shift_steps::Int = N ÷ shocks
        errs::Vector{T} = zeros(T, shocks - 1)
        for i in 1:(shocks - 1)
            shift_buffer .= u
            circshift!(shift_buffer, u, -shift_steps * i)
            errs[i]::T = RDE.turbo_diff_norm(u, shift_buffer) / sqrt(N)
        end
        maxerr::T = RDE.turbo_maximum(errs)
        periodicity_reward = T(1) - (max(maxerr - T(0.08), zero(T)) / sqrt(T(3)))
    else
        periodicity_reward = T(1)
    end

    if shocks > 1
        optimal_spacing::T = L / shocks
        shock_spacing::Vector{T} = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = T(1) - RDE.turbo_maximum_abs((shock_spacing .- optimal_spacing) ./ optimal_spacing)
    else
        shock_spacing_reward = T(1)
    end

    return (periodicity_reward::T + shock_spacing_reward::T) / T(2)
end


function Base.show(io::IO, ::PeriodicityReward)
    return print(io, "PeriodicityReward()")
end

function Base.show(io::IO, ::MIME"text/plain", ::PeriodicityReward)
    println(io, "PeriodicityReward")
    return nothing
end

# ----------------------------------------------------------------------------
# StabilityReward
# ----------------------------------------------------------------------------
"""
A reward that measures the stability of the solution over time.
It consists of three parts:
An amplitude stability reward, a shock profile similarity and a shock spacing reward.
The latter two are combined into a spatial stability reward by multiplication.
The final reward is the average of the amplitude stability and spatial stability.

When the target shock count is not achieved, the reward could be as high as ~0.5,
because of the amplitude stability reward. With the correct number of shocks,
the reward can approach 1.
"""
@kwdef struct StabilityReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    variation_scaling::T = 4.0f0
end

RDE_Env.initialize_cache(::StabilityReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

#= this was previously 5. This does not punish runs where we start perfect, then action
 leads to temporary instability, then settles in a good state the last 5 TUs=#
const STABILITY_WINDOW_DURATION = 100

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::StabilityReward,
        cache::RewardShiftBufferCache{T}, context::TargetShockCountContext
    ) where {T <: AbstractFloat}
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Extract all states from solution
    t_current = env.t
    sol_times = env.prob.sol.t::Vector{T}
    #TODO: maybe move this to reward_strategy struct?
    stability_window_duration = T(STABILITY_WINDOW_DURATION) #only view states within the last stability_window_duration seconds (for long dt's to not punish if we start bad and end good)
    lookback_index::Int = findfirst(sol_times .>= (t_current - stability_window_duration))
    # only use states after
    sol_states = env.prob.sol.u[lookback_index:end]::Vector{Vector{T}}
    n_states::Int = length(sol_states)

    # Get constants
    N = env.prob.params.N
    dx = RDE.get_dx(env.prob)
    L = env.prob.params.L
    target_shock_count = get_target_shock_count(context)
    shift_buffer = cache.shift_buffer

    # Collect data across time
    min_values = Vector{T}(undef, n_states)
    max_values = Vector{T}(undef, n_states)
    periodicity_rewards = Vector{T}(undef, n_states)
    shock_spacing_rewards = Vector{T}(undef, n_states)
    for (i, state) in enumerate(sol_states)
        u = @view state[1:N]

        # Collect span data
        u_min, u_max = RDE.turbo_extrema(u)
        min_values[i] = u_min
        max_values[i] = u_max


        # Calculate periodicity using target shock count
        if target_shock_count > 1
            shift_steps = N ÷ target_shock_count
            errs = zeros(T, target_shock_count - 1)
            for j in 1:(target_shock_count - 1)
                shift_buffer .= u
                circshift!(shift_buffer, u, -shift_steps * j)
                err = RDE.turbo_diff_norm(u, shift_buffer) / sqrt(N)
                err = max(err - T(0.037), zero(T)) #since N ÷ 3 is not a divisor of N, we set the baseline a bit lower
                errs[j] = err
            end
            maxerr = RDE.turbo_maximum(errs)
            periodicity_rewards[i] = exp(-rew_strat.variation_scaling * maxerr)
        else
            periodicity_rewards[i] = T(1)
        end

        # Calculate shock spacing using target shock count
        u, _ = RDE.split_sol_view(state)
        shock_inds = RDE.shock_indices(u, dx)
        shocks = length(shock_inds)
        if shocks > 0
            optimal_spacing = L / target_shock_count
            if shocks == 1
                shock_spacing = [L]
            else
                shock_spacing = mod.(RDE.periodic_diff(shock_inds), N) .* dx
            end
            # Use exponential reward function for spacing
            spacing_diffs = abs.(shock_spacing .- optimal_spacing)
            spacing_diffs = max.(spacing_diffs .- T(7.8f-3), zero(T)) #since N ÷ 3 is not a divisor of N, we set the baseline a bit lower
            spacing_errors = spacing_diffs ./ optimal_spacing
            spacing_rewards = exp.(-rew_strat.variation_scaling .* spacing_errors)
            shock_spacing_rewards[i] = RDE.turbo_minimum(spacing_rewards)
        else
            shock_spacing_rewards[i] = T(0)
        end
    end

    min_variation_rew = span_variation_reward(min_values, rew_strat.variation_scaling * 2)
    max_variation_rew = span_variation_reward(max_values, rew_strat.variation_scaling * 2)
    span_variation_rew = min(min_variation_rew, max_variation_rew)

    # Punish no span, to discourage getting constant solutions and 0.25 total reward.
    spans = max_values - min_values
    max_span = maximum(spans)
    small_span_threshold = T(0.1)
    small_span_punishment = min(T(1), inv(small_span_threshold) * max_span)

    # Calculate worst periodicity and shock spacing over time
    worst_periodicity = minimum(periodicity_rewards)
    worst_shock_spacing = minimum(shock_spacing_rewards)

    @debug "span_variation_rew: $span_variation_rew"
    @debug "small_span_punishment: $small_span_punishment"

    amplitude_stability = span_variation_rew * small_span_punishment
    spatial_stability = worst_periodicity * worst_shock_spacing
    stability = (amplitude_stability + spatial_stability) / T(2)
    @debug "amplitude_stability: $amplitude_stability"
    @debug "spatial_stability: $spatial_stability"
    @debug "stability: $stability"
    return stability
end

# ----------------------------------------------------------------------------
# StabilityTargetReward
# ----------------------------------------------------------------------------
struct StabilityTargetReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    stability_weight::T
    target_weight::T
    stability_reward::StabilityReward{T}
end

function StabilityTargetReward(;
        stability_weight::T = 0.5f0,
        target_weight::T = 0.5f0,
        kwargs...
    ) where {T <: AbstractFloat}
    return StabilityTargetReward{T}(
        stability_weight,
        target_weight,
        StabilityReward{T}(; kwargs...)
    )
end

function RDE_Env.initialize_cache(rew_strat::StabilityTargetReward, N::Int, ::Type{T}) where {T}
    return WrappedRewardCache(
        RDE_Env.initialize_cache(rew_strat.stability_reward, N, T),
        NoCache()
    )
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::StabilityTargetReward,
        cache::WrappedRewardCache, context::TargetShockCountContext
    ) where {T <: AbstractFloat}
    # Compute stability reward using inner cache
    stability_value = RDE_Env.compute_reward(env, rew_strat.stability_reward, cache.inner_cache, context)

    # Count current shocks
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    dx::T = RDE.get_dx(env.prob)
    current_shock_count = RDE.count_shocks(current_u, dx)

    # Get target from environment's goal cache
    target_shock_count = get_target_shock_count(context)

    # Compute target reward (1.0 if match, 0.0 otherwise)
    target_value = (current_shock_count == target_shock_count) ? one(T) : zero(T)

    # Weighted combination
    return rew_strat.stability_weight * stability_value + rew_strat.target_weight * target_value
end

function Base.show(io::IO, rew_strat::StabilityTargetReward)
    return print(io, "StabilityTargetReward(stability_weight=$(rew_strat.stability_weight), target_weight=$(rew_strat.target_weight))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::StabilityTargetReward)
    println(io, "StabilityTargetReward:")
    println(io, "  stability_weight: $(rew_strat.stability_weight)")
    println(io, "  target_weight: $(rew_strat.target_weight)")
    return println(io, "  stability_reward: $(rew_strat.stability_reward)")
end

# ----------------------------------------------------------------------------
# TargetGatedPressureTVReward
# ----------------------------------------------------------------------------

"""
    PressureStepTVCache

Previous-step mean `u_p` for [`TargetGatedPressureTVReward`](@ref) when the
injection profile has no knot schedule (e.g. linear / constant `u_p`).
"""
mutable struct PressureStepTVCache{T <: AbstractFloat} <: AbstractCache
    prev_up::T
end

PressureStepTVCache{T}() where {T <: AbstractFloat} = PressureStepTVCache{T}(T(NaN))

function RDE_Env.reset_cache!(cache::PressureStepTVCache{T}) where {T <: AbstractFloat}
    cache.prev_up = T(NaN)
    return nothing
end

"""
    TargetGatedPressureTVReward

Penalty for injection-pressure chatter, applied only when the current shock
count equals the target. Returns `1` during transients.

When on target, returns `1 - scale * tv / u_pmax`, clamped to `[0, 1]`:
- multistep: total variation of adjacent committed knots
- otherwise: `|u_p(t) - u_p(t-dt)|` via [`PressureStepTVCache`](@ref)

This is a smoothness-on-target prior, not a prescribed schedule.
"""
@kwdef struct TargetGatedPressureTVReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    scale::T = 1.0f0
end

function committed_knot_pressures(injection)
    inner = injection isa RDE.MovingFrameInjectionProfile ? injection.inner : injection
    if inner isa RDE.UniformMultiStepPressureProfile
        return inner.u_p_values
    end
    return nothing
end

function mean_adjacent_pressure_delta(knots::AbstractVector{T}) where {T <: AbstractFloat}
    n = length(knots)
    n < 2 && return zero(T)
    s = zero(T)
    for i in 2:n
        s += abs(knots[i] - knots[i - 1])
    end
    return s / T(n - 1)
end

RDE_Env.initialize_cache(::TargetGatedPressureTVReward, ::Int, ::Type{T}) where {T} =
    PressureStepTVCache{T}()

function on_target_tv_fraction(
        env::RDEEnv{T},
        cache::Union{NoCache, PressureStepTVCache{T}},
    ) where {T <: AbstractFloat}
    knots = committed_knot_pressures(env.prob.injection)
    if knots !== nothing
        frac = mean_adjacent_pressure_delta(knots) / env.u_pmax
        if cache isa PressureStepTVCache
            cache.prev_up = mean_u_p(env.prob.injection)
        end
        return frac
    end
    if cache isa NoCache
        return zero(T)
    end
    up = mean_u_p(env.prob.injection)
    if isnan(cache.prev_up)
        cache.prev_up = up
        return zero(T)
    end
    frac = abs(up - cache.prev_up) / env.u_pmax
    cache.prev_up = up
    return frac
end

function Base.show(io::IO, rew_strat::TargetGatedPressureTVReward)
    return print(io, "TargetGatedPressureTVReward(scale=$(rew_strat.scale))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::TargetGatedPressureTVReward)
    println(io, "TargetGatedPressureTVReward:")
    return println(io, "  scale: $(rew_strat.scale)")
end

function RDE_Env.compute_reward(
        env::RDEEnv{T},
        rew_strat::TargetGatedPressureTVReward,
        cache::Union{NoCache, PressureStepTVCache{T}},
        context::TargetShockCountContext,
    ) where {T <: AbstractFloat}
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    dx::T = RDE.get_dx(env.prob)
    current_shock_count = RDE.count_shocks(current_u, dx)
    if current_shock_count != get_target_shock_count(context)
        if cache isa PressureStepTVCache
            cache.prev_up = mean_u_p(env.prob.injection)
        end
        return one(T)
    end
    frac = on_target_tv_fraction(env, cache)
    return one(T) - clamp(rew_strat.scale * frac, zero(T), one(T))
end

# ----------------------------------------------------------------------------
# TargetMatchReward
# ----------------------------------------------------------------------------

"""
    TargetMatchReward()

`1` if the current shock count equals the target, else `0`. No spatial
stability and no `u_p` schedule. Use as the dense inner of
[`DenseLockReward`](@ref) so off-target transients are not scored for being
a pretty start-mode wave.
"""
struct TargetMatchReward <: AbstractScalarRewardStrategy end

function Base.show(io::IO, ::TargetMatchReward)
    return print(io, "TargetMatchReward()")
end

function Base.show(io::IO, ::MIME"text/plain", ::TargetMatchReward)
    return println(io, "TargetMatchReward")
end

function RDE_Env.compute_reward(
        env::RDEEnv{T},
        ::TargetMatchReward,
        ::AbstractCache,
        context::TargetShockCountContext,
    ) where {T <: AbstractFloat}
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    dx::T = RDE.get_dx(env.prob)
    current_shock_count = RDE.count_shocks(current_u, dx)
    return (current_shock_count == get_target_shock_count(context)) ? one(T) : zero(T)
end

# ----------------------------------------------------------------------------
# StepCostReward
# ----------------------------------------------------------------------------

"""
    StepCostReward(inner; cost = 0.5f0)

Subtract a constant `cost` from `inner` each step. This is a time preference
(prefer finishing), not a prescribed `u_p` schedule.
"""
struct StepCostReward{R <: AbstractScalarRewardStrategy, T <: AbstractFloat} <: AbstractScalarRewardStrategy
    inner::R
    cost::T
end

function StepCostReward(
        inner::AbstractScalarRewardStrategy;
        cost::T = 0.5f0,
    ) where {T <: AbstractFloat}
    return StepCostReward{typeof(inner), T}(inner, cost)
end

RDE_Env.initialize_cache(rew_strat::StepCostReward, N::Int, ::Type{T}) where {T} =
    RDE_Env.initialize_cache(rew_strat.inner, N, T)

function Base.show(io::IO, rew_strat::StepCostReward)
    return print(io, "StepCostReward(cost=$(rew_strat.cost), inner=$(rew_strat.inner))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::StepCostReward)
    println(io, "StepCostReward:")
    println(io, "  cost: $(rew_strat.cost)")
    return println(io, "  inner: $(rew_strat.inner)")
end

function RDE_Env.compute_reward(
        env::RDEEnv{T},
        rew_strat::StepCostReward,
        cache,
        context::TargetShockCountContext,
    ) where {T <: AbstractFloat}
    inner = RDE_Env.compute_reward(env, rew_strat.inner, cache, context)
    return inner - T(rew_strat.cost)
end

# ----------------------------------------------------------------------------
# MultiSectionReward
# ----------------------------------------------------------------------------

@kwdef mutable struct MultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int = 8
    lowest_action_magnitude_reward::T = 0.0f0 #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T} = Float32[1, 1, 5, 1]
end

RDE_Env.initialize_cache(::MultiSectionReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rew_strat::MultiSectionReward)
    return print(io, "MultiSectionReward(n_sections=$(rew_strat.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::MultiSectionReward)
    println(io, "MultiSectionReward:")
    println(io, "  n_sections: $(rew_strat.n_sections)")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::MultiSectionReward,
        cache::RewardShiftBufferCache{T}, context::TargetShockCountContext
    ) where {T}
    common_reward = global_reward(env, rew_strat, cache, context)
    efforts = compute_section_control_efforts(env, rew_strat.n_sections)
    individual_modifiers = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, efforts)
    return common_reward .* individual_modifiers
end


# ----------------------------------------------------------------------------
# CompositeReward
# ----------------------------------------------------------------------------

mutable struct CompositeReward{T <: AbstractFloat} <: CachedCompositeReward
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    span_reward::Bool
    weights::Vector{T}
    function CompositeReward{T}(;
            lowest_action_magnitude_reward::T = one(T),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
        ) where {T <: AbstractFloat}
        return new{T}(
            lowest_action_magnitude_reward,
            span_reward, weights
        )
    end
    function CompositeReward(;
            lowest_action_magnitude_reward::Float32 = 1.0f0,
            span_reward::Bool = true,
            weights::Vector{Float32} = [0.25f0, 0.25f0, 0.25f0, 0.25f0],
        )
        return CompositeReward{Float32}(
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            span_reward = span_reward,
            weights = weights
        )
    end
end

RDE_Env.initialize_cache(::CompositeReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, ::CompositeReward)
    return print(io, "CompositeReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::CompositeReward)
    println(io, "CompositeReward:")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    println(io, "  span_reward: $(rew_strat.span_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::CompositeReward,
        cache::RewardShiftBufferCache{T}, context::TargetShockCountContext
    ) where {T}
    reward = global_reward(env, rew_strat, cache, context)
    # Control-effort scalar modifier based on Δu_p
    Δmax = committed_pressure_delta(env.prob.injection) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, Δmax)
    reward *= action_effort_modifier
    @logmsg LogLevel(-10000) "set_reward!: $reward"
    return reward
end

# ----------------------------------------------------------------------------
# TimeAggCompositeReward
# ----------------------------------------------------------------------------

abstract type TimeAggregation end
struct TimeAvg <: TimeAggregation end
struct TimeMax <: TimeAggregation end
struct TimeMin <: TimeAggregation end
struct TimeSum <: TimeAggregation end
struct TimeProd <: TimeAggregation end

struct TimeAggCompositeReward{T <: AbstractFloat} <: CachedCompositeReward
    aggregation::TimeAggregation
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    span_reward::Bool
    weights::Vector{T}
    function TimeAggCompositeReward{T}(;
            aggregation::TimeAggregation = TimeMin(),
            lowest_action_magnitude_reward::T = one(T),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
        ) where {T <: AbstractFloat}
        return new{T}(
            aggregation,
            lowest_action_magnitude_reward,
            span_reward, weights
        )
    end
    function TimeAggCompositeReward(;
            aggregation::TimeAggregation = TimeMin(),
            lowest_action_magnitude_reward::T = T(1.0),
            span_reward::Bool = true,
            weights::Vector{T} = [T(0.25), T(0.25), T(0.25), T(0.25)],
        ) where {T <: AbstractFloat}
        return TimeAggCompositeReward{T}(
            aggregation = aggregation,
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            span_reward = span_reward,
            weights = weights
        )
    end
end

RDE_Env.initialize_cache(::TimeAggCompositeReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rew_strat::TimeAggCompositeReward)
    return print(io, "TimeAggCompositeReward(aggregation=$(rew_strat.aggregation))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::TimeAggCompositeReward)
    println(io, "TimeAggCompositeReward:")
    println(io, "  aggregation: $(rew_strat.aggregation)")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    println(io, "  span_reward: $(rew_strat.span_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function aggregate(rewards::AbstractVector, aggregation::TimeAvg)
    return mean(rewards)
end
function aggregate(rewards::AbstractVector, aggregation::TimeMax)
    return maximum(rewards)
end
function aggregate(rewards::AbstractVector, aggregation::TimeMin)
    return minimum(rewards)
end
function aggregate(rewards::AbstractVector, aggregation::TimeSum)
    return sum(rewards)
end
function aggregate(rewards::AbstractVector, aggregation::TimeProd)
    return prod(rewards)
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::TimeAggCompositeReward,
        cache::RewardShiftBufferCache{T}, context::TargetShockCountContext
    ) where {T}
    N = env.prob.params.N
    dx::T = RDE.get_dx(env.prob)
    L = env.prob.params.L
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(context)
    sol = env.prob.sol
    if !isnothing(sol)
        us = [uλ[1:N] for uλ in sol.u[2:end]]
        rewards = zeros(T, length(us))
    else
        @debug "No solution found for env at time $(env.t), step $(env.steps_taken)"
        us = [env.state[1:N]]
        rewards = zeros(T, 1)
    end
    for (i, u) in enumerate(us)
        rewards[i] = global_reward(u, L, dx, rew_strat, shift_buffer, target_shock_count)
    end
    agg_reward = aggregate(rewards, rew_strat.aggregation)

    Δmax = committed_pressure_delta(env.prob.injection) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, Δmax)
    return agg_reward * action_effort_modifier
end

# ----------------------------------------------------------------------------
# ConstantTargetReward
# ----------------------------------------------------------------------------

@kwdef struct ConstantTargetReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    target::T = 0.64f0
end

function Base.show(io::IO, rew_strat::ConstantTargetReward)
    return print(io, "ConstantTargetReward(target=$(rew_strat.target))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::ConstantTargetReward)
    println(io, "ConstantTargetReward:")
    return println(io, "  target: $(rew_strat.target)")
end

function RDE_Env.compute_reward(env::RDEEnv{T}, rew_strat::ConstantTargetReward{T}, cache, context::TargetShockCountContext) where {T}
    return -abs(rew_strat.target - mean_u_p(env.prob.injection)) + one(T)
end

# ----------------------------------------------------------------------------
# TimeAggMultiSectionReward
# ----------------------------------------------------------------------------

mutable struct TimeAggMultiSectionReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    aggregation::TimeAggregation
    n_sections::Int
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
end
function Base.show(io::IO, rew_strat::TimeAggMultiSectionReward)
    return print(io, "TimeAggMultiSectionReward(n_sections=$(rew_strat.n_sections), aggregation=$(typeof(rew_strat.aggregation)))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::TimeAggMultiSectionReward)
    println(io, "TimeAggMultiSectionReward:")
    println(io, "  aggregation: $(typeof(rew_strat.aggregation))")
    println(io, "  n_sections: $(rew_strat.n_sections)")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::TimeAggMultiSectionReward,
        cache::RewardShiftBufferCache{T}, context::TargetShockCountContext
    ) where {T}
    N = env.prob.params.N
    n_sections = rew_strat.n_sections
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(context)

    # Get all states from the solution
    if !isnothing(env.prob.sol)
        us = [uλ[1:N] for uλ in env.prob.sol.u[2:end]]
    else
        us = [env.state[1:N]]
    end
    section_rewards = zeros(T, n_sections, length(us))

    # Calculate rewards for each section and time step
    for (i, u) in enumerate(us)
        dx::T = RDE.get_dx(env.prob)
        common_reward = global_reward(u, env.prob.params.L, dx, rew_strat, shift_buffer, target_shock_count)
        efforts = compute_section_control_efforts(env, n_sections)
        individual_modifiers = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, efforts)
        section_rewards[:, i] = common_reward .* individual_modifiers
    end

    # Aggregate rewards over time for each section
    agg_section_rewards = zeros(T, n_sections)
    for i in 1:n_sections
        agg_section_rewards[i] = aggregate(section_rewards[i, :], rew_strat.aggregation)
    end

    return agg_section_rewards
end

# Ensure T is known when building defaults
function TimeAggMultiSectionReward{T}(;
        aggregation::TimeAggregation = TimeMin(),
        n_sections::Int = 4,
        lowest_action_magnitude_reward::T = zero(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)]
    ) where {T <: AbstractFloat}
    return TimeAggMultiSectionReward{T}(
        aggregation,
        n_sections,
        lowest_action_magnitude_reward,
        weights
    )
end

function TimeAggMultiSectionReward(;
        aggregation::TimeAggregation = TimeMin(),
        n_sections::Int = 4,
        lowest_action_magnitude_reward::Float32 = 0.0f0,
        weights::Vector{Float32} = [1.0f0, 1.0f0, 5.0f0, 1.0f0]
    )
    return TimeAggMultiSectionReward{Float32}(
        aggregation,
        n_sections,
        lowest_action_magnitude_reward,
        weights
    )
end

RDE_Env.initialize_cache(::TimeAggMultiSectionReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

# ----------------------------------------------------------------------------
# TimeDiffNormReward
# ----------------------------------------------------------------------------

@kwdef struct TimeDiffNormReward{T <: AbstractFloat} <: AbstractScalarRewardStrategy
    threshold::T = 1.1f0
    threshold_reward::T = 0.3f0
end

function RDE_Env.compute_reward(env::RDEEnv{T}, rew_strat::TimeDiffNormReward{T}, ::AbstractCache, ::AbstractCache) where {T}
    if isnothing(env.prob.sol)
        return zero(T)
    end
    # Spatial size for each snapshot
    N = env.prob.params.N
    # Extract only the velocity field u from each saved state [u; λ]
    us = map(v -> v[1:N], env.prob.sol.u)
    n = length(us)
    # We will compute pairwise spectral-distance between all time snapshots.
    # There are n*(n-1)/2 unique pairs, store their norms in a flat vector.
    diff_norms = zeros(T, Int((n^2 - n) // 2))
    ind = 1
    # FFT magnitude spectra for each snapshot; use real FFT for efficiency.
    ft_us = rfft.(us)
    abs_ft_us = map(v -> abs.(v), ft_us)
    for i in 1:n, j in (i + 1):n
        # L2 distance between magnitude spectra, normalized by sqrt(N)
        diff_norms[ind] = RDE.turbo_diff_norm(abs_ft_us[i], abs_ft_us[j]) / sqrt(N)
        ind += 1
    end
    # Use fast turbo maximum for the worst-case spectral change across time
    max_norm = RDE.turbo_maximum(diff_norms)
    # Map max_norm to (0,1] via exponential shaping so that
    #   reward = threshold_reward when max_norm == threshold
    a = log(rew_strat.threshold_reward) / rew_strat.threshold
    return exp(a * max_norm)
end

# ----------------------------------------------------------------------------
# PeriodMinimumReward
# ----------------------------------------------------------------------------

struct PeriodMinimumReward{T <: AbstractFloat} <: CachedCompositeReward
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
    function PeriodMinimumReward{T}(;
            lowest_action_magnitude_reward::T = zero(T),
            weights::Vector{T} = [one(T), one(T), T(5), one(T)],
        ) where {T <: AbstractFloat}
        return new{T}(
            lowest_action_magnitude_reward,
            weights
        )
    end
    function PeriodMinimumReward(;
            lowest_action_magnitude_reward::Float32 = 0.0f0,
            weights::Vector{Float32} = Float32[1, 1, 5, 1],
        )
        return PeriodMinimumReward{Float32}(
            lowest_action_magnitude_reward = lowest_action_magnitude_reward,
            weights = weights
        )
    end
end

RDE_Env.initialize_cache(::PeriodMinimumReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rew_strat::PeriodMinimumReward)
    return print(io, "PeriodMinimumReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::PeriodMinimumReward)
    println(io, "PeriodMinimumReward:")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function time_minimum_global_reward(env::RDEEnv{T}, rew_strat::CompositeRewardStrategy, cache::RewardShiftBufferCache{T}, context::TargetShockCountContext)::T where {T <: AbstractFloat}
    N::Int = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Pre-compute constants to avoid type instability in loop
    L::T = env.prob.params.L
    dx::T = RDE.get_dx(env.prob)
    weight_sum::T = sum(rew_strat.weights)
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(context)

    # Initialize with large values to find minimum
    min_rewards::Vector{T} = fill(T(10000), 5)

    # Type-stable iteration - convert to Vector{Vector{T}} if needed
    sol_states = env.prob.sol.u::Vector{Vector{T}}

    for state::Vector{T} in sol_states
        # Use view to avoid allocation, with fallback type assertion
        u = @view state[1:N]

        span_rew, periodicity_rew, shock_rew, shock_spacing_rew, low_span_punishment =
            global_rewards(u, L, dx, rew_strat, shift_buffer, target_shock_count)

        # Update minimums efficiently
        min_rewards[1] = min(min_rewards[1], span_rew)
        min_rewards[2] = min(min_rewards[2], periodicity_rew)
        min_rewards[3] = min(min_rewards[3], shock_rew)
        min_rewards[4] = min(min_rewards[4], shock_spacing_rew)
        min_rewards[5] = min(min_rewards[5], low_span_punishment)
    end

    # Efficient weighted sum without temporary arrays - manually unrolled
    weighted_rewards = (
        min_rewards[1] * rew_strat.weights[1] +
            min_rewards[2] * rew_strat.weights[2] +
            min_rewards[3] * rew_strat.weights[3] +
            min_rewards[4] * rew_strat.weights[4]
    ) / weight_sum
    return min_rewards[5] * weighted_rewards  # low_span_punishment * weighted_sum
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::PeriodMinimumReward,
        cache::RewardShiftBufferCache{T}, context::TargetShockCountContext
    ) where {T}
    global_reward = time_minimum_global_reward(env, rew_strat, cache, context)
    Δmax = committed_pressure_delta(env.prob.injection) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, Δmax)
    return global_reward * action_effort_modifier
end

struct PeriodMinimumVariationReward{T <: AbstractFloat} <: CachedCompositeReward
    lowest_action_magnitude_reward::T #reward will be \in [lowest_action_magnitude_reward, 1]
    weights::Vector{T}
    variation_penalties::Vector{T}  # α values for each component [span, periodicity, shock, shock_spacing]
end
function PeriodMinimumVariationReward{T}(;
        lowest_action_magnitude_reward::T = one(T),
        weights::Vector{T} = [one(T), one(T), T(5), one(T)],
        variation_penalties::Vector{T} = [T(2), T(2), T(2), T(2)],
    ) where {T <: AbstractFloat}
    return PeriodMinimumVariationReward{T}(
        lowest_action_magnitude_reward,
        weights,
        variation_penalties
    )
end
function PeriodMinimumVariationReward(;
        lowest_action_magnitude_reward::Float32 = 1.0f0,
        weights::Vector{Float32} = Float32[1, 1, 5, 1],
        variation_penalties::Vector{Float32} = Float32[2.0, 2.0, 2.0, 2.0],
    )
    return PeriodMinimumVariationReward{Float32}(
        lowest_action_magnitude_reward = lowest_action_magnitude_reward,
        weights = weights,
        variation_penalties = variation_penalties
    )
end

RDE_Env.initialize_cache(::PeriodMinimumVariationReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, ::PeriodMinimumVariationReward)
    return print(io, "PeriodMinimumVariationReward()")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::PeriodMinimumVariationReward)
    println(io, "PeriodMinimumVariationReward:")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    println(io, "  weights: $(rew_strat.weights)")
    return println(io, "  variation_penalties: $(rew_strat.variation_penalties)")
end

function calculate_variation_punishment(values::Vector{T}, α::T) where {T <: AbstractFloat}
    if length(values) <= 1
        return one(T)  # No variation with single value
    end

    mean_abs_value = mean(abs.(values))
    if mean_abs_value < T(1.0e-6)
        return one(T)  # Avoid division by zero
    end

    coefficient_of_variation = std(values) / mean_abs_value
    return exp(-α * coefficient_of_variation)
end

function time_minimum_global_reward_with_variation(env::RDEEnv{T}, rew_strat::PeriodMinimumVariationReward, cache::RewardShiftBufferCache{T}, context::TargetShockCountContext) where {T <: AbstractFloat}
    N = env.prob.params.N
    if isnothing(env.prob.sol)
        return zero(T)
    end

    # Pre-compute constants to avoid type instability in loop
    L = env.prob.params.L::T
    dx = RDE.get_dx(env.prob)::T
    weight_sum = sum(rew_strat.weights)::T
    shift_buffer = cache.shift_buffer
    target_shock_count = get_target_shock_count(context)

    # Type-stable iteration
    sol_states = env.prob.sol.u
    n_states = length(sol_states)

    # Pre-allocate reward vectors with correct size
    all_span_rewards = Vector{T}(undef, n_states)
    all_periodicity_rewards = Vector{T}(undef, n_states)
    all_shock_rewards = Vector{T}(undef, n_states)
    all_shock_spacing_rewards = Vector{T}(undef, n_states)
    all_low_span_punishments = Vector{T}(undef, n_states)

    for (i, state) in enumerate(sol_states)
        # Use view to avoid allocation
        u = @view state[1:N]

        span_rew, periodicity_rew, shock_rew, shock_spacing_rew, low_span_punishment =
            global_rewards(u, L, dx, rew_strat, shift_buffer, target_shock_count)

        all_span_rewards[i] = span_rew
        all_periodicity_rewards[i] = periodicity_rew
        all_shock_rewards[i] = shock_rew
        all_shock_spacing_rewards[i] = shock_spacing_rew
        all_low_span_punishments[i] = low_span_punishment
    end

    # Calculate variation punishment for each component
    span_var_punishment = calculate_variation_punishment(all_span_rewards, T(rew_strat.variation_penalties[1]))
    periodicity_var_punishment = calculate_variation_punishment(all_periodicity_rewards, T(rew_strat.variation_penalties[2]))
    shock_var_punishment = calculate_variation_punishment(all_shock_rewards, T(rew_strat.variation_penalties[3]))
    shock_spacing_var_punishment = calculate_variation_punishment(all_shock_spacing_rewards, T(rew_strat.variation_penalties[4]))

    variation_punishment = minimum([span_var_punishment, periodicity_var_punishment, shock_var_punishment, shock_spacing_var_punishment])
    # Apply variation punishment to each component before taking minimum
    min_span_rew = minimum(all_span_rewards)
    min_periodicity_rew = minimum(all_periodicity_rewards)
    min_shock_rew = minimum(all_shock_rewards)
    min_shock_spacing_rew = minimum(all_shock_spacing_rewards)

    # Low span punishment: only minimum, no variation punishment
    min_low_span_punishment = minimum(all_low_span_punishments)

    # Efficient weighted sum
    weighted_rewards = (
        min_span_rew * rew_strat.weights[1] +
            min_periodicity_rew * rew_strat.weights[2] +
            min_shock_rew * rew_strat.weights[3] +
            min_shock_spacing_rew * rew_strat.weights[4]
    ) / weight_sum

    return min_low_span_punishment * variation_punishment * weighted_rewards
end

function RDE_Env.compute_reward(env::RDEEnv{T}, rew_strat::PeriodMinimumVariationReward, cache::RewardShiftBufferCache{T}, context::TargetShockCountContext) where {T}
    global_reward = time_minimum_global_reward_with_variation(env, rew_strat, cache, context)
    Δmax = committed_pressure_delta(env.prob.injection) / env.u_pmax
    action_effort_modifier = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, Δmax)
    return global_reward * action_effort_modifier
end

@kwdef mutable struct MultiSectionPeriodMinimumReward{T <: AbstractFloat} <: MultiAgentCachedCompositeReward
    n_sections::Int = 8
    lowest_action_magnitude_reward::T = 0.0f0
    weights::Vector{T} = Float32[1, 1, 5, 1]
end

RDE_Env.initialize_cache(::MultiSectionPeriodMinimumReward, N::Int, ::Type{T}) where {T} = RewardShiftBufferCache{T}(zeros(T, N))

function Base.show(io::IO, rew_strat::MultiSectionPeriodMinimumReward)
    return print(io, "MultiSectionPeriodMinimumReward(n_sections=$(rew_strat.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::MultiSectionPeriodMinimumReward)
    println(io, "MultiSectionPeriodMinimumReward:")
    println(io, "  n_sections: $(rew_strat.n_sections)")
    println(io, "  lowest_action_magnitude_reward: $(rew_strat.lowest_action_magnitude_reward)")
    return println(io, "  weights: $(rew_strat.weights)")
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::MultiSectionPeriodMinimumReward,
        cache::RewardShiftBufferCache{T}, context::TargetShockCountContext
    ) where {T}
    # Compute the time minimum global reward (same as PeriodMinimumReward)
    common_reward = time_minimum_global_reward(env, rew_strat, cache, context)

    # Multi-section control-effort punishment
    efforts = compute_section_control_efforts(env, rew_strat.n_sections)
    individual_modifiers = action_magnitude_factor(rew_strat.lowest_action_magnitude_reward, efforts)

    return common_reward .* individual_modifiers
end

"""
    MultiplicativeReward(rewards::AbstractRewardStrategy...)

A reward wrapper that multiplies the rewards from multiple reward types.
If any wrapped reward is vector-valued, this wrapper is vector-valued.
"""
struct MultiplicativeScalarReward <: AbstractScalarRewardStrategy
    rewards::Vector{AbstractScalarRewardStrategy}
end

struct MultiplicativeVectorReward <: AbstractVectorRewardStrategy
    rewards::Vector{AbstractRewardStrategy}
end

function MultiplicativeReward(rewards::Vector{<:AbstractRewardStrategy})
    if isempty(rewards)
        throw(ArgumentError("MultiplicativeReward requires at least one reward"))
    end
    if any(
            !(r isa AbstractScalarRewardStrategy || r isa AbstractVectorRewardStrategy)
                for r in rewards
        )
        throw(ArgumentError("MultiplicativeReward rewards must subtype AbstractScalarRewardStrategy or AbstractVectorRewardStrategy"))
    end
    if any(r -> r isa AbstractVectorRewardStrategy, rewards)
        return MultiplicativeVectorReward(Vector{AbstractRewardStrategy}(rewards))
    end
    return MultiplicativeScalarReward(Vector{AbstractScalarRewardStrategy}(rewards))
end

function MultiplicativeReward(rewards::AbstractRewardStrategy...)
    return MultiplicativeReward(collect(rewards))
end

function Base.show(io::IO, rew_strat::MultiplicativeScalarReward)
    return print(io, "MultiplicativeReward($(length(rew_strat.rewards)) components, scalar)")
end

function Base.show(io::IO, rew_strat::MultiplicativeVectorReward)
    return print(io, "MultiplicativeReward($(length(rew_strat.rewards)) components, vector)")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::MultiplicativeScalarReward)
    println(io, "MultiplicativeReward (scalar):")
    println(io, "  components: $(length(rew_strat.rewards))")
    for (i, reward) in enumerate(rew_strat.rewards)
        println(io, "  reward $i: $(typeof(reward))")
    end
    return
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::MultiplicativeVectorReward)
    println(io, "MultiplicativeReward (vector):")
    println(io, "  components: $(length(rew_strat.rewards))")
    for (i, reward) in enumerate(rew_strat.rewards)
        println(io, "  reward $i: $(typeof(reward))")
    end
    return
end

RDE_Env.initialize_cache(rew_strat::MultiplicativeScalarReward, N::Int, ::Type{T}) where {T} =
    CompositeRewardCache(Tuple(RDE_Env.initialize_cache(r, N, T) for r in rew_strat.rewards))

RDE_Env.initialize_cache(rew_strat::MultiplicativeVectorReward, N::Int, ::Type{T}) where {T} =
    CompositeRewardCache(Tuple(RDE_Env.initialize_cache(r, N, T) for r in rew_strat.rewards))

function compute_multiplicative_reward(env, rewards, caches, context)
    reward = RDE_Env.compute_reward(env, rewards[1], caches[1], context)
    for (i, r) in enumerate(rewards[2:end])
        next_reward = RDE_Env.compute_reward(env, r, caches[i + 1], context)
        reward = next_reward .* reward
    end
    return reward
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::MultiplicativeScalarReward,
        cache::CompositeRewardCache, context::TargetShockCountContext
    ) where {T}
    return compute_multiplicative_reward(env, rew_strat.rewards, cache.caches, context)
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::MultiplicativeVectorReward,
        cache::CompositeRewardCache, context::TargetShockCountContext
    ) where {T}
    return compute_multiplicative_reward(env, rew_strat.rewards, cache.caches, context)
end

"""
    ExponentialAverageReward(wrapped_reward::AbstractRewardStrategy; α = 0.2)

A reward wrapper that applies exponential averaging to smooth reward signals.
The wrapped reward must be either scalar or vector valued; the wrapper preserves that shape.
"""
struct ExponentialAverageScalarReward{R <: AbstractScalarRewardStrategy, T <: AbstractFloat} <: AbstractScalarRewardStrategy
    wrapped_reward::R
    α::T  # averaging parameter
end

struct ExponentialAverageVectorReward{R <: AbstractVectorRewardStrategy, T <: AbstractFloat} <: AbstractVectorRewardStrategy
    wrapped_reward::R
    α::T  # averaging parameter
end

function ExponentialAverageReward(wrapped_reward::R; α::T = 0.2f0) where {R <: AbstractScalarRewardStrategy, T <: AbstractFloat}
    return ExponentialAverageScalarReward{R, T}(wrapped_reward, α)
end

function ExponentialAverageReward(wrapped_reward::R; α::T = 0.2f0) where {R <: AbstractVectorRewardStrategy, T <: AbstractFloat}
    return ExponentialAverageVectorReward{R, T}(wrapped_reward, α)
end

function Base.show(io::IO, rew_strat::ExponentialAverageScalarReward)
    return print(io, "ExponentialAverageReward(α=$(rew_strat.α), wrapped=$(typeof(rew_strat.wrapped_reward)), scalar)")
end

function Base.show(io::IO, rew_strat::ExponentialAverageVectorReward)
    return print(io, "ExponentialAverageReward(α=$(rew_strat.α), wrapped=$(typeof(rew_strat.wrapped_reward)), vector)")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::ExponentialAverageScalarReward)
    println(io, "ExponentialAverageReward (scalar):")
    println(io, "  α: $(rew_strat.α)")
    return println(io, "  wrapped_reward: $(typeof(rew_strat.wrapped_reward))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::ExponentialAverageVectorReward)
    println(io, "ExponentialAverageReward (vector):")
    println(io, "  α: $(rew_strat.α)")
    return println(io, "  wrapped_reward: $(typeof(rew_strat.wrapped_reward))")
end

RDE_Env.initialize_cache(rew_strat::ExponentialAverageScalarReward, N::Int, ::Type{T}) where {T} =
    WrappedRewardCache(
    RDE_Env.initialize_cache(rew_strat.wrapped_reward, N, T),
    ExponentialAverageCache{T}(nothing)
)

RDE_Env.initialize_cache(rew_strat::ExponentialAverageVectorReward, N::Int, ::Type{T}) where {T} =
    WrappedRewardCache(
    RDE_Env.initialize_cache(rew_strat.wrapped_reward, N, T),
    ExponentialAverageCache{T}(nothing)
)

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::ExponentialAverageScalarReward,
        cache::WrappedRewardCache, context::TargetShockCountContext
    ) where {T}
    current_reward = RDE_Env.compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache, context)
    if isnothing(cache.outer_cache.average)
        cache.outer_cache.average = current_reward
        return current_reward
    end
    cache.outer_cache.average = rew_strat.α * current_reward + (one(rew_strat.α) - rew_strat.α) * cache.outer_cache.average
    return cache.outer_cache.average
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::ExponentialAverageVectorReward,
        cache::WrappedRewardCache, context::TargetShockCountContext
    ) where {T}
    current_reward = RDE_Env.compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache, context)
    if isnothing(cache.outer_cache.average)
        cache.outer_cache.average = current_reward
        return current_reward
    end
    cache.outer_cache.average = rew_strat.α .* current_reward .+ (one(rew_strat.α) - rew_strat.α) .* cache.outer_cache.average
    return cache.outer_cache.average
end

"""
    DtWeightedReward{R} <: AbstractScalarRewardStrategy

Multiply the wrapped scalar reward by `env.dt` so returns integrate over physical time.
Do not wrap `TransitionBasedReward` (it already returns `-env.dt` until lock).
"""
struct DtWeightedReward{R <: AbstractScalarRewardStrategy} <: AbstractScalarRewardStrategy
    wrapped_reward::R
end

function Base.show(io::IO, rew_strat::DtWeightedReward)
    return print(io, "DtWeightedReward(wrapped=$(typeof(rew_strat.wrapped_reward)))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::DtWeightedReward)
    println(io, "DtWeightedReward:")
    return println(io, "  wrapped_reward: $(typeof(rew_strat.wrapped_reward))")
end

function RDE_Env.initialize_cache(rew_strat::DtWeightedReward, N::Int, ::Type{T}) where {T}
    return WrappedRewardCache(
        RDE_Env.initialize_cache(rew_strat.wrapped_reward, N, T),
        NoCache()
    )
end

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::DtWeightedReward,
        cache::WrappedRewardCache, context::TargetShockCountContext
    ) where {T}
    inner = RDE_Env.compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache, context)
    return inner * env.dt
end

"""
    TransitionBasedReward{R<:AbstractScalarRewardStrategy} <: AbstractScalarRewardStrategy

A reward wrapper that gives -1*dt reward until a successful transition is detected,
then terminates the environment. Uses transition detection logic similar to detect_transition.

# Fields
- `wrapped_reward::T`: The underlying reward to wrap and monitor
- `reward_stability_length::Float32`: Time duration (in env time units) to maintain stable rewards
- `reward_threshold::Float32`: Minimum reward threshold for stability check

# Notes
- Returns -1.0*dt until transition is found
- When transition is detected, sets env.terminated = true
- Transition occurs when: shock count reaches target AND rewards stay above threshold for stability_length time
- Internal state (past_rewards, past_shock_counts, transition_found) is stored in the cache
"""
@kwdef struct TransitionBasedReward{R <: AbstractScalarRewardStrategy, T <: AbstractFloat} <: AbstractScalarRewardStrategy
    wrapped_reward::R = StabilityReward(1.0f0)
    reward_stability_length::T = 20.0f0 # in time units
    reward_threshold::T = 0.9f0
end

function TransitionBasedReward(
        wrapped_reward::AbstractScalarRewardStrategy;
        kwargs...
    )
    return TransitionBasedReward(; wrapped_reward, kwargs...)
end

function Base.show(io::IO, rew_strat::TransitionBasedReward)
    return print(io, "TransitionBasedReward(wrapped=$(typeof(rew_strat.wrapped_reward)))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::TransitionBasedReward)
    println(io, "TransitionBasedReward:")
    println(io, "  reward_stability_length: $(rew_strat.reward_stability_length)")
    println(io, "  reward_threshold: $(rew_strat.reward_threshold)")
    return println(io, "  wrapped_reward: $(typeof(rew_strat.wrapped_reward))")
end

RDE_Env.initialize_cache(rew_strat::TransitionBasedReward, N::Int, ::Type{T}) where {T} =
    WrappedRewardCache(
    RDE_Env.initialize_cache(rew_strat.wrapped_reward, N, T),
    TransitionBasedCache{T}(T[], Int[], false)
)

function RDE_Env.compute_reward(
        env::RDEEnv{T}, rew_strat::TransitionBasedReward{S, T},
        cache::WrappedRewardCache, context::TargetShockCountContext
    ) where {T, S}
    # Compute the wrapped reward using inner cache
    wrapped_reward_value = RDE_Env.compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache, context)

    # Handle vector rewards by taking minimum (similar to detect_transition)
    reward_scalar = if wrapped_reward_value isa AbstractVector
        minimum(wrapped_reward_value)
    else
        wrapped_reward_value
    end

    # Count current shocks
    N = env.prob.params.N
    u = @view env.state[1:N]
    dx::T = RDE.get_dx(env.prob)
    current_shock_count = RDE.count_shocks(u, dx)
    target_shock_count = get_target_shock_count(context)

    # Update history in outer cache
    push!(cache.outer_cache.past_rewards, T(reward_scalar))
    push!(cache.outer_cache.past_shock_counts, current_shock_count)

    # Check for transition (only if we have enough history)
    if length(cache.outer_cache.past_shock_counts) >= 2
        cache.outer_cache.transition_found = detect_transition_realtime(rew_strat, cache.outer_cache, env.dt, target_shock_count)

        if cache.outer_cache.transition_found
            env.terminated = true
            @logmsg LogLevel(-500) "Transition detected! Terminating environment at t=$(env.t)"
            env.info["Termination.Reason"] = "Transition detected"
            env.info["Termination.env_t"] = env.t
            return T(100.0)  # Positive reward when transition found
        end
    end

    # Return -1*dt until transition
    return T(-env.dt)
end

function detect_transition_realtime(rew_strat::TransitionBasedReward{T, U}, cache::TransitionBasedCache{U}, dt::U, target_shock_count::Int) where {T, U <: AbstractFloat}
    stability_steps = round(Int, rew_strat.reward_stability_length / dt)
    if length(cache.past_shock_counts) < stability_steps
        @debug "Not enough shock counts to detect transition"
        return false
    end


    # Check if we've had stable rewards for long enough since then
    stability_rewards = cache.past_rewards[(end - stability_steps + 1):end]
    stability_shock_counts = cache.past_shock_counts[(end - stability_steps + 1):end]
    if all(stability_shock_counts .== target_shock_count) && minimum(stability_rewards) > rew_strat.reward_threshold
        return true
    end
    return false
end

# ----------------------------------------------------------------------------
# DenseLockReward (after TransitionBasedReward — needs those types)
# ----------------------------------------------------------------------------

"""
    DenseLockCache

Inner train-reward cache plus the eval lock detector
(`StabilityReward` history in a `TransitionBasedCache`).
"""
struct DenseLockCache{IC <: AbstractCache, LC <: AbstractCache, T <: AbstractFloat} <: AbstractCache
    inner::IC
    lock_inner::LC
    lock_hist::TransitionBasedCache{T}
end

function RDE_Env.reset_cache!(cache::DenseLockCache)
    RDE_Env.reset_cache!(cache.inner)
    RDE_Env.reset_cache!(cache.lock_inner)
    RDE_Env.reset_cache!(cache.lock_hist)
    return nothing
end

"""
    DenseLockReward(inner; bonus = 150)

Dense `inner` reward each step. When the eval lock criterion is met
(`StabilityReward` > 0.9 for 20 time units at the target count), terminate
and add `bonus`. Aligns train with eval (lock sooner is better) without
encoding a `u_p` schedule.
"""
@kwdef struct DenseLockReward{R <: AbstractScalarRewardStrategy, T <: AbstractFloat} <: AbstractScalarRewardStrategy
    inner::R
    lock::TransitionBasedReward{StabilityReward{T}, T} = TransitionBasedReward(StabilityReward(one(T)))
    bonus::T = 150.0f0
end

function DenseLockReward(
        inner::AbstractScalarRewardStrategy;
        bonus::T = 150.0f0,
        kwargs...,
    ) where {T <: AbstractFloat}
    lock = TransitionBasedReward(StabilityReward(one(T)); kwargs...)
    return DenseLockReward{typeof(inner), T}(inner, lock, bonus)
end

function RDE_Env.initialize_cache(rew_strat::DenseLockReward, N::Int, ::Type{T}) where {T}
    return DenseLockCache(
        RDE_Env.initialize_cache(rew_strat.inner, N, T),
        RDE_Env.initialize_cache(rew_strat.lock.wrapped_reward, N, T),
        TransitionBasedCache{T}(T[], Int[], false),
    )
end

function Base.show(io::IO, rew_strat::DenseLockReward)
    return print(io, "DenseLockReward(bonus=$(rew_strat.bonus), inner=$(rew_strat.inner))")
end

function Base.show(io::IO, ::MIME"text/plain", rew_strat::DenseLockReward)
    println(io, "DenseLockReward:")
    println(io, "  bonus: $(rew_strat.bonus)")
    println(io, "  inner: $(rew_strat.inner)")
    return println(io, "  lock: $(rew_strat.lock)")
end

function RDE_Env.compute_reward(
        env::RDEEnv{T},
        rew_strat::DenseLockReward,
        cache::DenseLockCache,
        context::TargetShockCountContext,
    ) where {T <: AbstractFloat}
    inner = RDE_Env.compute_reward(env, rew_strat.inner, cache.inner, context)
    stab = RDE_Env.compute_reward(env, rew_strat.lock.wrapped_reward, cache.lock_inner, context)
    stab_scalar = stab isa AbstractVector ? minimum(stab) : stab
    N = env.prob.params.N
    u = @view env.state[1:N]
    dx::T = RDE.get_dx(env.prob)
    current_shock_count = RDE.count_shocks(u, dx)
    target_shock_count = get_target_shock_count(context)
    push!(cache.lock_hist.past_rewards, T(stab_scalar))
    push!(cache.lock_hist.past_shock_counts, current_shock_count)
    if length(cache.lock_hist.past_shock_counts) >= 2
        cache.lock_hist.transition_found = detect_transition_realtime(
            rew_strat.lock, cache.lock_hist, env.dt, target_shock_count
        )
        if cache.lock_hist.transition_found
            env.terminated = true
            @logmsg LogLevel(-500) "DenseLockReward: lock at t=$(env.t)"
            env.info["Termination.Reason"] = "Transition detected"
            env.info["Termination.env_t"] = env.t
            return inner + T(rew_strat.bonus)
        end
    end
    return inner
end
