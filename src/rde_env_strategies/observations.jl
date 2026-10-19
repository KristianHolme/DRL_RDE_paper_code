# ============================================================================
# Cache Types
# ============================================================================

struct MinisectionCache{T <: AbstractFloat} <: AbstractCache
    minisection_u::Vector{T}
    minisection_λ::Vector{T}
end

struct PressureHistoryCache{T <: AbstractFloat} <: AbstractCache
    minisection_u::Vector{T}
    minisection_λ::Vector{T}
    pressure_history::Vector{T}
end

function RDE_Env.reset_cache!(cache::PressureHistoryCache{T}) where {T}
    cache.pressure_history .= T(-1)
    return nothing
end

mutable struct PressureHistoryWaveRewardCache{T <: AbstractFloat} <: AbstractCache
    pressure_history::Vector{T}
    last_reward::T
end

function RDE_Env.reset_cache!(cache::PressureHistoryWaveRewardCache{T}) where {T}
    cache.pressure_history .= T(-1)
    cache.last_reward = zero(T)
    return nothing
end

struct FourierMagnitudeActionHistoryCache{T <: AbstractFloat} <: AbstractCache
    pressure_history::Vector{T}
    dt_history::Vector{T}
end

function RDE_Env.reset_cache!(cache::FourierMagnitudeActionHistoryCache{T}) where {T}
    cache.pressure_history .= T(-1)
    cache.dt_history .= T(-1)
    return nothing
end

struct MultiCenteredPressureHistoryObservationCache{T <: AbstractFloat} <: AbstractCache
    minisection_u::Vector{T}
    minisection_λ::Vector{T}
    pressure_history::Matrix{T}  # Each column is history for one section
end

function RDE_Env.reset_cache!(cache::MultiCenteredPressureHistoryObservationCache{T}) where {T}
    cache.pressure_history .= T(-1)
    return nothing
end

struct MultiAgentSinusoidalPECache{T <: AbstractFloat, IC <: AbstractCache} <: AbstractCache
    inner_cache::IC
    pe_matrix::Matrix{T}
end

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
function compute_sectioned_observation!(
        minisection_observations_u,
        minisection_observations_λ,
        env::RDEEnv{T},
        obs_strategy::RDE_Env.AbstractObservationStrategy,
        context::TargetShockCountContext
    ) where {T}
    prob = env.prob #::RDEProblem{T, M, RS, C}
    N = prob.params.N
    dx = RDE.get_dx(prob)::T
    control_shift_strategy = control_shift(prob)

    if control_shift_strategy isa RDE_Env.MovingFrameControlShift
        current_u = copy(env.state[1:N])
        current_λ = copy(env.state[(N + 1):end])
        shift = Int(round(RDE.get_control_shift(control_shift_strategy, current_u, env.t) / dx))
        if shift != 0
            circshift!(current_u, -shift)
            circshift!(current_λ, -shift)
        end
    else
        current_u = @view env.state[1:N]
        current_λ = @view env.state[(N + 1):end]
    end
    max_shocks = T(4)
    max_pressure = T(2)

    minisections = obs_strategy.minisections
    minisection_size = N ÷ minisections

    get_minisection_observations!(minisection_observations_u, current_u, minisection_size)
    @. minisection_observations_u /= max_pressure
    get_minisection_observations!(minisection_observations_λ, current_λ, minisection_size)

    shocks::T = RDE.count_shocks(current_u, dx) / max_shocks
    target_shock_count::T = get_target_shock_count(context) / max_shocks

    return shocks, target_shock_count
end

function get_minisection_observations!(output, data, minisection_size)
    minisection_data = reshape(data, minisection_size, :)
    RDE.turbo_column_maximum!(output, minisection_data)
    return output
end

function get_minisection_observations(data, minisection_size)
    minisection_u = reshape(data, minisection_size, :)
    minisection_observations = RDE.turbo_column_maximum(minisection_u)
    return minisection_observations
end
# ============================================================================
# Observation Types and Implementations
# ============================================================================

# ----------------------------------------------------------------------------
# FourierObservation
# ----------------------------------------------------------------------------

struct FourierObservation <: RDE_Env.AbstractObservationStrategy
    fft_terms::Int
end

RDE_Env.initialize_cache(::FourierObservation, N::Int, ::Type{T}) where {T} = NoCache()

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::FourierObservation,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T}
    N = env.prob.params.N

    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]

    # env.cache.circ_u[:] .= current_u
    # env.cache.circ_λ[:] .= current_λ

    fft_u = abs.(fft(current_u)) #was circ_u
    fft_λ = abs.(fft(current_λ)) #was circ_λ

    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)

    # Create views directly into obs for each component
    u_obs = @view obs[1:n_terms]
    λ_obs = @view obs[(n_terms + 1):(2 * n_terms)]

    # Get views of relevant FFT terms (avoiding intermediate allocation)
    u_terms = @view fft_u[1:n_terms]
    λ_terms = @view fft_λ[1:n_terms]

    # Min-max normalization to [0,1] range, writing directly into obs
    u_terms_min = RDE.turbo_minimum(u_terms)
    u_terms_max = RDE.turbo_maximum(u_terms)
    λ_terms_min = RDE.turbo_minimum(λ_terms)
    λ_terms_max = RDE.turbo_maximum(λ_terms)

    u_range_inv = inv(u_terms_max - u_terms_min + T(1.0e-8))
    λ_range_inv = inv(λ_terms_max - λ_terms_min + T(1.0e-8))

    # Normalize and scale to [-1,1] in one pass
    @. u_obs = 2 * (u_terms - u_terms_min) * u_range_inv - 1
    @. λ_obs = 2 * (λ_terms - λ_terms_min) * λ_range_inv - 1

    # Control parameters already naturally bounded - write directly into obs
    obs[2 * n_terms + 1] = current_s(env.prob.injection) / env.smax
    obs[2 * n_terms + 2] = mean_u_p(env.prob.injection) / env.u_pmax

    return obs
end

# ----------------------------------------------------------------------------
# FourierMagnitudeObservation
# ----------------------------------------------------------------------------

# |rfft| of u and λ for the first n_modes bins, scaled by 1/N. Dim = 2 * n_modes.
@kwdef struct FourierMagnitudeObservation <: RDE_Env.AbstractObservationStrategy
    n_modes::Int = 8
end

RDE_Env.initialize_cache(::FourierMagnitudeObservation, ::Int, ::Type{T}) where {T} = NoCache()

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::FourierMagnitudeObservation,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T}
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]
    n_modes = min(strategy.n_modes, N ÷ 2 + 1)
    scale = inv(T(N))
    u_hat = rfft(current_u)
    λ_hat = rfft(current_λ)
    for i in 1:n_modes
        obs[i] = abs(u_hat[i]) * scale
        obs[n_modes + i] = abs(λ_hat[i]) * scale
    end
    return obs
end

function Base.show(io::IO, obs::FourierMagnitudeObservation)
    return print(io, "FourierMagnitudeObservation(n_modes=$(obs.n_modes))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::FourierMagnitudeObservation)
    println(io, "FourierMagnitudeObservation:")
    println(io, "  n_modes: $(obs.n_modes)")
    return nothing
end

# |rfft| of u and λ for the first n_modes bins, scaled by 1/N, plus last
# mean u_p / u_pmax, n_shocks/4, target/4. Dim = 2 * n_modes + 3.
@kwdef struct FourierMagnitudePressureShockObservation <: RDE_Env.AbstractObservationStrategy
    n_modes::Int = 8
end

RDE_Env.initialize_cache(::FourierMagnitudePressureShockObservation, ::Int, ::Type{T}) where {T} = NoCache()

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::FourierMagnitudePressureShockObservation,
        ::AbstractCache,
        context::TargetShockCountContext,
    ) where {T}
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]
    n_modes = min(strategy.n_modes, N ÷ 2 + 1)
    scale = inv(T(N))
    u_hat = rfft(current_u)
    λ_hat = rfft(current_λ)
    for i in 1:n_modes
        obs[i] = abs(u_hat[i]) * scale
        obs[n_modes + i] = abs(λ_hat[i]) * scale
    end
    dx = RDE.get_dx(env.prob)::T
    obs[2 * n_modes + 1] = mean_u_p(env.prob.injection) / env.u_pmax
    obs[2 * n_modes + 2] = RDE.count_shocks(current_u, dx) / T(4)
    obs[2 * n_modes + 3] = get_target_shock_count(context) / T(4)
    return obs
end

function Base.show(io::IO, obs::FourierMagnitudePressureShockObservation)
    return print(io, "FourierMagnitudePressureShockObservation(n_modes=$(obs.n_modes))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::FourierMagnitudePressureShockObservation)
    println(io, "FourierMagnitudePressureShockObservation:")
    println(io, "  n_modes: $(obs.n_modes)")
    return nothing
end

function observation_dt_max(env::RDEEnv{T}) where {T}
    action_strat = env.action_strat
    if action_strat isa LinearScalarPressureWithDtAction
        return action_strat.dt_max
    else
        return env.dt
    end
end

# |rfft| of u and λ, last `history_length` mean u_p / u_pmax and dt / dt_max,
# n_shocks/4, target/4. Dim = 2 * n_modes + 2 * history_length + 2.
@kwdef struct FourierMagnitudeActionHistoryObservation <: RDE_Env.AbstractObservationStrategy
    n_modes::Int = 8
    history_length::Int = 8
end

function RDE_Env.initialize_cache(
        obs::FourierMagnitudeActionHistoryObservation,
        ::Int,
        ::Type{T},
    ) where {T}
    sentinel = fill(T(-1), obs.history_length)
    return FourierMagnitudeActionHistoryCache{T}(copy(sentinel), copy(sentinel))
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::FourierMagnitudeActionHistoryObservation,
        cache::FourierMagnitudeActionHistoryCache{T},
        context::TargetShockCountContext,
    ) where {T}
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]
    n_modes = min(strategy.n_modes, N ÷ 2 + 1)
    history_length = strategy.history_length
    scale = inv(T(N))
    u_hat = rfft(current_u)
    λ_hat = rfft(current_λ)
    for i in 1:n_modes
        obs[i] = abs(u_hat[i]) * scale
        obs[n_modes + i] = abs(λ_hat[i]) * scale
    end

    current_pressure = mean_u_p(env.prob.injection) / env.u_pmax
    dt_max = observation_dt_max(env)
    current_dt = env.dt / dt_max
    pressure_view = @view obs[(2 * n_modes + 1):(2 * n_modes + history_length)]
    dt_view = @view obs[(2 * n_modes + history_length + 1):(2 * n_modes + 2 * history_length)]
    if all(≈(-1), cache.pressure_history)
        cache.pressure_history .= current_pressure
        cache.dt_history .= current_dt
        pressure_view .= current_pressure
        dt_view .= current_dt
    else
        circshift!(cache.pressure_history, -1)
        circshift!(cache.dt_history, -1)
        cache.pressure_history[end] = current_pressure
        cache.dt_history[end] = current_dt
        pressure_view .= cache.pressure_history
        dt_view .= cache.dt_history
    end

    dx = RDE.get_dx(env.prob)::T
    obs[2 * n_modes + 2 * history_length + 1] = RDE.count_shocks(current_u, dx) / T(4)
    obs[2 * n_modes + 2 * history_length + 2] = get_target_shock_count(context) / T(4)
    return obs
end

function Base.show(io::IO, obs::FourierMagnitudeActionHistoryObservation)
    return print(
        io,
        "FourierMagnitudeActionHistoryObservation(n_modes=$(obs.n_modes), history_length=$(obs.history_length))",
    )
end

function Base.show(io::IO, ::MIME"text/plain", obs::FourierMagnitudeActionHistoryObservation)
    println(io, "FourierMagnitudeActionHistoryObservation:")
    println(io, "  n_modes: $(obs.n_modes)")
    println(io, "  history_length: $(obs.history_length)")
    return nothing
end

# ----------------------------------------------------------------------------
# StateObservation
# ----------------------------------------------------------------------------

struct StateObservation <: RDE_Env.AbstractObservationStrategy end

RDE_Env.initialize_cache(::StateObservation, N::Int, ::Type{T}) where {T} = NoCache()
function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        rt::StateObservation,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T}
    N = length(env.state) ÷ 2
    u = @view env.state[1:N]
    λ = @view env.state[(N + 1):end]

    ϵ = T(1.0e-8)
    u_max_inv = inv(max(RDE.turbo_maximum_abs(u), ϵ))
    λ_max_inv = inv(max(RDE.turbo_maximum_abs(λ), ϵ))

    # Create views into obs for each component
    obs_u = @view obs[1:N]
    obs_λ = @view obs[(N + 1):(2 * N)]

    # Normalize directly into obs
    @. obs_u = u * u_max_inv
    @. obs_λ = λ * λ_max_inv

    # Write control parameters directly to the last two positions
    obs[2 * N + 1] = current_s(env.prob.injection) / env.smax
    obs[2 * N + 2] = mean_u_p(env.prob.injection) / env.u_pmax

    return obs
end
# ----------------------------------------------------------------------------
# SectionedStateObservation
# ----------------------------------------------------------------------------

@kwdef struct SectionedStateObservation <: RDE_Env.AbstractObservationStrategy
    minisections::Int = 32
end

# ----------------------------------------------------------------------------
# SectionedStateWithPressureHistoryObservation
# ----------------------------------------------------------------------------

@kwdef struct SectionedStateWithPressureHistoryObservation <: RDE_Env.AbstractObservationStrategy
    minisections::Int = 32
    history_length::Int = 14
end

RDE_Env.initialize_cache(obs::SectionedStateObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    MinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

RDE_Env.initialize_cache(obs::SectionedStateWithPressureHistoryObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    PressureHistoryCache{T}(
        zeros(T, n_total_minisections),
        zeros(T, n_total_minisections),
        fill(T(-1), obs.history_length)
    )
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::SectionedStateObservation,
        ::MinisectionCache{T},
        context::TargetShockCountContext,
    ) where {T}
    minisections = strategy.minisections

    # Create views into obs for each component
    minisection_observations_u = @view obs[1:minisections]
    minisection_observations_λ = @view obs[(minisections + 1):(2 * minisections)]

    shocks_normalized, target_shock_count_normalized = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, strategy, context
    )

    # Write scalar values directly to the end of obs
    obs[2 * minisections + 1] = shocks_normalized
    obs[2 * minisections + 2] = target_shock_count_normalized

    return obs
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::SectionedStateWithPressureHistoryObservation,
        pressure_cache::PressureHistoryCache{T},
        context::TargetShockCountContext,
    ) where {T}
    minisections = strategy.minisections
    history_length = strategy.history_length

    # Create views into obs for each component
    minisection_observations_u = @view obs[1:minisections]
    minisection_observations_λ = @view obs[(minisections + 1):(2 * minisections)]
    pressure_history_view = @view obs[(2 * minisections + 1):(2 * minisections + history_length)]

    shocks_normalized, target_shock_count_normalized = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, strategy, context
    )

    # Get current mean injection pressure
    current_pressure = mean_u_p(env.prob.injection) / env.u_pmax

    # Check if this is the first observation (all values are -1)
    if all(≈(-1), pressure_cache.pressure_history)
        # First observation: fill entire history with current pressure
        pressure_cache.pressure_history .= current_pressure
        pressure_history_view .= current_pressure
    else
        # Subsequent observations: shift history and add new value
        circshift!(pressure_cache.pressure_history, -1)
        pressure_cache.pressure_history[end] = current_pressure
        pressure_history_view .= pressure_cache.pressure_history
    end

    # Write scalar values directly to the end of obs
    obs[2 * minisections + history_length + 1] = shocks_normalized
    obs[2 * minisections + history_length + 2] = target_shock_count_normalized

    return obs
end

# ----------------------------------------------------------------------------
# PressureHistoryWaveRewardObservation
# ----------------------------------------------------------------------------

# Wave count / 4, last `history_length` mean u_p / u_pmax, previous env.reward. Dim = history_length + 2.
@kwdef struct PressureHistoryWaveRewardObservation <: RDE_Env.AbstractObservationStrategy
    history_length::Int = 8
end

function RDE_Env.initialize_cache(
        obs::PressureHistoryWaveRewardObservation,
        ::Int,
        ::Type{T},
    ) where {T}
    return PressureHistoryWaveRewardCache{T}(fill(T(-1), obs.history_length), zero(T))
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::PressureHistoryWaveRewardObservation,
        cache::PressureHistoryWaveRewardCache{T},
        ::AbstractCache,
    ) where {T}
    history_length = strategy.history_length
    current_pressure = mean_u_p(env.prob.injection) / env.u_pmax
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    dx = RDE.get_dx(env.prob)::T
    current_waves = RDE.count_shocks(current_u, dx) / T(4)
    pressure_history_view = @view obs[2:(history_length + 1)]
    if all(≈(-1), cache.pressure_history)
        cache.pressure_history .= current_pressure
        pressure_history_view .= current_pressure
    else
        circshift!(cache.pressure_history, -1)
        cache.pressure_history[end] = current_pressure
        pressure_history_view .= cache.pressure_history
    end
    obs[1] = current_waves
    obs[history_length + 2] = cache.last_reward
    cache.last_reward = T(env.reward)
    return obs
end

function Base.show(io::IO, obs::PressureHistoryWaveRewardObservation)
    return print(io, "PressureHistoryWaveRewardObservation(history_length=$(obs.history_length))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::PressureHistoryWaveRewardObservation)
    println(io, "PressureHistoryWaveRewardObservation:")
    println(io, "  history_length: $(obs.history_length)")
    return nothing
end

# ----------------------------------------------------------------------------
# SampledStateObservation
# ----------------------------------------------------------------------------

struct SampledStateObservation <: RDE_Env.AbstractObservationStrategy
    n_samples::Int
end

RDE_Env.initialize_cache(::SampledStateObservation, N::Int, ::Type{T}) where {T} = NoCache()

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::SampledStateObservation,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T}
    N = env.prob.params.N
    n = strategy.n_samples

    u = @view env.state[1:N]
    λ = @view env.state[(N + 1):end]

    # Create views into obs for each component
    sampled_u = @view obs[1:n]
    sampled_λ = @view obs[(n + 1):(2 * n)]

    # Sample the data directly into obs (avoiding indices array allocation)
    step = (N - 1) / (n - 1)
    for i in 1:n
        idx = round(Int, 1 + (i - 1) * step)
        sampled_u[i] = u[idx]
        sampled_λ[i] = λ[idx]
    end

    ϵ = T(1.0e-8)
    u_max_inv = inv(max(RDE.turbo_maximum_abs(sampled_u), ϵ))
    λ_max_inv = inv(max(RDE.turbo_maximum_abs(sampled_λ), ϵ))

    # Normalize in place
    @. sampled_u *= u_max_inv
    @. sampled_λ *= λ_max_inv

    # Write normalized time directly to obs
    obs[2 * n + 1] = env.t / env.prob.params.tmax

    return obs
end
# ----------------------------------------------------------------------------
# CompositeObservation
# ----------------------------------------------------------------------------

@kwdef mutable struct CompositeObservation <: RDE_Env.AbstractObservationStrategy
    fft_terms::Int = 8
end

RDE_Env.initialize_cache(::CompositeObservation, N::Int, ::Type{T}) where {T} = NoCache()

function Base.show(io::IO, obs::CompositeObservation)
    return print(io, "CompositeObservation(fft_terms=$(obs.fft_terms))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::CompositeObservation)
    println(io, "CompositeObservation:")
    return println(io, "  fft_terms: $(obs.fft_terms)")
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        strategy::CompositeObservation,
        ::AbstractCache,
        context::TargetShockCountContext,
    ) where {T}
    N = env.prob.params.N

    current_u = @view env.state[1:N]

    # env.cache.circ_u[:] .= current_u

    fft_u = abs.(fft(current_u)) #was circ_u

    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)::Int

    # Create view into obs for FFT observations
    u_obs = @view obs[1:n_terms]

    # Take relevant FFT terms (view to avoid allocation)
    u_terms = @view fft_u[2:(n_terms + 1)] #the first is always one, so not useful
    u_terms_min = RDE.turbo_minimum(u_terms)
    u_terms_max = RDE.turbo_maximum(u_terms)

    # Normalize directly into obs
    u_range_inv = inv(u_terms_max - u_terms_min + T(1.0e-8))
    @. u_obs = (u_terms - u_terms_min) * u_range_inv

    dx = RDE.get_dx(env.prob)::T
    max_shocks = T(6)
    u_min, u_max = RDE.turbo_extrema(current_u)
    span = u_max - u_min

    # Write scalar values directly to the end of obs
    obs[n_terms + 1] = mean_u_p(env.prob.injection) / env.u_pmax
    obs[n_terms + 2] = RDE.count_shocks(current_u, dx) / max_shocks
    obs[n_terms + 3] = get_target_shock_count(env) / max_shocks
    obs[n_terms + 4] = span / T(3)

    return obs
end

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# MultiCenteredObservation
# ----------------------------------------------------------------------------

@kwdef struct MultiCenteredObservation <: RDE_Env.AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    minisections::Int = 32
end
MultiCenteredObservation(n::Int) = MultiCenteredObservation(n_sections = n)

@kwdef struct MultiCenteredWithPressureHistoryObservation <: RDE_Env.AbstractMultiAgentObservationStrategy
    n_sections::Int = 8
    minisections::Int = 32
    history_length::Int = 14
end
MultiCenteredWithPressureHistoryObservation(n::Int) = MultiCenteredWithPressureHistoryObservation(n_sections = n)

function Base.show(io::IO, obs_strategy::MultiCenteredObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredObservation(n_sections=$(obs_strategy.n_sections), minisections=$(obs_strategy.minisections))")
    end
end

RDE_Env.initialize_cache(obs::MultiCenteredObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    MinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

RDE_Env.initialize_cache(obs::MultiCenteredWithPressureHistoryObservation, N::Int, ::Type{T}) where {T} = begin
    MultiCenteredPressureHistoryObservationCache{T}(
        zeros(T, obs.minisections),
        zeros(T, obs.minisections),
        fill(T(-1), obs.history_length, obs.n_sections)
    )
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredObservation)
    println(io, "MultiCenteredObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    return println(io, "  minisections: $(obs_strategy.minisections)")
end

function Base.show(io::IO, obs_strategy::MultiCenteredWithPressureHistoryObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredWithPressureHistoryObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredWithPressureHistoryObservation(n_sections=$(obs_strategy.n_sections), minisections=$(obs_strategy.minisections), history_length=$(obs_strategy.history_length))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredWithPressureHistoryObservation)
    println(io, "MultiCenteredWithPressureHistoryObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    println(io, "  minisections: $(obs_strategy.minisections)")
    return println(io, "  history_length: $(obs_strategy.history_length)")
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T, A, O, R, G, V, OBS},
        obs_strategy::MultiCenteredObservation,
        obs_cache::MinisectionCache{T},
        context::TargetShockCountContext,
    ) where {T, A, O, R, G, V, OBS}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections

    # Use cached buffers for minisection observations
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ

    shocks, target_shock_count = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, obs_strategy, context
    )

    # Fill the matrix directly
    for i in 1:n_sections
        # Get views into obs for this section
        obs_u_view = @view obs[1:minisections, i]
        obs_λ_view = @view obs[(minisections + 1):(minisections * 2), i]

        # Copy with circshift directly into obs
        obs_u_view .= circshift(minisection_observations_u, -minisections_per_section * (i - 1))
        obs_λ_view .= circshift(minisection_observations_λ, -minisections_per_section * (i - 1))

        # Add the additional values
        obs[end - 1, i] = shocks #is this needed?
        obs[end, i] = target_shock_count
    end

    return obs
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        obs_strategy::MultiCenteredWithPressureHistoryObservation,
        obs_cache::PressureHistoryCache{T},
        context::TargetShockCountContext,
    ) where {T}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    history_length = obs_strategy.history_length
    minisections_per_section = minisections ÷ n_sections

    # Use cached buffers
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ
    pressure_history_cache = obs_cache.pressure_history

    shocks, target_shock_count = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, obs_strategy, context
    )

    # Calculate points per section for extracting section-specific pressure
    N = env.prob.params.N
    points_per_section = N ÷ n_sections
    u_p_current = current_u_p(env.prob.injection)

    # Fill the matrix for each section
    for i in 1:n_sections
        # Views into obs: [u, λ, pressure_history, shocks, target]
        obs_u_view = @view obs[1:minisections, i]
        obs_λ_view = @view obs[(minisections + 1):(minisections * 2), i]
        pressure_history_view = @view obs[(2 * minisections + 1):(2 * minisections + history_length), i]

        # Copy with circshift! for centered observations (in-place, no allocation)
        circshift!(obs_u_view, minisection_observations_u, -minisections_per_section * (i - 1))
        circshift!(obs_λ_view, minisection_observations_λ, -minisections_per_section * (i - 1))

        # Get section-specific mean pressure (normalized)
        section_start = (i - 1) * points_per_section + 1
        section_end = i * points_per_section
        current_section_pressure = if u_p_current isa AbstractVector
            mean(@view u_p_current[section_start:section_end]) / env.u_pmax
        else
            u_p_current / env.u_pmax
        end

        # Handle pressure history for this section
        section_history = @view pressure_history_cache[:, i]
        if all(≈(-1), section_history)
            # First observation: fill entire history with current pressure
            section_history .= current_section_pressure
            pressure_history_view .= current_section_pressure
        else
            # Subsequent observations: shift history in-place and add new value
            circshift!(section_history, -1)
            section_history[end] = current_section_pressure
            pressure_history_view .= section_history
        end

        # Add scalar values at the end
        obs[end - 1, i] = shocks
        obs[end, i] = target_shock_count
    end

    return obs
end

# ----------------------------------------------------------------------------
# MultiCenteredWithIndexObservation
# ----------------------------------------------------------------------------

@kwdef struct MultiCenteredWithIndexObservation <: RDE_Env.AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    minisections::Int = 32
end
MultiCenteredWithIndexObservation(n::Int) = MultiCenteredWithIndexObservation(n_sections = n)

RDE_Env.initialize_cache(obs::MultiCenteredWithIndexObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    MinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

function Base.show(io::IO, obs_strategy::MultiCenteredWithIndexObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredWithIndexObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredWithIndexObservation(n_sections=$(obs_strategy.n_sections), minisections=$(obs_strategy.minisections))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredWithIndexObservation)
    println(io, "MultiCenteredWithIndexObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    return println(io, "  minisections: $(obs_strategy.minisections)")
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        obs_strategy::MultiCenteredWithIndexObservation,
        obs_cache::MinisectionCache{T},
        context::TargetShockCountContext,
    ) where {T}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections

    # Use cached buffers for minisection observations
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ

    shocks, target_shock_count = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, obs_strategy, context
    )

    # Fill the matrix directly
    for i in 1:n_sections
        # Get views into obs for this section
        obs_u_view = @view obs[1:minisections, i]
        obs_λ_view = @view obs[(minisections + 1):(minisections * 2), i]

        # Copy with circshift directly into obs
        obs_u_view .= circshift(minisection_observations_u, -minisections_per_section * (i - 1))
        obs_λ_view .= circshift(minisection_observations_λ, -minisections_per_section * (i - 1))

        # Add the additional values
        obs[end - 2, i] = shocks
        obs[end - 1, i] = target_shock_count
        obs[end, i] = T(i) / T(n_sections)  # Section index normalized by number of sections
    end

    return obs
end

# ----------------------------------------------------------------------------
# MeanInjectionPressureObservation
# ----------------------------------------------------------------------------

struct MeanInjectionPressureObservation <: RDE_Env.AbstractObservationStrategy end

RDE_Env.initialize_cache(::MeanInjectionPressureObservation, N::Int, ::Type{T}) where {T} = NoCache()

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T, A, O, R, V, OBS},
        strategy::MeanInjectionPressureObservation,
        ::AbstractCache,
        ::AbstractCache,
    ) where {T, A, O, R, V, OBS}
    obs[1] = mean_u_p(env.prob.injection)
    return obs
end

# ----------------------------------------------------------------------------
# MultiCenteredMovingFrameObservation
# ----------------------------------------------------------------------------

@kwdef struct MultiCenteredMovingFrameObservation <: RDE_Env.AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    minisections::Int = 32
end
MultiCenteredMovingFrameObservation(n::Int) = MultiCenteredMovingFrameObservation(n_sections = n)

function Base.show(io::IO, obs_strategy::MultiCenteredMovingFrameObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredMovingFrameObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredMovingFrameObservation(n_sections=$(obs_strategy.n_sections), minisections=$(obs_strategy.minisections))")
    end
end

RDE_Env.initialize_cache(obs::MultiCenteredMovingFrameObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    MinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredMovingFrameObservation)
    println(io, "MultiCenteredMovingFrameObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    println(io, "  minisections: $(obs_strategy.minisections)")
    return nothing
end


function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T, A, O, R, G, V, OBS},
        obs_strategy::MultiCenteredMovingFrameObservation,
        obs_cache::MinisectionCache{T},
        context::TargetShockCountContext,
    ) where {T, A, O, R, G, V, OBS}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections

    # Use cached buffers for minisection observations
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ
    control_shift_strategy = control_shift(env.prob)

    t = env.t
    dx = RDE.get_dx(env.prob)
    if t ≈ 0.0f0 #TODO: move into interface for control shift strategies?
        control_shift_strategy.position = 0.0f0
        control_shift_strategy.velocity = 0.0f0 # we dont shift observation at start
    else
        @assert !isnothing(env.prob.sol) "env.prob.sol is nothing"
        us, _ = RDE.split_sol(env.prob.sol.u)
        avg_speed = RDE_Env.get_avg_wave_speed(us, env.prob.sol.t, dx)
        if avg_speed > zero(T)
            control_shift_strategy.velocity = avg_speed
        end
        control_shift_strategy.position += avg_speed * env.dt # position at end of step
        L = env.prob.params.L
        control_shift_strategy.position = mod(control_shift_strategy.position, L)
    end
    control_shift_strategy.t_last = env.t

    shocks, target_shock_count = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, obs_strategy, context
    )

    if t ≈ 0.0f0
        # we guess at speed to shift u_p by in the first step
        control_shift_strategy.velocity = 1.71f0
    end

    # Fill the matrix directly
    for i in 1:n_sections
        # Get views into obs for this section
        obs_u_view = @view obs[1:minisections, i]
        obs_λ_view = @view obs[(minisections + 1):(minisections * 2), i]

        # Copy with circshift directly into obs
        obs_u_view .= circshift(minisection_observations_u, -minisections_per_section * (i - 1))
        obs_λ_view .= circshift(minisection_observations_λ, -minisections_per_section * (i - 1))

        # Add the additional values
        obs[end - 1, i] = shocks #is this needed?
        obs[end, i] = target_shock_count
    end

    return obs
end


# ----------------------------------------------------------------------------
# MultiAgentObservationWithSinusoidalPE
# ----------------------------------------------------------------------------

const MultiCenteredMAInner =
    Union{MultiCenteredObservation, MultiCenteredMovingFrameObservation}

@kwdef struct MultiAgentObservationWithSinusoidalPE{
        S <: MultiCenteredMAInner,
        T <: AbstractFloat,
    } <: RDE_Env.AbstractMultiAgentObservationStrategy
    inner::S
    scale::T
    n_pe_rows::Int = 64
end

function Base.getproperty(obs::MultiAgentObservationWithSinusoidalPE, name::Symbol)
    if name === :n_sections
        return getfield(obs, :inner).n_sections
    elseif name === :minisections
        return getfield(obs, :inner).minisections
    else
        return getfield(obs, name)
    end
end

function _fill_pe_matrix!(pe_matrix::Matrix{T}, n_sections::Int, d::Int) where {T}
    n_pe_apply = size(pe_matrix, 1)
    for i in 1:n_sections
        pos_i = T(2π) * T(i - 1) / T(n_sections)
        for row in 1:n_pe_apply
            k = (row - 1) ÷ 2
            divisor = T(10000)^(T(2 * k) / T(d))
            if iseven(row)
                pe_matrix[row, i] = sin(pos_i / divisor)
            else
                pe_matrix[row, i] = cos(pos_i / divisor)
            end
        end
    end
    return nothing
end

function RDE_Env.initialize_cache(
        obs::MultiAgentObservationWithSinusoidalPE,
        N::Int,
        ::Type{UT},
    ) where {UT <: AbstractFloat}
    inner = obs.inner
    inner_cache = RDE_Env.initialize_cache(inner, N, UT)
    n_pe_apply = min(obs.n_pe_rows, 2 * inner.minisections)
    n_sections = inner.n_sections
    pe_matrix = zeros(UT, n_pe_apply, n_sections)
    _fill_pe_matrix!(pe_matrix, n_sections, 64)
    return MultiAgentSinusoidalPECache{UT, typeof(inner_cache)}(inner_cache, pe_matrix)
end

function Base.show(io::IO, obs_strategy::MultiAgentObservationWithSinusoidalPE)
    return print(
        io,
        "MultiAgentObservationWithSinusoidalPE(inner=$(obs_strategy.inner), scale=$(obs_strategy.scale))",
    )
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        obs_strategy::MultiAgentObservationWithSinusoidalPE,
        obs_cache::MultiAgentSinusoidalPECache{T},
        context::TargetShockCountContext,
    ) where {T}
    inner = obs_strategy.inner
    RDE_Env.compute_observation!(obs, env, inner, obs_cache.inner_cache, context)
    n_pe_apply = size(obs_cache.pe_matrix, 1)
    scale_T = T(obs_strategy.scale)
    obs[1:n_pe_apply, :] .+= scale_T .* obs_cache.pe_matrix
    return obs
end

function RDE_Env._observation_space(
        params::RDEParam{T},
        strategy::MultiAgentObservationWithSinusoidalPE,
    ) where {T}
    return RDE_Env._observation_space(params, strategy.inner)
end


# ----------------------------------------------------------------------------
# SectionedStateMovingFrameObservation
# ----------------------------------------------------------------------------

@kwdef struct SectionedStateMovingFrameObservation <: RDE_Env.AbstractObservationStrategy
    minisections::Int = 32
end

RDE_Env.initialize_cache(obs::SectionedStateMovingFrameObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    MinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T},
        obs_strategy::SectionedStateMovingFrameObservation,
        obs_cache::MinisectionCache{T},
        context::TargetShockCountContext,
    ) where {T}
    minisections = obs_strategy.minisections

    # Use cached buffers for minisection observations
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ
    control_shift_strategy = control_shift(env.prob)

    t = env.t
    dx = RDE.get_dx(env.prob)
    if t ≈ zero(T) #TODO: move into interface for control shift strategies?
        control_shift_strategy.position = zero(T)
        control_shift_strategy.velocity = zero(T) # we dont shift at start
    else
        @assert !isnothing(env.prob.sol) "env.prob.sol is nothing"
        us, _ = RDE.split_sol(env.prob.sol.u)
        avg_speed = RDE_Env.get_avg_wave_speed(us, env.prob.sol.t, dx)
        if avg_speed > zero(T)
            control_shift_strategy.velocity = avg_speed
        end
        control_shift_strategy.position += avg_speed * env.dt # position at end of step
        L = env.prob.params.L
        control_shift_strategy.position = mod(control_shift_strategy.position, L)
    end
    control_shift_strategy.t_last = env.t

    shocks, target_shock_count = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, obs_strategy, context
    )

    if t ≈ zero(T)
        # we guess at speed to shift u_p by in the first step
        control_shift_strategy.velocity = T(1.71)
    end

    # Get views into obs
    obs_u_view = @view obs[1:minisections]
    obs_λ_view = @view obs[(minisections + 1):(minisections * 2)]

    # Copy directly into obs (no circshift since it's single agent/global frame relative to wave)
    # But wait, compute_sectioned_observation! already handles the shifting if control_shift_strategy is MovingFrameControlShift
    # because it uses control_shift(env.prob) inside.
    obs_u_view .= minisection_observations_u
    obs_λ_view .= minisection_observations_λ

    # Add the additional values
    obs[end - 1] = shocks
    obs[end] = target_shock_count

    return obs
end

function RDE_Env._observation_space(params::RDEParam{T}, strategy::FourierObservation) where {T}
    N = params.N
    n_fft_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    low = vcat(fill(-one(T), 2 * n_fft_terms), zero(T), zero(T))
    high = vcat(fill(one(T), 2 * n_fft_terms), one(T), one(T))
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(params::RDEParam{T}, strategy::FourierMagnitudeObservation) where {T}
    N = params.N
    n_modes = min(strategy.n_modes, N ÷ 2 + 1)
    n = 2 * n_modes
    return Drill.Box(zeros(T, n), fill(T(10), n))
end

function RDE_Env._observation_space(
        params::RDEParam{T}, strategy::FourierMagnitudePressureShockObservation
    ) where {T}
    N = params.N
    n_modes = min(strategy.n_modes, N ÷ 2 + 1)
    n_fft = 2 * n_modes
    n_tail = 3
    n = n_fft + n_tail
    low = vcat(zeros(T, n_fft), zeros(T, n_tail))
    high = vcat(fill(T(10), n_fft), ones(T, n_tail))
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(
        params::RDEParam{T}, strategy::FourierMagnitudeActionHistoryObservation
    ) where {T}
    N = params.N
    n_modes = min(strategy.n_modes, N ÷ 2 + 1)
    h = strategy.history_length
    n_fft = 2 * n_modes
    n_hist = 2 * h
    n_tail = 2
    n = n_fft + n_hist + n_tail
    low = vcat(zeros(T, n_fft), zeros(T, n_hist), zeros(T, n_tail))
    high = vcat(fill(T(10), n_fft), ones(T, n_hist), ones(T, n_tail))
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::PressureHistoryWaveRewardObservation) where {T}
    n = strategy.history_length + 2
    return Drill.Box(zeros(T, n), ones(T, n))
end

function RDE_Env._observation_space(params::RDEParam{T}, ::StateObservation) where {T}
    N = params.N
    low = vcat(fill(-one(T), 2N), zero(T), zero(T))
    high = vcat(fill(one(T), 2N), one(T), one(T))
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::SectionedStateObservation) where {T}
    minisections = strategy.minisections
    low = vcat(zeros(T, 2 * minisections), zero(T), zero(T))
    high = vcat(fill(T(5), 2 * minisections), T(5), T(5))
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::SectionedStateMovingFrameObservation) where {T}
    minisections = strategy.minisections
    low = vcat(zeros(T, 2 * minisections), zero(T), zero(T))
    high = vcat(fill(T(5), 2 * minisections), T(5), T(5))
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::SampledStateObservation) where {T}
    n_samples = strategy.n_samples
    low = vcat(zeros(T, 2 * n_samples), zero(T))
    high = vcat(fill(T(5), 2 * n_samples), T(5))
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(params::RDEParam{T}, strategy::CompositeObservation) where {T}
    N = params.N
    n_fft_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    low = vcat(fill(-one(T), n_fft_terms), zero(T), zero(T), zero(T))
    high = vcat(fill(T(5), n_fft_terms), T(5), T(5), T(5))
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(params::RDEParam{T}, ::MeanInjectionPressureObservation) where {T}
    return Drill.Box([zero(T)], [T(1.2)]) #XXX bad hardcoded value
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::MultiCenteredObservation) where {T}
    obs_length = strategy.minisections * 2 + 2
    n_agents = strategy.n_sections
    low = zeros(T, obs_length, n_agents)
    high = ones(T, obs_length, n_agents)
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::SectionedStateWithPressureHistoryObservation) where {T}
    minisections = strategy.minisections
    history_length = strategy.history_length
    obs_length = 2 * minisections + history_length + 2
    low = zeros(T, obs_length)
    high = ones(T, obs_length)
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::MultiCenteredWithPressureHistoryObservation) where {T}
    minisections = strategy.minisections
    history_length = strategy.history_length
    obs_length = 2 * minisections + history_length + 2
    n_agents = strategy.n_sections
    low = zeros(T, obs_length, n_agents)
    high = fill(T(5), obs_length, n_agents)
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::MultiCenteredMovingFrameObservation) where {T}
    minisections = strategy.minisections
    obs_length = 2 * minisections + 2
    n_agents = strategy.n_sections
    low = zeros(T, obs_length, n_agents)
    high = fill(T(5), obs_length, n_agents)
    return Drill.Box(low, high)
end

function RDE_Env._observation_space(::RDEParam{T}, strategy::MultiCenteredWithIndexObservation) where {T}
    minisections = strategy.minisections
    obs_length = 2 * minisections + 3
    n_agents = strategy.n_sections
    low = zeros(T, obs_length, n_agents)
    high = fill(T(5), obs_length, n_agents)
    return Drill.Box(low, high)
end

function RDE_Env._multi_agent_action_space(::RDEParam, ::VectorPressureAction)
    return Drill.Box([-1.0f0], [1.0f0])
end

function RDE_Env._multi_agent_action_space(::RDEParam, ::LinearVectorPressureAction)
    return Drill.Box([-1.0f0], [1.0f0])
end

function RDE_Env._multi_agent_action_space(env::RDE_Env.RDEEnv, action_strat::VectorPressureAction)
    return Drill.Box([-1.0f0], [1.0f0])
end

function RDE_Env._multi_agent_action_space(env::RDE_Env.RDEEnv, action_strat::LinearVectorPressureAction)
    return Drill.Box([-1.0f0], [1.0f0])
end
