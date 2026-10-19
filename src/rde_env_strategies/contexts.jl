"""
    FixedTargetShockCountStrategy(target_shock_count::Int)

A context strategy that sets the target shock count to a fixed value.
"""
function get_target_shock_count end
function set_target_shock_count! end

mutable struct TargetShockCountContext <: AbstractCache
    target_shock_count::Int
end

get_target_shock_count(context::TargetShockCountContext) = context.target_shock_count
function set_target_shock_count!(context::TargetShockCountContext, v::Int)
    context.target_shock_count = v
    return nothing
end

get_target_shock_count(env::AbstractRDEEnv) = get_target_shock_count(env.cache.context)
function set_target_shock_count!(env::AbstractRDEEnv, v::Int)
    set_target_shock_count!(env.cache.context, v)
    return nothing
end

abstract type AbstractShockCountContextStrategy <: RDE_Env.AbstractContextStrategy end

struct FixedTargetShockCountStrategy <: AbstractShockCountContextStrategy
    target_shock_count::Int
end

function RDE_Env.initialize_cache(strategy::FixedTargetShockCountStrategy, ::Int, ::Type{T}) where {T}
    return TargetShockCountContext(strategy.target_shock_count)
end

function set_target_shock_count!(::FixedTargetShockCountStrategy, env::AbstractRDEEnv, v::Int)
    env.context_strat = FixedTargetShockCountStrategy(v)
    set_target_shock_count!(env.cache.context, v)
    return nothing
end

RDE_Env.on_reset!(::TargetShockCountContext, ::FixedTargetShockCountStrategy, ::AbstractRDEEnv) = nothing

"""
    RandomTargetShockCountStrategy(min_value::Int = 1, max_value::Int = 4)

A context strategy that sets the target shock count to a random value between min_value and max_value.
"""
@kwdef struct RandomTargetShockCountStrategy <: AbstractShockCountContextStrategy
    min_value::Int = 1
    max_value::Int = 4
end

function RDE_Env.initialize_cache(strategy::RandomTargetShockCountStrategy, ::Int, ::Type{T}) where {T}
    return TargetShockCountContext(rand(strategy.min_value:strategy.max_value))
end

function RDE_Env.on_reset!(context::TargetShockCountContext, strategy::RandomTargetShockCountStrategy, ::AbstractRDEEnv)
    context.target_shock_count = rand(strategy.min_value:strategy.max_value)
    return nothing
end

function set_target_shock_count!(::RandomTargetShockCountStrategy, ::AbstractRDEEnv, ::Int)
    error("Cannot set target shock count with RandomTargetShockCountStrategy, use FixedTargetShockCountStrategy instead.")
end

# ----------------------------------------------------------------------------
# EvalCycleTargetShockCountStrategy
# ----------------------------------------------------------------------------
@kwdef struct EvalCycleTargetShockCountStrategy <: AbstractShockCountContextStrategy
    repetitions_per_config::Int = 4
end
mutable struct EvalCycleTargetShockCountContext <: AbstractCache
    target_shocks::Vector{Int}
    current_config::Int
end

function RDE_Env.initialize_cache(strategy::EvalCycleTargetShockCountStrategy, ::Int, ::Type{T}) where {T}
    target_shocks = repeat(1:4, inner = strategy.repetitions_per_config * 3)
    return EvalCycleTargetShockCountContext(target_shocks, 1)
end

function RDE_Env.on_reset!(context::EvalCycleTargetShockCountContext, ::EvalCycleTargetShockCountStrategy, ::AbstractRDEEnv)
    context.current_config += 1
    if context.current_config > length(context.target_shocks)
        context.current_config = 1
    end
    return nothing
end

function get_target_shock_count(context::EvalCycleTargetShockCountContext)
    return context.target_shocks[context.current_config]
end

set_target_shock_count!(env::RDE_Env.MultiAgentRDEEnv, target_shock_count::Int) = set_target_shock_count!(env.core_env, target_shock_count)
get_target_shock_count(env::RDE_Env.MultiAgentRDEEnv) = get_target_shock_count(env.core_env)
