abstract type AbstractSchedule end

struct LinearSchedule{T} <: AbstractSchedule where {T <: Number}
    initial_value::T
    final_value::T
    function LinearSchedule(initial_value, final_value)
        return new{typeof(initial_value)}(initial_value, final_value)
    end
end

function (s::LinearSchedule)(p::Number)
    return s.final_value - (s.final_value - s.initial_value) * p
end
Base.string(s::LinearSchedule) = "LinearSchedule($(s.initial_value),$(s.final_value))"
struct CosineSchedule{T} <: AbstractSchedule where {T <: Number}
    high::T
    low::T
    period::Int
    function CosineSchedule(high, low, period)
        return new{typeof(high)}(high, low, period)
    end
end

function (s::CosineSchedule)(p::Number)
    return ((s.high + s.low) + (s.high - s.low) * cos((p - one(p)) * π * s.period)) / 2
end
Base.string(s::CosineSchedule) = "CosineSchedule($(s.high),$(s.low),$(s.period))"

struct ExpDecaySchedule{T} <: AbstractSchedule where {T <: Number}
    initial_value::T
    decay_rate::T
    function ExpDecaySchedule(initial_value, decay_rate = 2)
        return new{typeof(initial_value)}(initial_value, decay_rate)
    end
end

function (s::ExpDecaySchedule)(p::Number)
    return s.initial_value * exp(-s.decay_rate * (one(p) - p))
end
Base.string(s::ExpDecaySchedule) = "ExpDecaySchedule($(s.initial_value),$(s.decay_rate))"
