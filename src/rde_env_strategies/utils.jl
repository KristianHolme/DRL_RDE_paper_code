function sigmoid(x::T) where {T <: AbstractFloat}
    return one(x) / (one(x) + exp(-x))
end

function reward_sigmoid(x::T) where {T <: AbstractFloat}
    a = 5 * one(x)
    b = T(0.5)
    return sigmoid(a * (x - b))
end

function sigmoid_to_linear(x::T) where {T <: AbstractFloat}
    cutoff = T(0.5)
    if x < cutoff
        return reward_sigmoid(x)
    else
        return reward_sigmoid(cutoff) / cutoff * x
    end
end

function linear_to_sigmoid(x::T) where {T <: AbstractFloat}
    cutoff = T(0.5)
    if x < cutoff
        return x
    else
        return reward_sigmoid(x)
    end
end
