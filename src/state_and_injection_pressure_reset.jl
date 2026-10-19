"""
    StateAndInjectionPressureReset{T}

Reset strategy that copies stored `u`, `λ`, and injection pressure `u_p` into the problem.
Used with `RDE.reset_state_and_pressure!` (e.g. ramp segment continuation or replay from saved state).
"""
struct StateAndInjectionPressureReset{T <: AbstractFloat} <: RDE.AbstractReset
    u::Vector{T}
    λ::Vector{T}
    u_p::T
end

function RDE.reset_state_and_pressure!(
        prob::RDE.RDEProblem,
        r::StateAndInjectionPressureReset{T},
    ) where {T <: AbstractFloat}
    prob.u0 .= r.u
    prob.λ0 .= r.λ
    prob.params.u_p = r.u_p
    return nothing
end
