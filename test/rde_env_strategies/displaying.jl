#TODO: move to appropriate files
function Base.show(io::IO, obs::SampledStateObservation)
    return print(io, "SampledStateObservation(n_samples=$(obs.n_samples))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::SampledStateObservation)
    println(io, "SampledStateObservation:")
    println(io, "  n_samples: $(obs.n_samples)")
    return nothing
end

function Base.show(io::IO, obs::SectionedStateObservation)
    print(io, "SectionedStateObservation(minisections=$(obs.minisections))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", obs::SectionedStateObservation)
    println(io, "SectionedStateObservation:")
    println(io, "  minisections: $(obs.minisections)")
    return nothing
end

function Base.show(io::IO, ::StateObservation)
    print(io, "StateObservation()")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ::StateObservation)
    println(io, "StateObservation: returns full state")
    return nothing
end

function Base.show(io::IO, ::FullStateObservation)
    print(io, "FullStateObservation()")
    return nothing
end

function Base.show(io::IO, obs::FourierObservation)
    print(io, "FourierObservation(fft_terms=$(obs.fft_terms))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", obs::FourierObservation)
    println(io, "FourierObservation:")
    println(io, "  fft_terms: $(obs.fft_terms)")
    return nothing
end


function Base.show(io::IO, a::ScalarAreaScalarPressureAction)
    print(io, "ScalarAreaScalarPressureAction()")
    return nothing
end

function Base.show(io::IO, a::VectorPressureAction)
    print(io, "VectorPressureAction(n_sections=$(a.n_sections))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", a::VectorPressureAction)
    println(io, "VectorPressureAction:")
    println(io, "  n_sections: $(a.n_sections)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockSpanReward)
    println(io, "ShockSpanReward:")
    println(io, "  span_scale: $(rt.span_scale)")
    println(io, "  shock_weight: $(rt.shock_weight)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockPreservingReward)
    println(io, "ShockPreservingReward:")
    println(io, "  span_scale: $(rt.span_scale)")
    println(io, "  shock_weight: $(rt.shock_weight)")
    return println(io, "  abscence_limit: $(rt.abscence_limit)")
end
