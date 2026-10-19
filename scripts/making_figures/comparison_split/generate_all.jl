# Render comparison_split figures as PNG and PDF (no display).
using DrWatson
@quickactivate :DRL_RDE

segmented = [
    "observation_diagram.jl",
    "segmented_transition_success.jl",
    "config_facet.jl",
    "success_rate_matrices.jl",
    "training_curves.jl",
    "sss_ssm_3to4.jl",
]
for file in segmented
    @info "Rendering" file
    include(joinpath(@__DIR__, file))
end

fair_path = datadir("studies", "ar14_all_target", "fair_clock", "uniform_fair_clock.jld2")
if isfile(fair_path)
    @info "Rendering" "uniform_vs_baselines.jl"
    include(joinpath(@__DIR__, "uniform_vs_baselines.jl"))
    @info "Rendering" "uniform_3to2.jl"
    include(joinpath(@__DIR__, "uniform_3to2.jl"))
else
    @warn "Skipping uniform comparison figures; fair-clock file missing" fair_path
end
@info "comparison_split figures done"
