# Render all manuscript figures (main set + revision comparison_split).
using DrWatson
@quickactivate :DRL_RDE

files = [
    "explaining_control.jl",
    "rde_diagram.jl",
    "sample_run_with_timescales.jl",
    "speed_analysis_ramp_plot.jl",
    "two-step-example.jl",
]
foreach(files) do file
    include(scriptsdir("making_figures", file))
end

include(scriptsdir("making_figures", "comparison_split", "generate_all.jl"))

@info "All figures generated successfully."
