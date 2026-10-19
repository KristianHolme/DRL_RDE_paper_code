using DrWatson
@quickactivate :DRL_RDE
using Statistics
using DataFrames
using CairoMakie
CairoMakie.activate!()

shock_u_ps = RDE.SHOCK_PRESSURES

# Collect results
res = collect_results!(datadir("two_step_transition_time"), black_list = ["transition_success_matrix", "transition_time_matrix"])
# dropmissing!(res)

transition_plot_dir = plotsdir("transition_plots")
if !isdir(transition_plot_dir)
    mkpath(transition_plot_dir)
end

print_transition_time_summary(res)

success_matrix, time_matrix = transition_overview_matrices(res)
fig = Figure(size = (1200, 800))

ax1 = Axis(
    fig[1, 1],
    title = "Transition Success Rate",
    xlabel = "Target Shock Count",
    ylabel = "Start Shock Count"
)
hm1 = heatmap!(
    ax1, 1:4, 1:4, success_matrix',
    colormap = :viridis,
    colorrange = (0, 100)
)
Colorbar(fig[1, 2], hm1, label = "Success Rate (%)")

ax2 = Axis(
    fig[2, 1],
    title = "Mean Transition Time",
    xlabel = "Target Shock Count",
    ylabel = "Start Shock Count"
)
hm2 = heatmap!(
    ax2, 1:4, 1:4, time_matrix',
    colormap = :viridis,
    nan_color = :black
)
Colorbar(fig[2, 2], hm2, label = "Mean Transition Time")

safesave(
    datadir("two_step_transition_time", "transition_time_matrices.jld2"),
    Dict(
        "transition_success_matrix" => success_matrix,
        "transition_time_matrix" => time_matrix,
    )
)

display(fig)
save(plotsdir("transition_analysis_overview.png"), fig)

plot_best_transitions(res, shock_u_ps, transition_plot_dir, show_plots = true, parallel = true)

println("\nPer-target aggregated selection (step baseline):")
println("=============================================")

best_configs = get_best_configs_per_target(res)
time_matrix_per_target = transition_time_matrix_per_target(res, best_configs)

per_target_dir = datadir("two_step_transition_time_per_target")
if !isdir(per_target_dir)
    mkpath(per_target_dir)
end
safesave(
    joinpath(per_target_dir, "transition_time_matrices.jld2"),
    Dict("transition_time_matrix" => time_matrix_per_target),
)
@info "Saved per-target time matrix to $per_target_dir"
