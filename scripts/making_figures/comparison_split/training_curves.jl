# Segmented training curves: 2×2 stationary (top) / moving (bottom).
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using CairoMakie
CairoMakie.activate!()

include(scriptsdir("transition_dataset_specs.jl"))
include(joinpath(@__DIR__, "plot_knobs.jl"))

figsize = (16cm, 10cm)
reward_ylabel = metric_label("Episodic reward"; better = higher_is_better)
##
# utilities
# Top row: stationary (SSS, MSS); bottom row: moving (SSM, MSM)
const FRAME_ORDERED_DATASETS = [
    ("SAS3", r -> r.movingframe == false, "SSS"),
    ("MA3", r -> r.movingframe == false, "MSS"),
    ("SAS3", r -> r.movingframe == true, "SSM"),
    ("MA3", r -> r.movingframe == true, "MSM"),
]

function get_filtered_results(experiment_name, filterfn)
    results_dir = datadir("studies", experiment_name, "results")
    full_res = collect_results!(results_dir)
    full_res = full_res[.!occursin.(r"_#\d+\.jld2$", full_res.path), :]
    full_res = filter(filterfn, full_res)
    return full_res
end

function plot_histories_by_frame(
        aggregated_list,
        key::AbstractString;
        labels,
        ylabel,
        plot_band = true,
        figsize = (16cm, 10cm),
        xtick_vals = [0.0, 4.0e5, 8.0e5],
        xtick_labels = ["0", "4×10⁵", "8×10⁵"],
        colormap = :viridis,
        yscale = identity,
    )
    fig = Figure(; size = figsize, backgroundcolor = bgcolor)
    gl = fig[1, 1] = GridLayout()
    axes = Axis[]

    all_dts = sort(
        unique(
            Float64(config.dt)
                for aggregated in aggregated_list
                for config in keys(aggregated)
                if haskey(config, :dt)
        ),
    )
    n_dts = length(all_dts)
    dt_colors = n_dts == 0 ? Any[] : Makie.resample_cmap(colormap, n_dts)
    colors_by_dt = Dict(dt => dt_colors[i] for (i, dt) in enumerate(all_dts))

    for (idx, aggregated) in enumerate(aggregated_list)
        row = div(idx - 1, 2) + 1
        col = mod1(idx, 2)
        ax = Axis(
            gl[row, col];
            title = labels[idx],
            xticks = (xtick_vals, xtick_labels),
            xlabel = row == 2 ? "Timesteps" : "",
            ylabel = col == 1 ? ylabel : "",
            yscale = yscale,
        )
        plot_aggregated_histories!(
            ax,
            aggregated,
            key;
            plot_band = plot_band,
            color_by = :dt,
            colors = colors_by_dt,
            marker_by = nothing,
            show_legend = false,
            return_legend = false,
            linewidth = 2,
        )

        col > 1 && hideydecorations!(ax; ticks = false, grid = false)
        row < 2 && hidexdecorations!(ax; ticks = false, grid = false)
        push!(axes, ax)
    end

    linkyaxes!(axes...)
    if n_dts > 0
        # Invisible proxy so Colorbar can track Makie.Categorical (experimental API).
        # Color values must be the Δt categories themselves (not 1:n), or ticks show indices.
        pl_cbar = scatter!(
            axes[1],
            fill(NaN, n_dts),
            fill(NaN, n_dts);
            color = all_dts,
            colormap = Makie.Categorical(colormap),
            visible = false,
        )
        Colorbar(
            fig[1, 2],
            pl_cbar;
            label = "Δt",
            tellheight = false,
        )
    end
    Label(gl[1, 0], "Stationary"; rotation = π / 2, tellheight = false, font = :bold)
    Label(gl[2, 0], "Moving"; rotation = π / 2, tellheight = false, font = :bold)
    colgap!(gl, 6)
    rowgap!(gl, 6)
    return fig
end
##
# setup: aggregate eval-reward histories over seeds and targets (group by Δt only)
reward_key = "eval/ep_rew_mean"
nicknames = [nick for (_, _, nick) in FRAME_ORDERED_DATASETS]

reward_aggregated = map(FRAME_ORDERED_DATASETS) do (experiment_name, filterfn, _)
    return aggregate_histories_by_seed(
        get_filtered_results(experiment_name, filterfn),
        reward_key;
        group_cols = ["dt"],
        logs_dir = datadir("studies", experiment_name, "logs"),
    )
end
##
# plot
fig = with_theme(comparison_theme()) do
    return plot_histories_by_frame(
        reward_aggregated,
        reward_key;
        labels = nicknames,
        ylabel = reward_ylabel,
        plot_band = true,
        figsize,
        colormap = continuous_colormap,
    )
end
##
# save
dpi = 300
out_dir = mkpath(plotsdir("comparison_split"))
foreach((".png", ".pdf")) do suffix
    wsave(
        joinpath(out_dir, "training_curves_reward" * suffix),
        fig;
        px_per_unit = dpi / inch,
    )
end
println("saved training_curves_reward to ", out_dir)
