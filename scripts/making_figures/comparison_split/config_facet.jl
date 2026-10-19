# Facet comparison: rainclouds for success rate and transition time.
# Segmented only: 2 rows (success, time) x 2 columns (frame, agent).
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using CairoMakie
CairoMakie.activate!()
using DataFrames
using Makie: hist, boxplot!, scatter!
using Statistics
using Random

include(scriptsdir("transition_dataset_specs.jl"))
include(joinpath(@__DIR__, "plot_knobs.jl"))

# Median/mean line inside boxplots. White vanishes on light fills / sepia.
figsize = (14cm, 14cm)
color_a = color_facet_a
color_b = color_facet_b

##
# utilities

function config_tags(nickname::AbstractString)
    return (
        frame = nickname[3] == 'S' ? :stationary : :moving,
        agent = nickname[1] == 'S' ? :single : :multi,
        control = nickname[2] == 'U' ? :uniform : :segmented,
    )
end

function load_transition_points()
    points = DataFrame(
        nickname = String[],
        experiment_name = String[],
        dt = Float32[],
        target_shock_count = Int[],
        start_shock_count = Int[],
        success = Float64[],
        transition_time = Union{Missing, Float64}[],
        frame = Symbol[],
        agent = Symbol[],
        control = Symbol[],
    )

    for (experiment_name, filter_function, nickname) in TRANSITION_SEGMENTED_DATASETS
        results_dir = datadir("studies", experiment_name, "results")
        if !isdir(results_dir)
            @warn "Results directory not found, skipping" experiment_name
            continue
        end

        df = collect_results!(results_dir)
        df = df[.!occursin.(r"_#\d+\.jld2$", df.path), :]
        df = filter(filter_function, df)
        if nrow(df) == 0
            @warn "No results after filtering, skipping" experiment_name nickname
            continue
        end

        DRL_RDE.add_transition_matrices!(df, experiment_name)
        tags = config_tags(nickname)

        for row in eachrow(df)
            success_matrix = row.transition_success_matrix
            time_matrix = row.transition_time_matrix
            target = Int(row.target_shock_count)

            for start in setdiff(1:4, target)
                success_val = Float64(success_matrix[start, target])
                time_val = time_matrix[start, target]
                transition_time = isnan(time_val) ? missing : Float64(time_val)

                push!(
                    points,
                    (
                        nickname,
                        experiment_name,
                        Float32(row.dt),
                        target,
                        start,
                        success_val,
                        transition_time,
                        tags.frame,
                        tags.agent,
                        tags.control,
                    ),
                )
            end
        end
    end

    if nrow(points) == 0
        error(
            "No transition points loaded. Ensure datadir() is populated and " *
                "transition_analyses exist for all result rows.",
        )
    end

    return points
end

function facet_data(
        points::DataFrame,
        spec,
        value_col::Symbol;
        value_filter = row -> true,
        cat_a = 1.0,
        cat_b = 2.0,
        color_a = color_facet_a,
        color_b = color_facet_b,
    )
    positions = Float64[]
    values = Float64[]
    colors = Any[]

    for (slot, level, color) in (
            (cat_a, spec.level_a, color_a),
            (cat_b, spec.level_b, color_b),
        )
        subset = points[points[!, spec.group_col] .== level, :]
        for row in eachrow(subset)
            if !value_filter(row)
                continue
            end
            val = row[value_col]
            if val isa Missing || (val isa Real && isnan(val))
                continue
            end
            push!(positions, slot)
            push!(values, Float64(val))
            push!(colors, color)
        end
    end

    return positions, values, colors
end

function plot_facet_boxplot!(
        ax::Axis,
        points::DataFrame,
        spec,
        value_col::Symbol;
        cat_a = 1.0,
        cat_b = 2.0,
        color_a = color_facet_a,
        color_b = color_facet_b,
        box_width = 0.35,
        mediancolor = :white,
        medianlinewidth = 2.5,
        x_lims = nothing,
    )
    positions, values, _ = facet_data(
        points, spec, value_col;
        cat_a, cat_b, color_a, color_b,
    )

    if isempty(values)
        @warn "No data for boxplot" value_col spec.title
        return ax
    end

    for slot in (cat_a, cat_b)
        mask = positions .== slot
        if !any(mask)
            continue
        end
        color = slot == cat_a ? color_a : color_b
        boxplot!(
            ax,
            fill(slot, sum(mask)),
            values[mask];
            width = box_width,
            color = RGBAf(color, 1.0),
            strokecolor = :black,
            strokewidth = 1.5,
            mediancolor,
            medianlinewidth,
            show_outliers = false,
        )
    end

    ax.xticks = ([cat_a, cat_b], [spec.labels...])
    if !isnothing(x_lims)
        ax.limits = (x_lims[1], x_lims[2], nothing, nothing)
    end

    return ax
end

rand_localized(min, max) = min + (max - min) * rand()
rand_localized(min, max, n::Int) = [rand_localized(min, max) for _ in 1:n]

function plot_facet_raincloud!(
        ax::Axis,
        points::DataFrame,
        spec,
        value_col::Symbol;
        value_filter = row -> true,
        cat_a = 1.0,
        cat_b = 2.0,
        color_a = color_facet_a,
        color_b = color_facet_b,
        hist_width = 0.45,
        boxplot_w = 0.12,
        boxplot_offset = 0.15,
        scatter_offset = 0.35,
        scatter_width = 0.15,
        hist_bins = 12,
        scatter_markersize = 5.0,
        mediancolor = :white,
        medianlinewidth = 2.5,
        x_lims = nothing,
    )
    positions, values, _ = facet_data(
        points, spec, value_col;
        value_filter, cat_a, cat_b, color_a, color_b,
    )

    if isempty(values)
        @warn "No data for raincloud" value_col spec.title
        return ax
    end

    for slot in (cat_a, cat_b)
        mask = positions .== slot
        if !any(mask)
            continue
        end
        vals = values[mask]
        color = slot == cat_a ? color_a : color_b

        hist!(
            ax,
            vals;
            offset = slot,
            scale_to = -hist_width,
            direction = :x,
            bins = hist_bins,
            color = RGBAf(color, 1.0),
            strokecolor = color,
            strokewidth = 0.5,
        )

        boxplot!(
            ax,
            fill(slot + boxplot_offset, length(vals)),
            vals;
            width = boxplot_w,
            color = RGBAf(color, 1.0),
            strokecolor = :black,
            strokewidth = 1.5,
            mediancolor,
            medianlinewidth,
            show_outliers = false,
        )

        jitter = rand_localized.(
            scatter_offset,
            scatter_offset + scatter_width,
            length(vals),
        )
        scatter!(
            ax,
            slot .+ jitter,
            vals;
            markersize = scatter_markersize,
            color = color,
            strokecolor = :black,
            strokewidth = 0.0,
        )
    end

    ax.xticks = ([cat_a, cat_b], [spec.labels...])
    if !isnothing(x_lims)
        ax.limits = (x_lims[1], x_lims[2], nothing, nothing)
    end

    return ax
end

function config_facet_rainclouds_figure(
        points::DataFrame;
        figsize = (14cm, 14cm),
        facet_specs = [
            (
                group_col = :frame,
                level_a = :stationary,
                level_b = :moving,
                title = "Reference frame",
                labels = ("Stationary", "Moving"),
            ),
            (
                group_col = :agent,
                level_a = :single,
                level_b = :multi,
                title = "Agent type",
                labels = ("Single", "Multi"),
            ),
        ],
        color_a = color_facet_a,
        color_b = color_facet_b,
        cat_a = 1.0,
        cat_b = 3.2,
        hist_width = 0.45,
        boxplot_w = 0.12,
        boxplot_offset = 0.15,
        scatter_offset = 0.35,
        scatter_width = 0.15,
        hist_bins = 12,
        scatter_markersize = 5.0,
        mediancolor = :white,
        medianlinewidth = 2.5,
        x_pad = 0.1,
    )
    x_lims = (cat_a - hist_width - x_pad, cat_b + scatter_offset + scatter_width + x_pad)

    return with_theme(comparison_theme()) do
        fig = Figure(; size = figsize, backgroundcolor = bgcolor)

        success_axes = Vector{Axis}(undef, length(facet_specs))
        time_axes = Vector{Axis}(undef, length(facet_specs))

        for (col, spec) in enumerate(facet_specs)
            ax_s = Axis(
                fig[1, col];
                title = spec.title,
                ylabel = metric_label("Success rate (%)"; better = higher_is_better),
                limits = (x_lims[1], x_lims[2], 0, 100),
            )
            plot_facet_raincloud!(
                ax_s, points, spec, :success;
                cat_a, cat_b, color_a, color_b,
                hist_width, boxplot_w, boxplot_offset,
                scatter_offset, scatter_width, hist_bins,
                scatter_markersize, mediancolor, medianlinewidth, x_lims,
            )
            success_axes[col] = ax_s

            ax_t = Axis(
                fig[2, col];
                ylabel = metric_label("Transition time"; better = lower_is_better),
                limits = (x_lims[1], x_lims[2], 0, nothing),
            )
            plot_facet_raincloud!(
                ax_t, points, spec, :transition_time;
                value_filter = row -> !ismissing(row.transition_time),
                cat_a, cat_b, color_a, color_b,
                hist_width, boxplot_w, boxplot_offset,
                scatter_offset, scatter_width, hist_bins,
                scatter_markersize, mediancolor, medianlinewidth, x_lims,
            )
            time_axes[col] = ax_t
        end

        linkyaxes!(success_axes...)
        linkyaxes!(time_axes...)

        for col in 2:length(facet_specs)
            hidexdecorations!(success_axes[col]; grid = false, ticks = false)
            hideydecorations!(success_axes[col]; grid = false, label = true)
            hideydecorations!(time_axes[col]; grid = false, label = true)
        end
        hidexdecorations!(success_axes[1]; grid = false, ticks = false)

        return fig
    end
end

##
points = load_transition_points()
@info "Loaded transition points" nrow(points) configs = unique(points.nickname)
Random.seed!(2026)

##
fig = config_facet_rainclouds_figure(
    points;
    figsize,
    color_a,
    color_b,
    cat_a = 1.0,
    cat_b = 2.2,
    hist_width = 0.45,
    boxplot_w = 0.12,
    boxplot_offset = 0.15,
    scatter_offset = 0.35,
    scatter_width = 0.15,
    hist_bins = 12,
    scatter_markersize = 5.0,
    mediancolor,
    medianlinewidth = 2.5,
)
##
dpi = 300
out_dir = mkpath(plotsdir("comparison_split"))
foreach((".png", ".pdf")) do suffix
    out = joinpath(out_dir, "config_facet" * suffix)
    wsave(out, fig; px_per_unit = dpi / inch)
    @info "Saved figure" out
end
