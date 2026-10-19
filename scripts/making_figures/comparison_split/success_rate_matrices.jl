# Success-rate matrices for segmented controllers only (SSS/SSM/MSS/MSM).
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using CairoMakie
CairoMakie.activate!()
using DataFrames
using Statistics

include(scriptsdir("transition_dataset_specs.jl"))
include(joinpath(@__DIR__, "plot_knobs.jl"))

figsize = (16cm, 12cm)
##
# utilities
function mean_success_matrix(df::AbstractDataFrame)
    mat = fill(NaN, 4, 4)
    for target in 1:4
        rows = df[df.target_shock_count .== target, :]
        if isempty(rows)
            continue
        end
        for start in setdiff(1:4, target)
            vals = map(rows.transition_success_matrix) do sm
                return Float64(sm[start, target])
            end
            mat[start, target] = mean(vals)
        end
    end
    return mat
end

function mean_time_matrix(df::AbstractDataFrame)
    mat = fill(NaN, 4, 4)
    for target in 1:4
        rows = df[df.target_shock_count .== target, :]
        if isempty(rows)
            continue
        end
        for start in setdiff(1:4, target)
            vals = Float64[]
            for tm in rows.transition_time_matrix
                v = Float64(tm[start, target])
                isfinite(v) && v > 0 && push!(vals, v)
            end
            mat[start, target] = isempty(vals) ? NaN : mean(vals)
        end
    end
    return mat
end

function load_metric_matrices_by_dt(;
        datasets = TRANSITION_SEGMENTED_DATASETS,
        dts = 2 .^ (-2.0f0:3.0f0),
        metric = :success,
    )
    matrices = Dict{Tuple{String, Float32}, Matrix{Float64}}()
    for (experiment_name, filter_function, nickname) in datasets
        results_dir = datadir("studies", experiment_name, "results")
        if !isdir(results_dir)
            @warn "Results directory not found, skipping" experiment_name
            continue
        end

        df = collect_results!(results_dir)
        df = df[.!occursin.(r"_#\d+\.jld2$", df.path), :]
        df = filter(filter_function, df)
        if isempty(df)
            @warn "No results after filtering, skipping" experiment_name nickname
            continue
        end
        DRL_RDE.add_transition_matrices!(df, experiment_name)

        for dt in dts
            df_dt = filter(r -> r.dt ≈ dt, df)
            if isempty(df_dt)
                @warn "No rows for dt, skipping cell" nickname dt
                continue
            end
            matrices[(nickname, Float32(dt))] = if metric == :time
                mean_time_matrix(df_dt)
            else
                mean_success_matrix(df_dt)
            end
        end
    end
    return matrices
end

function plot_success_heatmap!(
        ax::Axis,
        mat::AbstractMatrix;
        colormap = :viridis,
        colorrange = (0, 100),
        annotate = true,
        annotationsize = heatmap_annotationsize,
        nan_color = heatmap_nan_color,
        fmt = v -> string(round(Int, v)),
        reversed = false,
    )
    hm = heatmap!(ax, 1:4, 1:4, mat'; colormap, colorrange, nan_color)
    if annotate
        lo, hi = colorrange
        thresh = lo + 0.55 * (hi - lo)
        for target in 1:4, start in 1:4
            val = mat[start, target]
            if isnan(val)
                continue
            end
            dark = reversed ? val > thresh : val < thresh
            text!(
                ax,
                target,
                start;
                text = fmt(val),
                align = (:center, :center),
                fontsize = annotationsize,
                color = dark ? :white : :black,
            )
        end
    end
    return hm
end

function plot_success_rate_matrices(
        matrices::Dict;
        nicknames = [nick for (_, _, nick) in TRANSITION_SEGMENTED_DATASETS],
        dts = 2 .^ (-2.0f0:3.0f0),
        figsize = (16cm, 16cm),
        colormap = :viridis,
        colorrange = (0, 100),
        colorbar_label = "Success rate (%)",
        fmt = v -> string(round(Int, v)),
        reversed = false,
    )
    fig = Figure(; size = figsize, backgroundcolor = bgcolor)
    gl = fig[1, 1] = GridLayout()
    hm = nothing
    n_nick = length(nicknames)

    for (i, nickname) in enumerate(nicknames)
        for (j, dt) in enumerate(dts)
            key = (nickname, Float32(dt))
            ax = Axis(
                gl[i, j];
                aspect = 1,
                xticks = 1:4,
                yticks = 1:4,
                title = i == 1 ? "Δt = $dt" : "",
                xlabel = i == n_nick ? "Target" : "",
                ylabel = j == 1 ? "Initial" : "",
            )
            if haskey(matrices, key)
                hm = plot_success_heatmap!(
                    ax, matrices[key];
                    colormap, colorrange, fmt, reversed,
                )
            else
                text!(ax, 2.5, 2.5; text = "n/a", align = (:center, :center))
            end

            if j > 1
                hideydecorations!(ax; grid = false)
            end
            if i < n_nick
                hidexdecorations!(ax; grid = false)
            end
        end
        Label(
            gl[i, 0],
            nickname;
            rotation = π / 2,
            tellheight = false,
            font = :bold,
        )
    end

    if !isnothing(hm)
        Colorbar(fig[1, 2], hm; label = colorbar_label, height = Relative(0.6))
    end
    colgap!(gl, 4)
    rowgap!(gl, 4)
    return fig
end
##
# setup: load mean success matrices per (nickname, Δt)
dts = 2 .^ (-2.0f0:3.0f0)
nicknames = [nick for (_, _, nick) in TRANSITION_SEGMENTED_DATASETS]
success_matrices = load_metric_matrices_by_dt(; dts, metric = :success)
time_matrices = load_metric_matrices_by_dt(; dts, metric = :time)
##
fig = with_theme(comparison_theme()) do
    return plot_success_rate_matrices(
        success_matrices;
        nicknames, dts, figsize,
        colormap = success_colormap,
        colorrange = (0, 100),
        colorbar_label = colorbar_metric_label("Success rate (%)"; better = higher_is_better),
        fmt = v -> string(round(Int, v)),
    )
end
time_max = maximum(
    M -> maximum(x -> isfinite(x) ? x : 0.0, M),
    values(time_matrices);
    init = 0.0,
)
fig_time = with_theme(comparison_theme()) do
    return plot_success_rate_matrices(
        time_matrices;
        nicknames, dts, figsize,
        colormap = time_colormap,
        colorrange = (0, time_max),
        colorbar_label = colorbar_metric_label("Transition time"; better = lower_is_better),
        fmt = v -> string(round(Int, v)),
        reversed = true,
    )
end
##
out_dir = mkpath(plotsdir("comparison_split"))
foreach((".png", ".pdf")) do suffix
    wsave(joinpath(out_dir, "success_rate_matrices" * suffix), fig; px_per_unit = dpi / inch)
    wsave(joinpath(out_dir, "transition_time_matrices" * suffix), fig_time; px_per_unit = dpi / inch)
end
println("saved success and transition-time matrices to ", out_dir)
