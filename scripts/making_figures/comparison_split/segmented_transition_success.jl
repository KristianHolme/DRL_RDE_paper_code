# Segmented headline: SSS/SSM/MSS/MSM transition times and success vs Δt.
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using Statistics
using DataFrames
using CairoMakie
CairoMakie.activate!()
##
include(scriptsdir("transition_dataset_specs.jl"))
include(joinpath(@__DIR__, "plot_knobs.jl"))
figsize = (16cm, 20cm)
datasets = [
    (experiment_name, f, nick, c) for
        ((experiment_name, f, nick), c) in zip(
            TRANSITION_SEGMENTED_DATASETS,
            [color_sss, color_ssm, color_mss, color_msm],
        )
]
dts = 2 .^ (-2.0f0:3.0f0)
##
matrices = Vector{Matrix{Float64}}()
labels = String[]
colors = Any[]
foreach(datasets) do (experiment_name, _, nickname, color)
    _, tm = load_matrices(experiment_name, nickname; mode = :best_transition)
    push!(matrices, tm)
    push!(labels, nickname)
    push!(colors, color)
end
##
success_df = DataFrame(success_rate = Float32[], dt = Float32[], nickname = String[])
foreach(datasets) do (experiment_name, filter_function, nickname, _)
    df = collect_results!(datadir("studies", experiment_name, "results"))
    df = df[.!occursin.(r"_#\d+\.jld2$", df.path), :]
    for dt in dts
        df_dt = filter(r -> filter_function(r) && r.dt ≈ dt, df)
        isempty(df_dt) && continue
        mean_success_rates = map(eachrow(df_dt)) do row
            success_matrix = row.transition_success_matrix
            target = row.target_shock_count
            starts = setdiff(1:4, target)
            return mean(success_matrix[starts, target])
        end
        push!(success_df, (mean(mean_success_rates), dt, nickname))
    end
end
##
fig = with_theme(comparison_theme()) do
    fig = Figure(; size = figsize, backgroundcolor = bgcolor)
    axes = []
    elements = []
    failure_scatters = []
    gl_time = fig[1:2, 1:2] = GridLayout()
    n_datasets = length(matrices)
    total_width = 0.8
    gap = 0.3
    bar_width = total_width / n_datasets
    for (idx, target) in enumerate(1:4)
        start_shocks = setdiff(1:4, target)
        ax = Axis(
            gl_time[div((idx - 1), 2) + 1, mod(idx - 1, 2) + 1];
            title = "Target wave count: $target",
            xlabel = idx == 3 || idx == 4 ? "Initial wave count" : "",
            ylabel = idx == 1 || idx == 3 ? metric_label("Transition time"; better = lower_is_better) : "",
            xticks = (1:3, string.(start_shocks)),
        )
        idx % 2 == 0 && hideydecorations!(ax; ticks = false, grid = false)
        push!(axes, ax)
        for (k, matrix) in enumerate(matrices)
            positions = collect(1:3) .- total_width ./ 2 .+ (k - 1 / 2) .* bar_width
            heights = [matrix[start, target] for start in start_shocks]
            plot_heights = similar(heights)
            for i in eachindex(heights)
                h = heights[i]
                plot_heights[i] = if !isfinite(h)
                    bar_ymax  # failed → full bar ("infinite" transition time)
                else
                    min(h, bar_ymax)
                end
            end
            bp = barplot!(ax, positions, plot_heights; color = colors[k], width = bar_width, gap = gap)
            for (pos, h) in zip(positions, heights)
                isfinite(h) && h > bar_ymax || continue
                text!(
                    ax, pos, 0.92 * bar_ymax;
                    text = string(round(Int, h)),
                    align = (:center, :center),
                    fontsize = clip_label_fontsize,
                    color = :white,
                    rotation = π / 2,
                )
            end
            failed_indices = findall(isnan, heights)
            if !isempty(failed_indices)
                s = scatter!(
                    ax,
                    positions[failed_indices],
                    fill(0.98 * bar_ymax, length(failed_indices));
                    marker = :x,
                    color = :red,
                    markersize = 16,
                )
                push!(failure_scatters, s)
            end
            idx == 1 && push!(elements, bp)
        end
        ylims!(ax, 0, bar_ymax)
    end
    linkyaxes!(axes...)
    Label(
        fig[1:2, 1:2, BottomLeft()],
        panel_letter_a;
        fontsize = 24,
        padding = (0, 0, 0, 0),
        tellwidth = false,
        tellheight = false,
        halign = :left,
        valign = :bottom,
    )
    box_time = Box(
        fig[1:2, 1:2, Makie.GridLayoutBase.Outer()];
        alignmode = Outside(-5, -12, -8, -5),
        cornerradius = 4,
        color = (:darkgreen, 0.08),
        strokecolor = (:darkgreen, 0.6),
        strokewidth = 2,
        visible = true,
    )
    Makie.translate!(box_time.blockscene, 0, 0, -200)

    gl_success = fig[3, 1:2] = GridLayout()
    n_success = length(datasets)
    total_width_s = 0.7
    gap_s = 0.3
    bar_width_s = total_width_s / n_success
    ax_success = Axis(
        gl_success[1, 1:2];
        xlabel = L"Action duration $\Delta t$",
        ylabel = metric_label("Mean success rate (%)"; better = higher_is_better),
        xticks = (1:6, string.(dts)),
    )
    for (k, (_, _, nickname, color)) in enumerate(datasets)
        positions = collect(1:6) .- total_width_s ./ 2 .+ (k - 1 / 2) .* bar_width_s
        heights = [
            success_df[success_df.nickname .== nickname .&& success_df.dt .≈ dt, :].success_rate[1]
                for dt in dts
        ]
        barplot!(ax_success, positions, heights; color = color, width = bar_width_s, gap = gap_s)
        failed_indices = findall(h -> h ≈ 0, heights)
        if !isempty(failed_indices)
            successful_heights = filter(h -> !(h ≈ 0), heights)
            y_offset = isempty(successful_heights) ? 0.02f0 : max(0.02f0, maximum(successful_heights) * 0.01f0)
            scatter!(
                ax_success,
                positions[failed_indices],
                fill(y_offset, length(failed_indices));
                marker = :x,
                color = :red,
                markersize = 16,
            )
        end
    end
    Label(
        fig[3, 1:2, BottomLeft()],
        panel_letter_b;
        fontsize = 24,
        padding = (0, 0, 0, 0),
        tellwidth = false,
        tellheight = false,
        halign = :left,
        valign = :bottom,
    )
    box_success = Box(
        fig[3, 1:2, Makie.GridLayoutBase.Outer()];
        alignmode = Outside(-5, -12, -8, -5),
        cornerradius = 4,
        color = (:royalblue, 0.08),
        strokecolor = (:royalblue, 0.6),
        strokewidth = 2,
        visible = true,
    )
    Makie.translate!(box_success.blockscene, 0, 0, -200)
    legend_elements = isempty(failure_scatters) ? elements : vcat(elements, failure_scatters[1])
    legend_labels = isempty(failure_scatters) ? labels : vcat(labels, "Failure")
    Legend(fig[4, 1:2], legend_elements, legend_labels; orientation = :horizontal, nbanks = 1)
    return fig
end
##
out_dir = mkpath(plotsdir("comparison_split"))
dpi = 300
foreach((".png", ".pdf")) do suffix
    wsave(joinpath(out_dir, "segmented_transition_success" * suffix), fig; px_per_unit = dpi / inch)
end
println("saved segmented_transition_success to ", out_dir)
