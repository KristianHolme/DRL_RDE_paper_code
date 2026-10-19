# Uniform headline: two-step vs PID vs selected SU (fair clock), time heatmaps only.
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using CairoMakie
CairoMakie.activate!()
using JLD2
using Statistics
include(joinpath(@__DIR__, "plot_knobs.jl"))

figsize = (16cm, 5.5cm)
##
function mask_diag(M)
    out = Matrix{Float64}(M)
    for i in 1:4
        out[i, i] = NaN
    end
    return out
end

function plot_annotated_heatmap!(
        ax::Axis,
        mat;
        colormap,
        colorrange,
        nan_color = heatmap_nan_color,
        fmt = v -> string(round(v; digits = 1)),
        reversed = false,
        annotation_size = 11,
    )
    hm = heatmap!(ax, 1:4, 1:4, mat'; colormap, colorrange, nan_color)
    lo, hi = colorrange
    thresh = lo + 0.55 * (hi - lo)
    for target in 1:4, start in 1:4
        val = mat[start, target]
        !isfinite(val) && continue
        dark = reversed ? val > thresh : val < thresh
        text!(
            ax, target, start;
            text = fmt(val),
            align = (:center, :center),
            fontsize = annotation_size,
            color = dark ? :white : :black,
        )
    end
    return hm
end

function plot_uniform_time_matrices(time_mats, labels; time_range)
    fig = Figure(; size = figsize, backgroundcolor = bgcolor)
    gl = fig[1, 1] = GridLayout()
    hm_t = nothing
    for (i, label) in enumerate(labels)
        ax_t = Axis(
            gl[1, i];
            aspect = 1,
            xticks = 1:4,
            yticks = 1:4,
            title = label,
            xlabel = "Target wave count",
            ylabel = i == 1 ? "Initial wave count" : "",
        )
        i > 1 && hideydecorations!(ax_t; grid = false)
        hm_t = plot_annotated_heatmap!(
            ax_t,
            time_mats[i];
            colormap = time_colormap,
            colorrange = time_range,
            reversed = true,
        )
    end
    Colorbar(
        gl[1, 4], hm_t;
        label = rich(
            "Transition time",
            "\n",
            rich(lower_is_better; fontsize = better_subtitle_fontsize),
        ),
    )
    colgap!(gl, 8)
    return fig
end
##
fair_path = datadir("studies", "ar14_all_target", "fair_clock", "uniform_fair_clock.jld2")
fair = load(fair_path)
labels = ["Two-step", "PID", "SU"]
time_mats = mask_diag.([fair["two_step_time"], fair["pid_time"], fair["sus_time"]])
@info "SU seeds" fair["sus_seeds"]
@info "Aggregate" fair["summary"]
time_min = minimum(M -> minimum(x -> isfinite(x) ? x : Inf, M), time_mats)
time_max = maximum(M -> maximum(x -> isfinite(x) ? x : 0.0, M), time_mats)
##
fig = with_theme(comparison_theme()) do
    return plot_uniform_time_matrices(time_mats, labels; time_range = (time_min, time_max))
end
##
out_dir = mkpath(plotsdir("comparison_split"))
foreach((".png", ".pdf")) do suffix
    wsave(joinpath(out_dir, "uniform_vs_baselines" * suffix), fig; px_per_unit = dpi / inch)
end
println("saved uniform_vs_baselines to ", out_dir)
