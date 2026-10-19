using DrWatson
@quickactivate :DRL_RDE
using DataFrames
using Makie
##
using CairoMakie
CairoMakie.activate!()
##
# using GLMakie
# GLMakie.activate!()
##
# Load ramp results
name = ""
filename = isempty(name) ? "ramp_results.jld2" : "ramp_results_$(name).jld2"
results_path = joinpath(datadir("speed_analysis_ramp"), filename)
df_raw = load(results_path, "df")

##
# Drop rows with missing speed; filter shock speed range for plotting.
df = filter(r -> r.shock_speed < 2.0f0 && r.shock_speed > 1.4f0, df_raw)
# df = df_raw
df_plot = df[.!ismissing.(df.shock_speed), :]

"""Order rows like `build_cycle_ramp` in `speed_analysis_ramp_sweep.jl`: up → down → back."""
function sort_ramp_simulation_order(ramp_group::AbstractDataFrame, u_p_bottom::Float32)
    us = Float32(first(ramp_group.u_start))
    up_rows = sort(ramp_group[ramp_group.phase .== "up", :], :u_p)
    down_rows = sort(ramp_group[ramp_group.phase .== "down", :], :u_p; rev = true)
    back_sub = ramp_group[ramp_group.phase .== "back", :]
    back_rows = if us > u_p_bottom
        sort(back_sub, :u_p)
    elseif us < u_p_bottom
        sort(back_sub, :u_p; rev = true)
    else
        back_sub
    end
    return vcat(up_rows, down_rows, back_rows)
end

"""Ramp up / return vs ramp down."""
function phase_direction_color(phase::AbstractString, up_color, down_color)
    return phase == "down" ? down_color : up_color
end

"""Opaque RGB from phase; combined with `α` for GLMakie OIT (`transparency = true`)."""
function phase_direction_rgba(phase::AbstractString, α::Float32, up_color, down_color)
    tc = Makie.to_color(phase_direction_color(phase, up_color, down_color))
    return Makie.RGBAf(Float32(tc.r), Float32(tc.g), Float32(tc.b), α)
end

down_mask = df_plot.phase .== "down"
u_p_bottom_est = any(down_mask) ? Float32(minimum(df_plot[down_mask, :u_p])) : Float32(0.4)

## Phase-direction figure (circles)
up_color = :royalblue
down_color = :orangered3
# Brackets in data coordinates: `(x1, y1, x2, y2, label, orientation)` with `orientation` `:up` or `:down` (Makie `bracket!`).
phase_direction_brackets = [
    (0.39f0, 1.45f0, 0.58f0, 1.85f0, "1", :up),
    (0.59f0, 1.51f0, 0.8f0, 1.87f0, "2", :up),
    (0.79f0, 1.65f0, 0.95f0, 1.86f0, "3", :up),
    (0.95f0, 1.75f0, 1.03f0, 1.82f0, "4", :up),
    (0.9f0, 1.49f0, 1.03f0, 1.49f0, "unstable", :down),
]
# Legend `LineElement` uses fractional patch coordinates (0–1). Same geometry as Makie
# `bracket_bezierpath(::Val{:curly}, ...)` in `basic_recipes/bracket.jl`, sampled to a polyline.
function phase_direction_curly_bracket_legend_linepoints(;
        p1::Point2f = Point2f(0.1f0, 0.5f0),
        p2::Point2f = Point2f(0.9f0, 0.5f0),
        orientation::Symbol = :up,
        width::Float32 = 0.28f0,
        n_per_segment::Int = 18,
    )
    v = p2 - p1
    len = hypot(v[1], v[2])
    d1 = len > 0 ? Vec2f(v[1] / len, v[2] / len) : Vec2f(1.0f0, 0.0f0)
    d2 = Vec2f(-d1[2], d1[1])
    if (orientation === :up) != (d2[2] >= 0)
        d2 = -d2
    end
    d = d2
    p12 = Point2f(0.5f0 * (p1[1] + p2[1]), 0.5f0 * (p1[2] + p2[2])) + width * d
    c1 = p1 + width * d
    c2 = p12 - width * d
    c3 = p2 + width * d
    function cubic_samples(P0::Point2f, P1::Point2f, P2::Point2f, P3::Point2f, n::Int)
        pts = Vector{Point2f}(undef, n + 1)
        for i in 0:n
            t = Float32(i / n)
            t1 = 1.0f0 - t
            pts[i + 1] = Point2f(
                t1^3 * P0[1] + 3 * t1^2 * t * P1[1] + 3 * t1 * t^2 * P2[1] + t^3 * P3[1],
                t1^3 * P0[2] + 3 * t1^2 * t * P1[2] + 3 * t1 * t^2 * P2[2] + t^3 * P3[2],
            )
        end
        return pts
    end
    seg1 = cubic_samples(p1, c1, c2, p12, n_per_segment)
    seg2 = cubic_samples(p12, c2, c3, p2, n_per_segment)
    raw = [seg1; seg2[2:end]]
    return raw
end
# Vertically center the bracket in the legend patch: dy = 0.5 - mid(y_min, y_max) in fractional coords.
raw = phase_direction_curly_bracket_legend_linepoints()
ys = [p[2] for p in raw]
dy = 0.5f0 - (minimum(ys) + maximum(ys)) / 2.0f0
const phase_direction_bracket_legend_linepoints = [Point2f(p[1], p[2] + dy) for p in raw]
# Horizontal trace in legend patch (fractional 0–1) for line + scatter entries.
const phase_direction_legend_trace_linepoints = [Point2f(0.0f0, 0.5f0), Point2f(1.0f0, 0.5f0)]
# This figure is for Cairo (PDF/PNG); activate before building so `transparency` matches the backend.
CairoMakie.activate!()
fig_direction = with_theme(theme_drl_rde()) do
    fig = Figure(size = (16cm, 8cm))
    ax = Axis(fig[1, 1], xlabel = L"Injection pressure ($u_p$)", ylabel = "Wave speed")
    # Semi-transparent RGBA: GLMakie can use OIT (`transparency = true`); CairoMakie has no OIT
    # and should use `transparency = false` so Cairo alpha-composites colors normally.
    use_gl_oit = nameof(Makie.current_backend()) === :GLMakie
    phase_plot_alpha = 0.05f0
    # If false, lines use opaque colors; scatter still uses `phase_plot_alpha`.
    phase_direction_transparent_lines = true

    df_sorted = sort(df_plot, :u_start)
    for ramp_group in groupby(df_sorted, :u_start)
        sort_group = sort_ramp_simulation_order(ramp_group, u_p_bottom_est)
        isempty(sort_group) && continue
        for ph in ("up", "down", "back")
            seg = sort_group[sort_group.phase .== ph, :]
            isempty(seg) && continue
            line_alpha = phase_direction_transparent_lines ? phase_plot_alpha : 1.0f0
            rgba_line = phase_direction_rgba(ph, line_alpha, up_color, down_color)
            rgba_scatter = phase_direction_rgba(ph, phase_plot_alpha, up_color, down_color)
            lines!(
                ax, seg.u_p, seg.shock_speed;
                color = rgba_line,
                transparency = use_gl_oit && phase_direction_transparent_lines,
            )
            scatter!(
                ax, seg.u_p, seg.shock_speed;
                color = rgba_scatter,
                marker = :circle,
                markersize = 10,
                transparency = use_gl_oit,
            )
        end
    end

    if !isempty(phase_direction_brackets)
        ymax_data = Float32(maximum(df_plot.shock_speed))
        ymin_data = Float32(minimum(df_plot.shock_speed))
        y_span = max(ymax_data - ymin_data, 0.05f0)
        ymax_br = max(
            ymax_data,
            maximum(max(Float32(s[2]), Float32(s[4])) for s in phase_direction_brackets),
        )
        y_lim_top = ymax_br + 0.18f0 * y_span
        ylims!(ax, nothing, y_lim_top)
    end
    # Draw brackets last so they sit above semi-transparent lines/scatter.
    for (x1, y1, x2, y2, label, orientation) in phase_direction_brackets
        bracket!(
            ax,
            Point2f(x1, y1),
            Point2f(x2, y2);
            text = string(label),
            orientation = orientation,
            color = :black,
            linewidth = 1.2,
        )
    end

    # Legend swatches stay opaque even when the axis uses semi-transparent RGBA.
    legend_opaque_up = phase_direction_rgba("up", 1.0f0, up_color, down_color)
    legend_opaque_down = phase_direction_rgba("down", 1.0f0, up_color, down_color)

    Legend(
        fig[2, 1],
        [
            [
                LineElement(;
                    linepoints = phase_direction_legend_trace_linepoints,
                    linecolor = legend_opaque_up,
                    linewidth = 1.5f0,
                ),
                MarkerElement(; marker = :circle, color = legend_opaque_up, markersize = 10),
            ],
            [
                LineElement(;
                    linepoints = phase_direction_legend_trace_linepoints,
                    linecolor = legend_opaque_down,
                    linewidth = 1.5f0,
                ),
                MarkerElement(; marker = :circle, color = legend_opaque_down, markersize = 10),
            ],
            LineElement(;
                linepoints = phase_direction_bracket_legend_linepoints,
                linecolor = :black,
                linewidth = 1.5f0,
            ),
        ],
        [
            L"\text{Increasing } u_p",
            L"\text{Decreasing } u_p",
            L"\text{wave count}",
        ],
        orientation = :horizontal,
        halign = :center,
        tellwidth = false,
        tellheight = true,
    )
    rowgap!(fig.layout, 1, 8)
    return fig
end
##
dpi = 300
foreach([".pdf", ".png"]) do suffix
    wsave(plotsdir("figures", "speed_analysis_ramp_plot_direction$suffix"), fig_direction; px_per_unit = dpi / inch)
end
