# Shared visual knobs for comparison_split figures. Tweak and rerender.
# Final bar LinePattern hatching needs Makie 0.25 / the local fork.

# Background: transparent helps sepia/Zotero; black spines vanish on dark paper.
bgcolor = :transparent

# Continuous (heatmaps, Δt curves). Try :viridis, :lipari, :batlow, :plasma, :devon.
success_colormap = :viridis
# Reversed so low transition time uses the same "good" color as high success.
time_colormap = Reverse(success_colormap)
continuous_colormap = success_colormap  # Δt training curves, etc.

# Categorical (method colors). Wong is colorblind-safe.
# Swap the two blocks below to sample a continuous cmap into N colors, e.g. batlow:
n_categorical = 6
categorical_source = :viridis
# categorical_colors = Makie.resample_cmap(categorical_source, n_categorical)
categorical_colors = Makie.wong_colors()

hatch(color) = Makie.LinePattern(; backgroundcolor = color)

color_two_step = :grey
color_pid = :black
color_sus = categorical_colors[1]
color_sss = categorical_colors[2]
color_ssm = hatch(color_sss)
color_mss = categorical_colors[1]
color_msm = hatch(color_mss)

# Config-facet pair colors (stationary/single vs moving/multi).
color_facet_a = categorical_colors[1]
color_facet_b = categorical_colors[2]
# Median on Wong blue/yellow fills. Gray was too quiet; avoid white.
# Alternatives: :black, :gray10, categorical_colors[4] (Wong purple), :teal
mediancolor = :dimgray

panel_letter_a = "(a)"
panel_letter_b = "(b)"

higher_is_better = "(higher is better)"
lower_is_better = "(lower is better)"
better_subtitle_fontsize = 8pt

bar_ymax = 250.0
clip_label_fontsize = 9
heatmap_annotationsize = 8
heatmap_nan_color = :gray90

# Fourier magnitude panel: max bar height; brackets sit at 1.0.
fourier_bar_headroom = 0.85

# Observation diagram traces.
lambdacolor_main = :red
lambdacolor_sub = :lightsalmon
ucolor_main = :blue
ucolor_sub = :lightblue
wave_color = :palegreen3

dpi = 300

function comparison_theme()
    return merge(
        theme_drl_rde(),
        Theme(
            backgroundcolor = bgcolor,
            Axis = (backgroundcolor = bgcolor,),
            Legend = (backgroundcolor = bgcolor, framevisible = true),
            Colorbar = (backgroundcolor = bgcolor,),
        ),
    )
end

function metric_label(name; better)
    return rich(name, "\n", rich(better; fontsize = better_subtitle_fontsize))
end

function colorbar_metric_label(name; better)
    return rich(name, " ", rich(better; fontsize = better_subtitle_fontsize))
end
