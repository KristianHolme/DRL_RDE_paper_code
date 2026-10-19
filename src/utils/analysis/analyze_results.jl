"""
    interactive_analysis(results::DataFrame; default_x="dt", default_y="eval/ep_rew_mean",
                         default_color="target_shock_count", filter_func=nothing)

Create an interactive dashboard for analyzing experiment results. Similar to TensorBoard,
this tool allows you to compare hyperparameters against evaluation metrics.

The dashboard includes:
- Parameter selection menus for x and y axes (with optional default selections)
- Optional color/group variable for multi-dimensional analysis
- Multiple plot types (scatter, boxplot, violin, histogram2d, density)
- Real-time statistical summaries
- Regression lines for scatter plots

# Arguments
- `results::DataFrame`: The DataFrame containing the results to analyze (not modified)
- `default_x="dt"`: Default x-axis variable (must be a column name)
- `default_y="eval/ep_rew_mean"`: Default y-axis variable (must be a column name)
- `default_color="target_shock_count"`: Default color/group variable (must be a column name or "none")
- `filter_func=nothing`: Optional filter function to apply to the DataFrame before analysis

# Features
- Handles missing values gracefully (filters per plot, doesn't modify input)
- Supports both numerical and categorical variables
- Automatic type detection and appropriate handling
- Numeric columns with ≤8 unique values are treated as categorical for coloring
- Clean re-rendering without text overlap

# Example
```julia
results = collect_results!("path/to/results")
interactive_analysis(results, default_x="learning_rate", default_y="mean_reward")
interactive_analysis(results, default_color="n_sections")
interactive_analysis(results, filter_func=r -> r.target_shock_count == 3)
```
"""
# NOTE: Makie.automatic and Makie.wong_colors are non-public; keep until public alternatives exist.
const MAKIE_AUTOMATIC = Makie.automatic
get_wong_colors() = Makie.wong_colors()

function interactive_analysis(results::DataFrame; default_x = "dt", default_y = "eval/ep_rew_mean", default_color = "target_shock_count", filter_func = nothing)
    # Work with a copy to avoid modifying the original DataFrame
    df = copy(results)

    if !isnothing(filter_func)
        df = filter(filter_func, df)
    end

    if nrow(df) == 0
        @error "DataFrame is empty. Cannot create interactive analysis."
        return nothing
    end

    # Identify numerical and categorical columns
    num_cols = String[]
    cat_cols = String[]
    bool_cols = String[]

    for col in names(df)
        col_type = nonmissingtype(eltype(df[!, col]))
        if col_type <: Number
            push!(num_cols, col)
        elseif col_type <: Bool
            push!(bool_cols, col)
        elseif col_type <: Union{AbstractString, Symbol, Char}
            push!(cat_cols, col)
        end
    end

    # All columns that can be used for axes (numeric, boolean, and categorical)
    axis_cols = vcat(num_cols, bool_cols, cat_cols)

    if isempty(axis_cols)
        @error "No usable columns found in DataFrame. Cannot create interactive analysis."
        return nothing
    end

    if isempty(num_cols)
        @warn "No numerical columns found in DataFrame. Y-axis will be limited to categorical plots."
    end

    # Validate and set defaults
    if !isnothing(default_x) && !(default_x in axis_cols)
        @warn "Specified default_x '$default_x' not found. Using first available column."
        default_x = axis_cols[1]
    end
    if !isnothing(default_y) && !(default_y in axis_cols)
        @warn "Specified default_y '$default_y' not found. Using second available column."
        default_y = length(axis_cols) > 1 ? axis_cols[2] : axis_cols[1]
    end

    default_x = isnothing(default_x) ? axis_cols[1] : default_x
    default_y = isnothing(default_y) ? (length(axis_cols) > 1 ? axis_cols[2] : axis_cols[1]) : default_y

    # Validate default_color (can be "none" or any axis column)
    color_options = vcat(["none"], axis_cols)
    if !isnothing(default_color) && !(default_color in color_options)
        @warn "Specified default_color '$default_color' not found. Using 'none'."
        default_color = "none"
    end
    default_color = isnothing(default_color) ? "none" : default_color

    # Scale options - using actual functions as values
    scale_names = ["identity", "log10"]
    scale_functions = [identity, log10]
    # Create figure with layout
    fig = Figure(size = (1600, 900))

    # Create layout for controls and plots
    control_panel = fig[1, 1:2] = GridLayout(tellheight = true)
    main_plot_panel = fig[2:4, 1] = GridLayout()
    stats_panel = fig[2:4, 3] = GridLayout(tellwidth = true)

    # Control elements - allow all columns for x and y axes
    x_menu = Menu(control_panel[1, 1], options = axis_cols, default = default_x, width = 150)
    y_menu = Menu(control_panel[1, 2], options = axis_cols, default = default_y, width = 150)
    color_menu = Menu(control_panel[1, 3], options = color_options, default = default_color, width = 150)
    plot_type = Menu(
        control_panel[1, 4],
        options = ["scatter", "hexbin", "violin", "boxplot", "density"],
        default = "scatter",
        width = 120
    )

    # Scale menus - use the actual functions as options
    xscale_menu = Menu(control_panel[1, 5], options = zip(scale_names, scale_functions), default = "identity", width = 100)
    yscale_menu = Menu(control_panel[1, 6], options = zip(scale_names, scale_functions), default = "identity", width = 100)
    legend_title_toggle = Toggle(control_panel[1, 7], active = false)

    # Observables to track current scale functions
    xscale_obs = Observable{Function}(scale_functions[1])
    yscale_obs = Observable{Function}(scale_functions[1])

    # Labels for menus
    Label(control_panel[1, 1, Top()], "X Variable", padding = (0, 0, 5, 0))
    Label(control_panel[1, 2, Top()], "Y Variable", padding = (0, 0, 5, 0))
    Label(control_panel[1, 3, Top()], "Color/Group", padding = (0, 0, 5, 0))
    Label(control_panel[1, 4, Top()], "Plot Type", padding = (0, 0, 5, 0))
    Label(control_panel[1, 5, Top()], "X Scale", padding = (0, 0, 5, 0))
    Label(control_panel[1, 6, Top()], "Y Scale", padding = (0, 0, 5, 0))
    Label(control_panel[1, 7, Top()], "Legend Title", padding = (0, 0, 5, 0))

    # Create main plot axis with scale observables
    ax_main = Axis(
        main_plot_panel[1, 1],
        xminorticks = IntervalsBetween(5), xminorticksvisible = true,
        yminorticks = IntervalsBetween(5), yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        xscale = xscale_obs, yscale = yscale_obs
    )

    # Create statistical summary axes
    ax_hist_x = Axis(stats_panel[1, 1], title = "X Distribution", height = 150)
    ax_hist_y = Axis(stats_panel[2, 1], title = "Y Distribution", height = 150)
    ax_stats = Axis(stats_panel[3, 1], title = "Statistics")
    hidedecorations!(ax_stats)
    hidespines!(ax_stats)

    # Track colorbar to manage it properly (using a Ref to store the colorbar object)
    current_colorbar = Ref{Union{Nothing, Colorbar}}(nothing)
    # Track legend to manage it properly
    current_legend = Ref{Union{Nothing, Legend}}(nothing)

    # Helper function to sort categories (numeric sort if possible, otherwise lexicographic)
    function sort_categories(cats)
        # Try to parse all as numbers for numeric sorting
        parsed = tryparse.(Float64, cats)
        if all(!isnothing, parsed)
            # All parse as numbers, sort numerically
            perm = sortperm(parsed)
            return cats[perm]
        else
            # Fall back to lexicographic sort
            return sort(cats)
        end
    end

    # Helper function to check if data is valid for log10 scale (all values must be > 0)
    function is_valid_for_log10(data::AbstractVector{<:Number})
        return all(x -> x > 0, data)
    end
    is_valid_for_log10(::Any) = false  # Non-numeric data is not valid for log10

    # Helper function to extract valid data
    function get_valid_data(x_var, y_var, color_var)
        # Start with all rows
        valid_mask = trues(nrow(df))

        # Filter out rows with missing x or y
        valid_mask .&= .!ismissing.(df[!, x_var])
        valid_mask .&= .!ismissing.(df[!, y_var])

        # Filter out rows with missing color if specified
        if color_var != "none"
            valid_mask .&= .!ismissing.(df[!, color_var])
        end

        # Extract valid data - handle different types
        x_col_type = nonmissingtype(eltype(df[!, x_var]))
        y_col_type = nonmissingtype(eltype(df[!, y_var]))

        if x_col_type <: Number
            x_data = Vector{Float64}(df[valid_mask, x_var])
            x_is_categorical = false
        elseif x_col_type <: Bool
            x_data = Vector{Float64}(Float64.(df[valid_mask, x_var]))  # Convert bool to 0/1
            x_is_categorical = false
        else
            x_data = Vector{String}(string.(df[valid_mask, x_var]))
            x_is_categorical = true
        end

        if y_col_type <: Number
            y_data = Vector{Float64}(df[valid_mask, y_var])
            y_is_categorical = false
        elseif y_col_type <: Bool
            y_data = Vector{Float64}(Float64.(df[valid_mask, y_var]))  # Convert bool to 0/1
            y_is_categorical = false
        else
            y_data = Vector{String}(string.(df[valid_mask, y_var]))
            y_is_categorical = true
        end

        color_data = nothing
        is_categorical = false
        if color_var != "none"
            col_type = nonmissingtype(eltype(df[!, color_var]))
            if col_type <: Number
                raw_color_data = df[valid_mask, color_var]
                unique_vals = unique(raw_color_data)
                if length(unique_vals) <= 8
                    # Treat as categorical - convert to strings preserving numeric order
                    color_data = Vector{String}(string.(raw_color_data))
                    is_categorical = true
                else
                    color_data = Vector{Float64}(raw_color_data)
                    is_categorical = false
                end
            elseif col_type <: Bool
                # Bool always has <= 2 values, treat as categorical
                color_data = Vector{String}(string.(df[valid_mask, color_var]))
                is_categorical = true
            else
                color_data = Vector{String}(string.(df[valid_mask, color_var]))
                is_categorical = true
            end
        end

        n_valid = sum(valid_mask)
        n_missing = nrow(df) - n_valid

        return x_data, y_data, color_data, is_categorical, n_valid, n_missing, x_is_categorical, y_is_categorical
    end

    # Function to update plots
    function update_plots(x_var, y_var, color_var, plot_style)
        # Clear previous plots
        empty!(ax_main)
        empty!(ax_hist_x)
        empty!(ax_hist_y)
        empty!(ax_stats)

        # Delete previous colorbar if it exists
        if !isnothing(current_colorbar[])
            delete!(current_colorbar[])
            current_colorbar[] = nothing
        end

        # Delete previous legend if it exists
        if !isnothing(current_legend[])
            delete!(current_legend[])
            current_legend[] = nothing
        end

        # Get valid data
        x_data, y_data, color_data, is_categorical, n_valid, n_missing, x_is_categorical, y_is_categorical = get_valid_data(x_var, y_var, color_var)

        # Reset axis ticks to automatic before applying plot-specific overrides
        ax_main.xticks = MAKIE_AUTOMATIC
        ax_main.yticks = MAKIE_AUTOMATIC
        ax_hist_x.xticks = MAKIE_AUTOMATIC
        ax_hist_y.xticks = MAKIE_AUTOMATIC

        function create_group_legend()
            if legend_title_toggle.active[]
                current_legend[] = axislegend(ax_main, color_var, position = :rt)
            else
                current_legend[] = axislegend(ax_main, position = :rt)
            end

            return nothing
        end

        if n_valid < 2
            # Not enough data to plot
            text!(
                ax_main, 0.5, 0.5,
                text = "Insufficient data\n(only $(n_valid) valid points)",
                align = (:center, :center),
                space = :relative,
                fontsize = 20
            )
            ax_main.xlabel = x_var
            ax_main.ylabel = y_var
            return
        end

        # Update main plot based on type
        if plot_style == "scatter"
            # Handle categorical axes for scatter plots
            if x_is_categorical && y_is_categorical
                # Both categorical: create a grouped scatter/bar chart
                unique_x = unique(x_data)
                unique_y = unique(y_data)
                x_positions = Dict(cat => i for (i, cat) in enumerate(unique_x))
                y_positions = Dict(cat => i for (i, cat) in enumerate(unique_y))

                x_numeric = [x_positions[x] for x in x_data]
                y_numeric = [y_positions[y] for y in y_data]

                if color_var != "none"
                    if is_categorical
                        unique_cats = sort_categories(unique(color_data))
                        colors = get_wong_colors()
                        for (i, cat) in enumerate(unique_cats)
                            mask = color_data .== cat
                            color_idx = mod1(i, length(colors))
                            scatter!(
                                ax_main, x_numeric[mask], y_numeric[mask],
                                label = string(cat), color = colors[color_idx],
                                markersize = 10
                            )
                        end
                        create_group_legend()
                    else
                        sc = scatter!(
                            ax_main, x_numeric, y_numeric, color = color_data,
                            colormap = :viridis, markersize = 10
                        )
                        current_colorbar[] = Colorbar(fig[2:4, 2], sc, label = color_var)
                    end
                else
                    scatter!(ax_main, x_numeric, y_numeric, markersize = 10)
                end

                ax_main.xticks = (1:length(unique_x), string.(unique_x))
                ax_main.yticks = (1:length(unique_y), string.(unique_y))

            elseif x_is_categorical
                # X is categorical, Y is numeric: grouped scatter
                unique_x = unique(x_data)
                x_positions = Dict(cat => i for (i, cat) in enumerate(unique_x))
                x_numeric = [x_positions[x] for x in x_data]

                if color_var != "none"
                    if is_categorical
                        unique_cats = sort_categories(unique(color_data))
                        colors = get_wong_colors()
                        for (i, cat) in enumerate(unique_cats)
                            mask = color_data .== cat
                            color_idx = mod1(i, length(colors))
                            scatter!(
                                ax_main, x_numeric[mask], y_data[mask],
                                label = string(cat), color = colors[color_idx],
                                markersize = 10
                            )
                        end
                        create_group_legend()
                    else
                        sc = scatter!(
                            ax_main, x_numeric, y_data, color = color_data,
                            colormap = :viridis, markersize = 10
                        )
                        current_colorbar[] = Colorbar(fig[2:4, 2], sc, label = color_var)
                    end
                else
                    scatter!(ax_main, x_numeric, y_data, markersize = 10)
                end

                ax_main.xticks = (1:length(unique_x), string.(unique_x))

            elseif y_is_categorical
                # Y is categorical, X is numeric: grouped scatter
                unique_y = unique(y_data)
                y_positions = Dict(cat => i for (i, cat) in enumerate(unique_y))
                y_numeric = [y_positions[y] for y in y_data]

                if color_var != "none"
                    if is_categorical
                        unique_cats = sort_categories(unique(color_data))
                        colors = get_wong_colors()
                        for (i, cat) in enumerate(unique_cats)
                            mask = color_data .== cat
                            color_idx = mod1(i, length(colors))
                            scatter!(
                                ax_main, x_data[mask], y_numeric[mask],
                                label = string(cat), color = colors[color_idx],
                                markersize = 10
                            )
                        end
                        create_group_legend()
                    else
                        sc = scatter!(
                            ax_main, x_data, y_numeric, color = color_data,
                            colormap = :viridis, markersize = 10
                        )
                        current_colorbar[] = Colorbar(fig[2:4, 2], sc, label = color_var)
                    end
                else
                    scatter!(ax_main, x_data, y_numeric, markersize = 10)
                end

                ax_main.yticks = (1:length(unique_y), string.(unique_y))

            else
                # Both numeric: standard scatter plot
                if color_var != "none"
                    if is_categorical
                        # Categorical coloring
                        unique_cats = sort_categories(unique(color_data))
                        colors = get_wong_colors()
                        for (i, cat) in enumerate(unique_cats)
                            mask = color_data .== cat
                            color_idx = mod1(i, length(colors))
                            scatter!(
                                ax_main, x_data[mask], y_data[mask],
                                label = string(cat), color = colors[color_idx],
                                markersize = 10
                            )
                        end
                        create_group_legend()
                    else
                        # Continuous coloring
                        sc = scatter!(
                            ax_main, x_data, y_data, color = color_data,
                            colormap = :viridis, markersize = 10
                        )
                        current_colorbar[] = Colorbar(fig[2:4, 2], sc, label = color_var)
                    end
                else
                    scatter!(ax_main, x_data, y_data, markersize = 10)
                end

                # Add regression line if we have enough distinct points (only for numeric axes)
                if length(unique(x_data)) > 1 && !x_is_categorical && !y_is_categorical
                    try
                        model = lm(@formula(y ~ x), DataFrame(x = x_data, y = y_data))
                        x_range = range(minimum(x_data), maximum(x_data), length = 100)
                        y_pred = predict(model, DataFrame(x = collect(x_range)))
                        lines!(
                            ax_main, collect(x_range), y_pred, color = (:red, 0.6),
                            linewidth = 2, linestyle = :dash
                        )

                        # Add R² to plot
                        r2_val = r2(model)
                        text!(
                            ax_main, 0.02, 0.98,
                            text = "R² = $(round(r2_val, digits = 3))",
                            align = (:left, :top),
                            space = :relative,
                            fontsize = 12,
                            color = :red
                        )
                    catch e
                        @debug "Could not fit regression line" exception = e
                    end
                end
            end

        elseif plot_style == "hexbin"
            # Hexagonal binning for large datasets (only works with numeric data)
            if !x_is_categorical && !y_is_categorical
                h = hexbin!(
                    ax_main, x_data, y_data, cellsize = (
                        (maximum(x_data) - minimum(x_data)) / 20,
                        (maximum(y_data) - minimum(y_data)) / 20,
                    )
                )
                current_colorbar[] = Colorbar(fig[2:4, 2], h, label = "Count")
            else
                text!(
                    ax_main, 0.5, 0.5,
                    text = "Hexbin plot requires\nnumeric axes",
                    align = (:center, :center),
                    space = :relative,
                    fontsize = 16
                )
            end

        elseif plot_style == "boxplot"
            # Boxplot requires numeric y-axis
            if y_is_categorical
                text!(
                    ax_main, 0.5, 0.5,
                    text = "Boxplot requires\nnumeric y-axis",
                    align = (:center, :center),
                    space = :relative,
                    fontsize = 16
                )
            elseif color_var != "none" && is_categorical
                # Grouped boxplot
                unique_cats = sort_categories(unique(color_data))
                colors = get_wong_colors()

                for (i, cat) in enumerate(unique_cats)
                    mask = color_data .== cat
                    y_subset = y_data[mask]
                    if !isempty(y_subset)
                        color_idx = mod1(i, length(colors))
                        boxplot!(
                            ax_main, fill(i, length(y_subset)), y_subset,
                            color = colors[color_idx], label = string(cat)
                        )
                    end
                end
                ax_main.xticks = (1:length(unique_cats), string.(unique_cats))
                ax_main.xlabel = color_var
                create_group_legend()
            elseif x_is_categorical
                # Group by x-axis categories
                unique_x = unique(x_data)
                for (i, cat) in enumerate(unique_x)
                    mask = x_data .== cat
                    y_subset = y_data[mask]
                    if !isempty(y_subset)
                        boxplot!(ax_main, fill(i, length(y_subset)), y_subset, width = 0.3)
                    end
                end
                ax_main.xticks = (1:length(unique_x), string.(unique_x))
                ax_main.xlabel = x_var
                ax_main.ylabel = y_var
            else
                # Simple boxplot
                boxplot!(ax_main, ones(length(y_data)), y_data, width = 0.3)
                ax_main.xticks = ([1], [x_var])
            end

        elseif plot_style == "violin"
            # Violin plot requires numeric y-axis
            if y_is_categorical
                text!(
                    ax_main, 0.5, 0.5,
                    text = "Violin plot requires\nnumeric y-axis",
                    align = (:center, :center),
                    space = :relative,
                    fontsize = 16
                )
            elseif color_var != "none" && is_categorical
                # Grouped violin plot
                unique_cats = sort_categories(unique(color_data))
                colors = get_wong_colors()

                for (i, cat) in enumerate(unique_cats)
                    mask = color_data .== cat
                    y_subset = y_data[mask]
                    if !isempty(y_subset) && length(unique(y_subset)) > 1
                        color_idx = mod1(i, length(colors))
                        violin!(
                            ax_main, fill(i, length(y_subset)), y_subset,
                            color = colors[color_idx], label = string(cat)
                        )
                    end
                end
                ax_main.xticks = (1:length(unique_cats), string.(unique_cats))
                ax_main.xlabel = color_var
                create_group_legend()
            elseif x_is_categorical
                # Group by x-axis categories
                unique_x = unique(x_data)
                for (i, cat) in enumerate(unique_x)
                    mask = x_data .== cat
                    y_subset = y_data[mask]
                    if !isempty(y_subset) && length(unique(y_subset)) > 1
                        violin!(ax_main, fill(i, length(y_subset)), y_subset, width = 0.5)
                    end
                end
                ax_main.xticks = (1:length(unique_x), string.(unique_x))
                ax_main.xlabel = x_var
                ax_main.ylabel = y_var
            else
                # Simple violin plot
                if length(unique(y_data)) > 1
                    violin!(ax_main, ones(length(y_data)), y_data, width = 0.5)
                    ax_main.xticks = ([1], [x_var])
                end
            end

        elseif plot_style == "density"
            # Density plot requires numeric y-axis
            if y_is_categorical
                text!(
                    ax_main, 0.5, 0.5,
                    text = "Density plot requires\nnumeric y-axis",
                    align = (:center, :center),
                    space = :relative,
                    fontsize = 16
                )
            elseif color_var != "none" && is_categorical
                # Multiple density plots
                unique_cats = sort_categories(unique(color_data))
                colors = get_wong_colors()

                for (i, cat) in enumerate(unique_cats)
                    mask = color_data .== cat
                    y_subset = y_data[mask]
                    if length(y_subset) > 2 && length(unique(y_subset)) > 1
                        color_idx = mod1(i, length(colors))
                        density!(
                            ax_main, y_subset, color = (colors[color_idx], 0.5),
                            strokecolor = colors[color_idx], strokewidth = 2,
                            label = string(cat)
                        )
                    end
                end
                ax_main.xlabel = y_var
                ax_main.ylabel = "Density"
                create_group_legend()
            else
                # Single density plot
                if length(y_data) > 2
                    density!(ax_main, y_data)
                    ax_main.xlabel = y_var
                    ax_main.ylabel = "Density"
                end
            end
        end

        # Update histograms
        if length(x_data) > 0
            if x_is_categorical
                # Bar chart for categorical data
                unique_x = unique(x_data)
                counts_x = [count(==(cat), x_data) for cat in unique_x]
                barplot!(ax_hist_x, 1:length(unique_x), counts_x, color = (:blue, 0.5))
                ax_hist_x.xticks = (1:length(unique_x), string.(unique_x))
            else
                hist!(
                    ax_hist_x, x_data, bins = min(30, max(10, length(unique(x_data)))),
                    color = (:blue, 0.5), strokewidth = 1, strokecolor = :blue
                )
            end
            ax_hist_x.xlabel = x_var
        end

        if length(y_data) > 0
            if y_is_categorical
                # Bar chart for categorical data
                unique_y = unique(y_data)
                counts_y = [count(==(cat), y_data) for cat in unique_y]
                barplot!(ax_hist_y, 1:length(unique_y), counts_y, color = (:red, 0.5))
                ax_hist_y.xticks = (1:length(unique_y), string.(unique_y))
            else
                hist!(
                    ax_hist_y, y_data, bins = min(30, max(10, length(unique(y_data)))),
                    color = (:red, 0.5), strokewidth = 1, strokecolor = :red
                )
            end
            ax_hist_y.xlabel = y_var
        end

        # Compute and display statistics
        stats_lines = String[]
        push!(stats_lines, "Data Points:")
        push!(stats_lines, "  Valid: $(n_valid)")
        push!(stats_lines, "  Missing: $(n_missing)")
        push!(stats_lines, "")
        push!(stats_lines, "X: $(x_var)")
        if x_is_categorical
            unique_x = unique(x_data)
            push!(stats_lines, "  Unique values: $(length(unique_x))")
            # Find most common value
            counts = Dict(cat => count(==(cat), x_data) for cat in unique_x)
            most_common = argmax(cat -> counts[cat], unique_x)
            push!(stats_lines, "  Most common: $(most_common) ($(counts[most_common]) times)")
        else
            push!(stats_lines, "  Mean: $(round(mean(x_data), digits = 4))")
            push!(stats_lines, "  Std: $(round(std(x_data), digits = 4))")
            push!(stats_lines, "  Min: $(round(minimum(x_data), digits = 4))")
            push!(stats_lines, "  Max: $(round(maximum(x_data), digits = 4))")
        end
        push!(stats_lines, "")
        push!(stats_lines, "Y: $(y_var)")
        if y_is_categorical
            unique_y = unique(y_data)
            push!(stats_lines, "  Unique values: $(length(unique_y))")
            # Find most common value
            counts = Dict(cat => count(==(cat), y_data) for cat in unique_y)
            most_common = argmax(cat -> counts[cat], unique_y)
            push!(stats_lines, "  Most common: $(most_common) ($(counts[most_common]) times)")
        else
            push!(stats_lines, "  Mean: $(round(mean(y_data), digits = 4))")
            push!(stats_lines, "  Std: $(round(std(y_data), digits = 4))")
            push!(stats_lines, "  Min: $(round(minimum(y_data), digits = 4))")
            push!(stats_lines, "  Max: $(round(maximum(y_data), digits = 4))")
        end

        if !isnothing(color_data) && length(color_data) > 0
            push!(stats_lines, "")
            push!(stats_lines, "Color: $(color_var)")
            if is_categorical
                unique_colors = unique(color_data)
                push!(stats_lines, "  Unique values: $(length(unique_colors))")
            else
                push!(stats_lines, "  Mean: $(round(mean(color_data), digits = 4))")
                push!(stats_lines, "  Std: $(round(std(color_data), digits = 4))")
            end
        end

        # Add correlation if scatter plot (only for numeric data)
        if plot_style == "scatter" && length(x_data) > 1 && !x_is_categorical && !y_is_categorical
            corr = cor(x_data, y_data)
            push!(stats_lines, "")
            push!(stats_lines, "Correlation: $(round(corr, digits = 4))")
        end

        stats_text = join(stats_lines, "\n")

        text!(
            ax_stats, 0.05, 0.95,
            text = stats_text,
            align = (:left, :top),
            space = :relative,
            fontsize = 11
        )

        # Update labels
        if !(plot_style in ["boxplot", "violin", "density"])
            ax_main.xlabel = x_var
            ax_main.ylabel = y_var
        end
        return ax_main.title = "$(uppercasefirst(plot_style)): $(y_var) vs $(x_var)"
    end

    # Connect observables
    on(x_menu.selection) do x_var
        update_plots(x_var, y_menu.selection[], color_menu.selection[], plot_type.selection[])
    end

    on(y_menu.selection) do y_var
        update_plots(x_menu.selection[], y_var, color_menu.selection[], plot_type.selection[])
    end

    on(color_menu.selection) do color_var
        update_plots(x_menu.selection[], y_menu.selection[], color_var, plot_type.selection[])
    end

    on(plot_type.selection) do style
        update_plots(x_menu.selection[], y_menu.selection[], color_menu.selection[], style)
    end

    on(legend_title_toggle.active) do _
        update_plots(x_menu.selection[], y_menu.selection[], color_menu.selection[], plot_type.selection[])
    end

    # Scale menu listeners - update the observables when selection changes
    # Check for non-positive data before applying log10
    on(xscale_menu.selection) do scale_func
        if scale_func === log10
            # Get current x data to check for non-positive values
            x_data, _, _, _, _, _, x_is_categorical, _ = get_valid_data(
                x_menu.selection[], y_menu.selection[], color_menu.selection[]
            )
            if x_is_categorical || !is_valid_for_log10(x_data)
                @warn "Cannot apply log10 scale to X axis: data contains non-positive or non-numeric values. Reverting to identity."
                xscale_menu.i_selected[] = 1  # Reset menu to identity (first option)
                return
            end
        end
        xscale_obs[] = scale_func
    end

    on(yscale_menu.selection) do scale_func
        if scale_func === log10
            # Get current y data to check for non-positive values
            _, y_data, _, _, _, _, _, y_is_categorical = get_valid_data(
                x_menu.selection[], y_menu.selection[], color_menu.selection[]
            )
            if y_is_categorical || !is_valid_for_log10(y_data)
                @warn "Cannot apply log10 scale to Y axis: data contains non-positive or non-numeric values. Reverting to identity."
                yscale_menu.i_selected[] = 1  # Reset menu to identity (first option)
                return
            end
        end
        yscale_obs[] = scale_func
    end

    # Initial plot
    update_plots(x_menu.selection[], y_menu.selection[], color_menu.selection[], plot_type.selection[])

    # Add save button and counter
    save_counter = Ref(0)
    save_button = Button(control_panel[1, 8], label = "Save Plot", width = 100)
    on(save_button.clicks) do _
        save_counter[] += 1
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        filename = "analysis_$(timestamp)_$(save_counter[]).png"
        save(filename, fig)
        @info "Plot saved to: $(filename)"
    end

    display(fig)
    return fig
end

"""
    analyze_continued_training(experiment_name::String; filter_function::Function=(result)->true)

Analyze the results of continued training, comparing the original and continued performance.

# Arguments
- `experiment_name::String`: Name of the experiment containing continued models
- `filter_function::Function=(result)->true`: Optional function to filter which results to include in the analysis

# Returns
- `DataFrame`: DataFrame with comparison data for all continued models

# Example
```julia
# Analyze all continued models in an experiment:
results = analyze_continued_training("my_experiment")

# Analyze only models with target_shock_count=3:
results = analyze_continued_training("my_experiment",
                                   filter_function=(result) -> get(result, "target_shock_count", 0) == 3)
```
"""
function analyze_continued_training(experiment_name::String; filter_function::Function = (result) -> true)
    # Load results from the experiment
    results_dir = datadir("sb_experiments", experiment_name, "results")

    if !isdir(results_dir)
        @warn "Results directory not found: $results_dir"
        return DataFrame()
    end

    results = DrWatson.collect_results!(results_dir)

    # Filter for continued models
    continued_results = filter(r -> haskey(r, :continued_training) && r.continued_training, results)

    # Apply additional filter if provided
    filtered_results = filter(filter_function, continued_results)

    if nrow(filtered_results) == 0
        @warn "No continued training results found in experiment: $experiment_name"
        return DataFrame()
    end

    @info "Found $(nrow(filtered_results)) continued models in $experiment_name"

    # Create a DataFrame for comparison
    comparison_data = []

    for result in eachrow(filtered_results)
        # Extract original and continued metrics
        original_timesteps = result.total_timesteps - result.continued_timesteps
        continued_timesteps = result.continued_timesteps

        # Create a row with comparison data
        push!(
            comparison_data, Dict(
                "model" => basename(result.model_path),
                "original_timesteps" => original_timesteps,
                "continued_timesteps" => continued_timesteps,
                "total_timesteps" => result.total_timesteps,
                "mean_reward" => result.mean_reward,
                "std_reward" => result.std_reward,
                "scaled_mean_reward" => get(result, :scaled_mean_reward, nothing),
                "target_shock_count" => get(result, :target_shock_count, nothing),
                "continued_at" => get(result, :timestamp, nothing),
                "computer" => get(result, :computer, nothing),
                "dt" => get(result, :dt, 0.1),
                "tmax" => get(result, :tmax, 100.0)
            )
        )
    end

    # Convert to DataFrame
    comparison_df = DataFrame(comparison_data)

    # Print summary statistics
    println("\nSummary Statistics for Continued Training in $experiment_name:")
    println("=================================================================")
    println("Number of models: $(nrow(comparison_df))")
    println("Mean reward: $(round(mean(comparison_df.mean_reward), digits = 3)) ± $(round(std(comparison_df.mean_reward), digits = 3))")
    if "scaled_mean_reward" in names(comparison_df) && !all(ismissing.(comparison_df.scaled_mean_reward))
        println("Mean scaled reward: $(round(mean(skipmissing(comparison_df.scaled_mean_reward)), digits = 3)) ± $(round(std(skipmissing(comparison_df.scaled_mean_reward)), digits = 3))")
    end

    # Create visualizations
    create_continued_training_plots(comparison_df, experiment_name)

    return comparison_df
end

"""
    create_continued_training_plots(comparison_df::DataFrame, experiment_name::String)

Create visualizations for the continued training results.
"""
function create_continued_training_plots(comparison_df::DataFrame, experiment_name::String)
    # Create a directory for the plots
    plot_dir = plotsdir("sb_experiments", "continued_training_analysis", experiment_name)
    mkpath(plot_dir)

    # 1. Histogram of rewards
    if nrow(comparison_df) > 1
        fig1 = Figure(resolution = (800, 600))
        ax1 = Axis(
            fig1[1, 1],
            xlabel = "Mean Reward",
            ylabel = "Count",
            title = "Distribution of Rewards after Continued Training"
        )

        hist!(ax1, comparison_df.mean_reward, bins = min(20, nrow(comparison_df)))

        save(joinpath(plot_dir, "reward_histogram.png"), fig1)
    end

    # 2. Bar plot of rewards by target_shock_count (if available)
    if "target_shock_count" in names(comparison_df) && length(unique(skipmissing(comparison_df.target_shock_count))) > 1
        fig2 = Figure(resolution = (800, 600))
        ax2 = Axis(
            fig2[1, 1],
            xlabel = "Target Shock Count",
            ylabel = "Mean Reward",
            title = "Reward by Target Shock Count"
        )

        # Group data by target_shock_count
        shock_counts = sort(collect(unique(skipmissing(comparison_df.target_shock_count))))
        reward_means = [mean(comparison_df[comparison_df.target_shock_count .== sc, :mean_reward]) for sc in shock_counts]
        reward_stds = [std(comparison_df[comparison_df.target_shock_count .== sc, :mean_reward]) for sc in shock_counts]

        # Create barplot
        barplot!(ax2, shock_counts, reward_means)
        errorbars!(ax2, shock_counts, reward_means, reward_stds)

        save(joinpath(plot_dir, "reward_by_shock_count.png"), fig2)
    end

    # 3. Reward vs. total timesteps
    return if nrow(comparison_df) > 1
        fig3 = Figure(resolution = (800, 600))
        ax3 = Axis(
            fig3[1, 1],
            xlabel = "Total Timesteps",
            ylabel = "Mean Reward",
            title = "Reward vs. Total Timesteps"
        )

        scatter!(ax3, comparison_df.total_timesteps, comparison_df.mean_reward)

        save(joinpath(plot_dir, "reward_vs_timesteps.png"), fig3)
    end
end

"""
    compare_experiment_versions(original_experiment::String, continued_experiment::String;
                               filter_function::Function=(result)->true)

Compare results between original experiment and a continued version of it.

# Arguments
- `original_experiment::String`: Name of the original experiment
- `continued_experiment::String`: Name of the experiment with continued models
- `filter_function::Function=(result)->true`: Optional function to filter which results to include

# Returns
- `DataFrame`: DataFrame with comparison data

# Example
```julia
# Compare all models between original and continued experiment:
results = compare_experiment_versions("original_exp", "continued_exp")
```
"""
function compare_experiment_versions(
        original_experiment::String, continued_experiment::String;
        filter_function::Function = (result) -> true
    )
    # Load results from both experiments
    original_results = DrWatson.collect_results!(datadir("sb_experiments", original_experiment, "results"))
    continued_results = DrWatson.collect_results!(datadir("sb_experiments", continued_experiment, "results"))

    # Check if we have results
    if nrow(original_results) == 0
        @warn "No results found for original experiment: $original_experiment"
        return DataFrame()
    end
    if nrow(continued_results) == 0
        @warn "No results found for continued experiment: $continued_experiment"
        return DataFrame()
    end

    # Apply additional filter if provided
    original_results = filter(filter_function, original_results)
    continued_results = filter(filter_function, continued_results)

    @info "Found $(nrow(original_results)) original models and $(nrow(continued_results)) continued models"

    # Create a mapping from original model filenames to continued model results
    continued_mapping = Dict()
    for i in 1:nrow(continued_results)
        if haskey(continued_results[i, :], "continued_from")
            continued_mapping[continued_results[i, :].continued_from] = continued_results[i, :]
        end
    end

    # Create a DataFrame for comparison
    comparison_data = []

    for i in 1:nrow(original_results)
        original = original_results[i, :]

        # Extract the model filename from the model path
        model_filename = basename(original.model_path)

        if haskey(continued_mapping, model_filename)
            continued = continued_mapping[model_filename]

            # Create a row with comparison data
            push!(
                comparison_data, Dict(
                    "original_reward" => original.mean_reward,
                    "continued_reward" => continued.mean_reward,
                    "original_scaled_reward" => get(original, :scaled_mean_reward, original.mean_reward * get(original, :dt, 0.1) / get(original, :tmax, 100.0)),
                    "continued_scaled_reward" => continued.scaled_mean_reward,
                    "improvement_absolute" => continued.mean_reward - original.mean_reward,
                    "improvement_percent" => (continued.mean_reward - original.mean_reward) / abs(original.mean_reward) * 100,
                    "original_timesteps" => get(original, :total_timesteps, 0),
                    "additional_timesteps" => get(continued, :continued_timesteps, get(continued, :additional_timesteps, 0)),
                    "total_timesteps" => get(continued, :total_timesteps, 0),
                    "target_shock_count" => get(original, :target_shock_count, 0),
                    "original_model" => model_filename,
                    "continued_model" => basename(continued.model_path)
                )
            )
        end
    end

    # Convert to DataFrame
    comparison_df = DataFrame(comparison_data)

    if nrow(comparison_df) == 0
        @warn "No matching models found between original and continued experiments"
        return DataFrame()
    end

    # Print summary statistics
    println("\nSummary Statistics:")
    println("===================")
    println("Number of models: $(nrow(comparison_df))")
    println("Mean original reward: $(round(mean(comparison_df.original_reward), digits = 3)) ± $(round(std(comparison_df.original_reward), digits = 3))")
    println("Mean continued reward: $(round(mean(comparison_df.continued_reward), digits = 3)) ± $(round(std(comparison_df.continued_reward), digits = 3))")
    println("Mean improvement (absolute): $(round(mean(comparison_df.improvement_absolute), digits = 3)) ± $(round(std(comparison_df.improvement_absolute), digits = 3))")
    println("Mean improvement (percent): $(round(mean(comparison_df.improvement_percent), digits = 1))% ± $(round(std(comparison_df.improvement_percent), digits = 1))%")

    # Count models with improvement
    improved_models = count(comparison_df.improvement_absolute .> 0)
    println("Models with improvement: $improved_models / $(nrow(comparison_df)) ($(round(improved_models / nrow(comparison_df) * 100, digits = 1))%)")

    # Create visualizations
    create_comparison_plots(comparison_df, original_experiment, continued_experiment)

    return comparison_df
end

"""
    create_comparison_plots(comparison_df::DataFrame, original_experiment::String, continued_experiment::String)

Create visualizations comparing the original and continued training results.
"""
function create_comparison_plots(comparison_df::DataFrame, original_experiment::String, continued_experiment::String)
    # Create a directory for the plots
    plot_dir = plotsdir("sb_experiments", "continued_training_analysis")
    mkpath(plot_dir)

    # 1. Scatter plot of original vs continued reward
    fig1 = Figure(resolution = (800, 600))
    ax1 = Axis(
        fig1[1, 1],
        xlabel = "Original Mean Reward",
        ylabel = "Continued Mean Reward",
        title = "Reward Comparison: Original vs Continued"
    )

    scatter!(ax1, comparison_df.original_reward, comparison_df.continued_reward)

    # Add diagonal line (y = x)
    min_val = min(minimum(comparison_df.original_reward), minimum(comparison_df.continued_reward))
    max_val = max(maximum(comparison_df.original_reward), maximum(comparison_df.continued_reward))
    lines!(ax1, [min_val, max_val], [min_val, max_val], color = :red, linestyle = :dash)

    save(joinpath(plot_dir, "$(original_experiment)_vs_$(continued_experiment)_scatter.png"), fig1)

    # 2. Histogram of improvement
    fig2 = Figure(resolution = (800, 600))
    ax2 = Axis(
        fig2[1, 1],
        xlabel = "Reward Improvement (%)",
        ylabel = "Count",
        title = "Distribution of Reward Improvement"
    )

    hist!(ax2, comparison_df.improvement_percent, bins = 20)

    # Add vertical line at 0
    vlines!(ax2, [0], color = :red, linestyle = :dash)

    save(joinpath(plot_dir, "$(original_experiment)_vs_$(continued_experiment)_improvement_hist.png"), fig2)

    # 3. Box plot of rewards by target_shock_count
    if "target_shock_count" in names(comparison_df) && length(unique(comparison_df.target_shock_count)) > 1
        fig3 = Figure(resolution = (1000, 600))
        ax3 = Axis(
            fig3[1, 1],
            xlabel = "Target Shock Count",
            ylabel = "Mean Reward",
            title = "Reward by Target Shock Count: Original vs Continued"
        )

        # Group data by target_shock_count
        shock_counts = sort(unique(comparison_df.target_shock_count))

        # Create positions for grouped boxplots
        positions = []
        for (i, count) in enumerate(shock_counts)
            push!(positions, i - 0.2)  # Original
            push!(positions, i + 0.2)  # Continued
        end

        # Prepare data for boxplots
        grouped_data = []
        for count in shock_counts
            subset = comparison_df[comparison_df.target_shock_count .== count, :]
            push!(grouped_data, subset.original_reward)
            push!(grouped_data, subset.continued_reward)
        end

        # Create boxplots
        boxplot!(ax3, positions, grouped_data)

        # Add custom x-ticks and legend
        xticks = []
        for (i, count) in enumerate(shock_counts)
            push!(xticks, (i, string(count)))
        end
        ax3.xticks = xticks

        # Add legend
        elements = [PolyElement(color = :blue), PolyElement(color = :orange)]
        labels = ["Original", "Continued"]
        Legend(fig3[1, 2], elements, labels)

        save(joinpath(plot_dir, "$(original_experiment)_vs_$(continued_experiment)_by_shock_count.png"), fig3)
    end

    # 4. Improvement vs original performance
    fig4 = Figure(resolution = (800, 600))
    ax4 = Axis(
        fig4[1, 1],
        xlabel = "Original Mean Reward",
        ylabel = "Improvement (%)",
        title = "Improvement vs Original Performance"
    )

    scatter!(ax4, comparison_df.original_reward, comparison_df.improvement_percent)

    # Add horizontal line at 0
    hlines!(ax4, [0], color = :red, linestyle = :dash)

    save(joinpath(plot_dir, "$(original_experiment)_vs_$(continued_experiment)_improvement_vs_original.png"), fig4)

    # 5. Visualize best improved model
    best_idx = argmax(comparison_df.improvement_percent)
    best_model = comparison_df[best_idx, :]

    println("\nBest Improved Model:")
    println("====================")
    println("Original model: $(best_model.original_model)")
    println("Continued model: $(best_model.continued_model)")
    println("Original reward: $(round(best_model.original_reward, digits = 3))")
    println("Continued reward: $(round(best_model.continued_reward, digits = 3))")
    return println("Improvement: $(round(best_model.improvement_absolute, digits = 3)) ($(round(best_model.improvement_percent, digits = 1))%)")
end

"""
    find_incomplete_runs(experiment_name::String) -> Vector{String}

Find runs that are missing any component (log, result, or model files).
A run is considered incomplete if it's missing any of these components.
Intermediate model checkpoints that end with `_<step>.jld2` are ignored.

# Arguments
- `experiment_name::String`: Name of the experiment to analyze
- `dir::String="studies"`: Directory name containing the experiment

# Returns
- `Vector{String}`: List of incomplete run identifiers (configurations that need to be rerun)

# Examples
```julia
# Find all incomplete runs
incomplete = find_incomplete_runs("my_experiment")

# Find incomplete runs in a different directory
incomplete = find_incomplete_runs("my_experiment", dir="sb_experiments")
```
"""
function find_incomplete_runs(study_name::String; dir::String = "studies")
    # Get paths to log, result, and model directories
    log_dir = datadir(dir, study_name, "logs")
    results_dir = datadir(dir, study_name, "results")
    models_dir = datadir(dir, study_name, "models")

    # Check which directories exist and get their contents
    log_dirs = isdir(log_dir) ? Set(readdir(log_dir)) : Set{String}()
    result_files = isdir(results_dir) ? readdir(results_dir) : String[]
    model_files = isdir(models_dir) ? readdir(models_dir) : String[]
    final_model_files = filter(file -> !occursin(r"_\d+\.jld2$", file), model_files)

    # Convert result and model filenames to match log directory names (remove .jld2 extension)
    result_names = Set(replace(f, ".jld2" => "") for f in result_files)
    model_names = Set(replace(f, ".jld2" => "") for f in final_model_files)

    # Find runs that are missing components using set operations
    # A run is complete if it exists in all three sets
    all_runs = union(log_dirs, result_names, model_names)
    complete_runs = intersect(log_dirs, result_names, model_names)
    incomplete_runs = setdiff(all_runs, complete_runs)

    @info "Found $(length(log_dirs)) log directories, $(length(result_files)) result files, and $(length(final_model_files)) model files"

    # Build details for each incomplete run
    incomplete_details = Dict{String, Vector{String}}()
    for run in incomplete_runs
        missing = String[]
        if !(run in log_dirs)
            push!(missing, "log")
        end
        if !(run in result_names)
            push!(missing, "result")
        end
        if !(run in model_names)
            push!(missing, "model")
        end
        incomplete_details[run] = missing
    end

    # Print summary
    if !isempty(incomplete_runs)
        println("\nFound $(length(incomplete_runs)) incomplete runs:")
        for run in sort(collect(incomplete_runs))
            missing = join(incomplete_details[run], ", ")
            println("  - $run (missing: $missing)")
        end
    else
        println("\nNo incomplete runs found. All runs have logs, results, and models.")
    end

    return sort(collect(incomplete_runs))
end

"""
    cleanup_incomplete_runs(experiment_name::String; dry_run::Bool=true)

Delete log directories for runs that have been started but not completed.

# Arguments
- `experiment_name::String`: Name of the experiment to clean up
- `dry_run::Bool=true`: If true, only print what would be deleted without actually deleting

# Returns
- `Vector{String}`: List of deleted/incomplete run identifiers
"""
function cleanup_incomplete_runs(study_name::String; dry_run::Bool = true, dir::String = "studies")
    # Find incomplete runs
    studydir = datadir(dir, study_name)
    incomplete_runs = find_incomplete_runs(study_name)

    if isempty(incomplete_runs)
        return String[]
    end

    # Get path to log directory
    log_dir = joinpath(studydir, "logs")

    # Delete each incomplete run's log directory
    for run in incomplete_runs
        run_path = joinpath(log_dir, run)
        if dry_run
            # println("Would delete: $run_path")
        else
            rm(run_path, recursive = true)
            println("Deleted: $run_path")
        end
    end

    if dry_run
        println("\nThis was a dry run. No files were deleted.")
        println("To actually delete these files, run with dry_run=false")
    end

    return incomplete_runs
end

"""
    group_by_seed(df::DataFrame;
                  group_cols::Vector{<:Union{Symbol, String}} = Symbol[],
                  metric_cols::Vector{<:Union{Symbol, String}} = String[],
                  seed_col::Union{Symbol, String, Nothing} = nothing)

Group a DataFrame by seed and optionally other columns, computing summary statistics for metric columns.

If a `seed` column exists in the DataFrame, it will be used directly. Otherwise, the function
will attempt to extract the seed from the `experiment` column (assuming `experiment.seed`).

# Arguments
- `df::DataFrame`: The DataFrame containing the results
- `group_cols::Vector`: Additional columns to group by (besides seed). Default: empty
- `metric_cols::Vector`: Columns to compute statistics for. Default: auto-detect numeric columns
- `seed_col::Union{Symbol, String, Nothing}`: Column name containing seed values. If `nothing`,
   auto-detects from `seed` column or extracts from `experiment.seed`

# Returns
- `DataFrame`: Aggregated DataFrame with one row per group, containing:
  - The grouping columns
  - `n_seeds`: number of seeds in each group
  - `_indices`: vector of row indices from the original DataFrame for this group
  - For each metric: `{metric}_mean`, `{metric}_std`, `{metric}_min`, `{metric}_max`

# Example
```julia
# Group by seed only, auto-detect metrics
grouped = group_by_seed(results)

# Group by seed and target_shock_count
grouped = group_by_seed(results, group_cols=[:target_shock_count])

# Specify which metrics to aggregate
grouped = group_by_seed(results, metric_cols=["eval/ep_rew_mean", "train/ep_rew_mean"])

# Access original rows for a group
best_idx = argmax(grouped[!, "eval/ep_rew_mean_mean"])
original_rows = results[grouped[best_idx, :_indices], :]
```
"""
function group_by_seed(
        df::DataFrame;
        group_cols::Vector{<:Union{Symbol, String}} = Symbol[],
        metric_cols::Vector{<:Union{Symbol, String}} = String[],
        seed_col::Union{Symbol, String, Nothing} = nothing,
    )
    # Work with a copy to avoid modifying the original
    result_df = copy(df)

    # Add original row indices for tracking
    result_df[!, :_orig_idx] = 1:nrow(result_df)

    # Convert group_cols to Symbols
    group_cols = Symbol.(group_cols)

    # Determine or create seed column
    if !isnothing(seed_col)
        seed_col = Symbol(seed_col)
        if !(seed_col in Symbol.(names(result_df)))
            error("Specified seed column '$seed_col' not found in DataFrame")
        end
    elseif :seed in Symbol.(names(result_df))
        seed_col = :seed
    elseif :experiment in Symbol.(names(result_df))
        # Extract seed from experiment objects
        result_df[!, :_seed] = [exp.seed for exp in result_df.experiment]
        seed_col = :_seed
    else
        error("No seed column found. Provide a seed_col argument or ensure DataFrame has 'seed' or 'experiment' column")
    end

    # Determine metric columns if not specified
    if isempty(metric_cols)
        # Auto-detect numeric columns, excluding grouping columns and special columns
        exclude_cols = Set(
            vcat(
                group_cols,
                [seed_col, :path, :experiment, :_seed],
                # Exclude columns that are typically not metrics
                [:run, :computer, :timestamp]
            )
        )

        metric_cols = String[]
        for col in names(result_df)
            col_sym = Symbol(col)
            if col_sym in exclude_cols
                continue
            end
            col_type = nonmissingtype(eltype(result_df[!, col]))
            if col_type <: Number
                push!(metric_cols, col)
            end
        end
    end

    metric_cols = String.(metric_cols)

    # Validate metric columns exist
    for col in metric_cols
        if !(col in names(result_df))
            @warn "Metric column '$col' not found in DataFrame, skipping"
        end
    end
    metric_cols = filter(c -> c in names(result_df), metric_cols)

    if isempty(metric_cols)
        @warn "No metric columns to aggregate"
        return DataFrame()
    end

    # Combine seed with other grouping columns
    all_group_cols = vcat([seed_col], group_cols)

    # Filter to only valid grouping columns
    all_group_cols = filter(c -> c in Symbol.(names(result_df)), all_group_cols)

    # First, group by all columns (including seed) to get one row per configuration per seed
    # This handles the case where there might be duplicates

    # Now group by non-seed columns to aggregate across seeds
    if isempty(group_cols)
        # If no additional group columns, we just compute overall statistics
        stats_data = Dict{String, Any}()
        stats_data["n_seeds"] = length(unique(result_df[!, seed_col]))
        stats_data["_indices"] = collect(result_df[!, :_orig_idx])

        for col in metric_cols
            values = result_df[!, col]
            values = filter(v -> !ismissing(v) && !isnan(v), values)

            if isempty(values)
                stats_data["$(col)_mean"] = NaN
                stats_data["$(col)_std"] = NaN
                stats_data["$(col)_min"] = NaN
                stats_data["$(col)_max"] = NaN
            else
                stats_data["$(col)_mean"] = mean(values)
                stats_data["$(col)_std"] = length(values) > 1 ? std(values) : 0.0
                stats_data["$(col)_min"] = minimum(values)
                stats_data["$(col)_max"] = maximum(values)
            end
        end

        # Clean up temporary columns if we created them
        if :_seed in Symbol.(names(result_df))
            select!(result_df, Not(:_seed))
        end
        select!(result_df, Not(:_orig_idx))

        return DataFrame([stats_data])
    end

    # Group by non-seed columns and aggregate
    grouped = groupby(result_df, group_cols)

    summary_df = combine(grouped) do g
        stats = Dict{String, Any}()
        stats["n_seeds"] = length(unique(g[!, seed_col]))
        stats["_indices"] = collect(g[!, :_orig_idx])

        for col in metric_cols
            values = g[!, col]
            values = filter(v -> !ismissing(v) && !isnan(v), values)

            if isempty(values)
                stats["$(col)_mean"] = NaN
                stats["$(col)_std"] = NaN
                stats["$(col)_min"] = NaN
                stats["$(col)_max"] = NaN
            else
                stats["$(col)_mean"] = mean(values)
                stats["$(col)_std"] = length(values) > 1 ? std(values) : 0.0
                stats["$(col)_min"] = minimum(values)
                stats["$(col)_max"] = maximum(values)
            end
        end

        return DataFrame([stats])
    end

    # Clean up temporary columns if we created them
    if :_seed in Symbol.(names(result_df))
        select!(result_df, Not(:_seed))
    end
    select!(result_df, Not(:_orig_idx))

    return summary_df
end

"""
    categorical_comparison(df::DataFrame, category_col::Union{Symbol, String};
                           metric_col = "eval/ep_rew_mean",
                           title = nothing,
                           sort_by = :mean,
                           descending = true,
                           backend = :text,
                           return_string = false)

Analyze a categorical variable by computing summary statistics for each unique value.

# Arguments
- `df::DataFrame`: The DataFrame containing the data
- `category_col`: The column name (Symbol or String) of the categorical variable to group by
- `metric_col`: The column name for the metric to analyze (default: "eval/ep_rew_mean")
- `title`: Optional title for the table (default: auto-generated)
- `sort_by`: Column to sort results by (:value, :count, :mean, :median, :std) (default: :mean)
- `descending`: Sort in descending order (default: true)
- `backend`: Output format - `:text`, `:markdown`, `:html`, or `:latex` (default: :text)
- `return_string`: If true, return the table as a string instead of printing (default: false)

# Returns
- `(DataFrame, String|Nothing)`: The summary statistics DataFrame and optionally the table string

# Example
```julia
categorical_comparison(res, :n_sections)
categorical_comparison(res, "target_shock_count", metric_col = "train/ep_rew_mean")
categorical_comparison(res, :n_sections, backend = :markdown, return_string = true)
```
"""
function categorical_comparison(
        df::DataFrame,
        category_col::Union{Symbol, String};
        metric_col::Union{Symbol, String} = "eval/ep_rew_mean",
        title::Union{Nothing, String} = nothing,
        sort_by::Symbol = :mean,
        descending::Bool = true,
        backend::Symbol = :text,
        return_string::Bool = false,
    )
    category_col = Symbol(category_col)
    metric_col_str = string(metric_col)

    # Check columns exist
    if !(category_col in Symbol.(names(df)))
        error("Column '$category_col' not found in DataFrame. Available columns: $(names(df))")
    end
    if !(metric_col_str in names(df))
        error("Column '$metric_col_str' not found in DataFrame. Available columns: $(names(df))")
    end

    # Group by category and compute statistics using combine
    grouped = groupby(df, category_col)

    summary_df = combine(grouped) do g
        values = g[!, metric_col_str]
        values = filter(!isnan, values)  # Remove NaN values

        n = length(values)
        if n == 0
            return DataFrame(
                count = 0,
                mean = NaN,
                std = NaN,
                median = NaN,
                min = NaN,
                max = NaN,
                q1 = NaN,
                q3 = NaN,
                iqr = NaN,
                outliers = 0,
            )
        end

        sorted_vals = sort(values)
        q1 = quantile(sorted_vals, 0.25)
        q3 = quantile(sorted_vals, 0.75)
        iqr_val = q3 - q1

        # Outliers: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        lower_bound = q1 - 1.5 * iqr_val
        upper_bound = q3 + 1.5 * iqr_val
        n_outliers = count(v -> v < lower_bound || v > upper_bound, values)

        return DataFrame(
            count = n,
            mean = mean(values),
            std = n > 1 ? std(values) : 0.0,
            median = median(values),
            min = minimum(values),
            max = maximum(values),
            q1 = q1,
            q3 = q3,
            iqr = iqr_val,
            outliers = n_outliers,
        )
    end

    # Sort by specified column
    sort_col = sort_by == :value ? category_col : sort_by
    if sort_col in Symbol.(names(summary_df))
        sort!(summary_df, sort_col; rev = descending)
    end

    # Generate title if not provided
    if isnothing(title)
        title = "Statistics by $(category_col) (metric: $(metric_col_str))"
    end

    # Format floats nicely
    float_formatter = (v, i, j) -> v isa AbstractFloat ? @sprintf("%.4f", v) : string(v)

    # Define column labels
    col_labels = [
        string(category_col),
        "Count",
        "Mean",
        "Std",
        "Median",
        "Min",
        "Max",
        "Q1",
        "Q3",
        "IQR",
        "Outliers",
    ]

    # Common pretty_table kwargs
    common_kwargs = (;
        column_labels = col_labels,
        title = title,
        formatters = [float_formatter],
        alignment = [:l, :r, :r, :r, :r, :r, :r, :r, :r, :r, :r],
    )

    # Add highlighters only for text backend (they're backend-specific)
    if backend == :text
        mean_col_idx = 3
        best_mean_idx = descending ? 1 : nrow(summary_df)
        worst_mean_idx = descending ? nrow(summary_df) : 1

        hl_best = TextHighlighter(
            (data, i, j) -> (j == mean_col_idx && i == best_mean_idx),
            crayon"green bold",
        )
        hl_worst = TextHighlighter(
            (data, i, j) -> (j == mean_col_idx && i == worst_mean_idx),
            crayon"red",
        )
        common_kwargs = (; common_kwargs..., highlighters = [hl_best, hl_worst])
    end

    # Generate output
    if return_string
        io = IOBuffer()
        pretty_table(io, summary_df; backend = backend, common_kwargs...)
        table_str = String(take!(io))
        return summary_df, table_str
    else
        pretty_table(summary_df; backend = backend, common_kwargs...)
        return summary_df, nothing
    end
end

"""
    continuous_comparison(df::DataFrame, continuous_col::Union{Symbol, String};
                          metric_col = "eval/ep_rew_mean",
                          n_bins = 5,
                          bin_method = :quantile,
                          title = nothing,
                          sort_by = :mean,
                          descending = true,
                          backend = :text,
                          return_string = false)

Analyze a continuous hyperparameter by binning it and computing statistics per bin.
Also computes correlation coefficients between the hyperparameter and metric.

# Arguments
- `df::DataFrame`: The DataFrame containing the data
- `continuous_col`: The column name of the continuous hyperparameter to analyze
- `metric_col`: The column name for the metric to analyze (default: "eval/ep_rew_mean")
- `n_bins`: Number of bins to create (default: 5)
- `bin_method`: How to create bins - `:quantile` (equal count) or `:uniform` (equal width)
- `title`: Optional title for the table (default: auto-generated)
- `sort_by`: Column to sort results by (:value, :count, :mean, :median, :std) (default: :mean)
- `descending`: Sort in descending order (default: true)
- `backend`: Output format - `:text`, `:markdown`, `:html`, or `:latex` (default: :text)
- `return_string`: If true, return the table as a string instead of printing (default: false)

# Returns
- `NamedTuple` with fields:
  - `binned_stats::DataFrame`: Statistics per bin (from categorical_comparison)
  - `correlation::NamedTuple`: Pearson and Spearman correlation coefficients
  - `bin_edges::Vector`: The bin edge values used
  - `table_string::Union{String, Nothing}`: Formatted table if return_string=true

# Example
```julia
continuous_comparison(res, :learning_rate)
continuous_comparison(res, "gamma", n_bins=4, bin_method=:uniform)
continuous_comparison(res, :dt, metric_col="train/ep_rew_mean", sort_by=:value, descending=false)
```
"""
function continuous_comparison(
        df::DataFrame,
        continuous_col::Union{Symbol, String};
        metric_col::Union{Symbol, String} = "eval/ep_rew_mean",
        n_bins::Int = 5,
        bin_method::Symbol = :quantile,
        title::Union{Nothing, String} = nothing,
        sort_by::Symbol = :mean,
        descending::Bool = true,
        backend::Symbol = :text,
        return_string::Bool = false,
    )
    continuous_col = Symbol(continuous_col)
    metric_col_str = string(metric_col)

    # Check columns exist
    if !(continuous_col in Symbol.(names(df)))
        error("Column '$continuous_col' not found in DataFrame. Available columns: $(names(df))")
    end
    if !(metric_col_str in names(df))
        error("Column '$metric_col_str' not found in DataFrame. Available columns: $(names(df))")
    end

    # Extract valid data (non-missing, non-NaN)
    valid_mask = .!ismissing.(df[!, continuous_col]) .& .!ismissing.(df[!, metric_col_str])
    x_all = df[valid_mask, continuous_col]
    y_all = df[valid_mask, metric_col_str]

    # Convert to Float64 and filter NaN
    x_data = Float64[]
    y_data = Float64[]
    for (x, y) in zip(x_all, y_all)
        if !isnan(x) && !isnan(y)
            push!(x_data, Float64(x))
            push!(y_data, Float64(y))
        end
    end

    n = length(x_data)
    if n < 3
        @warn "Insufficient data points ($n) for continuous comparison"
        return nothing
    end

    # Compute correlation coefficients
    pearson_corr = cor(x_data, y_data)
    spearman_corr = corspearman(x_data, y_data)

    # Compute bin edges
    if bin_method == :quantile
        # Equal count bins (quantiles)
        bin_edges = quantile(x_data, range(0, 1, length = n_bins + 1))
        # Ensure unique edges (can happen with repeated values)
        bin_edges = unique(bin_edges)
        actual_n_bins = length(bin_edges) - 1
        if actual_n_bins < n_bins
            @warn "Reduced to $actual_n_bins bins due to repeated values in data"
        end
    else
        # Equal width bins
        bin_edges = collect(range(minimum(x_data), maximum(x_data), length = n_bins + 1))
        actual_n_bins = n_bins
    end

    # Create bin labels and assign data to bins
    bin_labels = String[]
    for b in 1:(length(bin_edges) - 1)
        lower = round(bin_edges[b], sigdigits = 4)
        upper = round(bin_edges[b + 1], sigdigits = 4)
        if b == length(bin_edges) - 1
            push!(bin_labels, "[$lower, $upper]")  # Last bin is inclusive on both ends
        else
            push!(bin_labels, "[$lower, $upper)")
        end
    end

    # Assign each data point to a bin
    bin_assignments = String[]
    for x in x_data
        assigned = false
        for b in 1:(length(bin_edges) - 1)
            if b == length(bin_edges) - 1
                # Last bin: inclusive on both ends
                if x >= bin_edges[b] && x <= bin_edges[b + 1]
                    push!(bin_assignments, bin_labels[b])
                    assigned = true
                    break
                end
            else
                if x >= bin_edges[b] && x < bin_edges[b + 1]
                    push!(bin_assignments, bin_labels[b])
                    assigned = true
                    break
                end
            end
        end
        if !assigned
            # Edge case: assign to last bin
            push!(bin_assignments, bin_labels[end])
        end
    end

    # Create temporary DataFrame with binned column
    temp_df = DataFrame()
    temp_df[!, :_bin] = bin_assignments
    temp_df[!, metric_col_str] = y_data

    # Generate title if not provided
    if isnothing(title)
        title = "Statistics by $(continuous_col) (binned, metric: $(metric_col_str))"
    end

    # Print correlation info
    println("Correlation Analysis for $(continuous_col) vs $(metric_col_str):")
    println("  Pearson:  $(round(pearson_corr, digits = 4))")
    println("  Spearman: $(round(spearman_corr, digits = 4))")
    println()

    # Use categorical_comparison on the binned data
    # Sort by :value to keep bins in order (ascending by bin range)
    actual_sort_by = sort_by == :value ? :_bin : sort_by
    binned_stats, table_str = categorical_comparison(
        temp_df, :_bin;
        metric_col = metric_col_str,
        title = title,
        sort_by = actual_sort_by,
        descending = descending,
        backend = backend,
        return_string = return_string,
    )

    # Rename _bin column to the original column name for clarity
    rename!(binned_stats, :_bin => continuous_col)

    return (
        binned_stats = binned_stats,
        correlation = (pearson = pearson_corr, spearman = spearman_corr),
        bin_edges = bin_edges,
        table_string = table_str,
    )
end
