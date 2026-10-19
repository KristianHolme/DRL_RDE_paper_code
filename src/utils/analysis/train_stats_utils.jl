"""
    get_train_log_mvhist(log_path::String) -> MVHistory
extract the tensorboardlog file and convert it to a MVHistory.
"""
function get_train_log_mvhist(log_path::String)
    logger = TB.TBReader(log_path)
    mvhist = convert(VH.MVHistory, logger)
    return mvhist
end

get_train_log_mvhist(log_paths::Vector{String}) = get_train_log_mvhist.(log_paths)

"""
    plot_mvhist_key!(ax, histories::Vector{MVHistory}, key::String)
get data using key, and plot into ax using scatterlines!
"""
function plot_mvhist_key!(ax, histories::Vector{<:VH.MVHistory}, key::String)
    data = get.(histories, Symbol(key)) #vector of tuples
    iterations = getindex.(data, 1)
    values = getindex.(data, 2)
    scatterlines!.(ax, iterations, values)
    return ax
end

plot_mvhist_key!(ax, hist::VH.MVHistory, key) = plot_mvhist_key!(ax, [hist], key)

"""
    plot_mvhist_key(histories::Vector{MVHistory}, key; kwargs...)
plot the key from the histories into a figure.

# Arguments
- `histories::Vector{MVHistory}`: Vector of MVHistory objects
- `key::String`: Key to plot
- `kwargs...`: Keyword arguments to pass to the Axis constructor

# Returns
- `Figure`: Makie figure containing the plot
"""
function plot_mvhist_key(histories::Vector{<:VH.MVHistory}, key; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Iteration", ylabel = key; kwargs...)
    plot_mvhist_key!(ax, histories, key)
    return fig
end

plot_mvhist_key(hist::VH.MVHistory, key; kwargs...) = plot_mvhist_key([hist], key; kwargs...)

"""
    default_get_log_dir(path::String, logs_dir::String) -> String

Default function to convert result path to log directory path.
Strips the `.jld2` extension from the basename and joins with logs_dir.

# Arguments
- `path::String`: Path to the result file (e.g., from DataFrame `path` column)
- `logs_dir::String`: Base directory containing log directories

# Returns
- `String`: Path to the log directory
"""
function default_get_log_dir(path::String, logs_dir::String)
    return replace(joinpath(logs_dir, basename(path)), ".jld2" => "")
end

"""
    align_histories(history_data::Vector{Tuple{Vector, Vector}}) -> Tuple{Vector, Matrix}

Align multiple histories to common iteration points using intersection.

# Arguments
- `history_data::Vector{Tuple{Vector, Vector}}`: Vector of (iterations, values) tuples

# Returns
- `Tuple{Vector, Matrix}`: (common_iterations, aligned_values_matrix) where
  aligned_values_matrix has rows = iterations and cols = histories
"""
function align_histories(history_data::Vector{Tuple{Vector, Vector}})
    if isempty(history_data)
        return (Float64[], Matrix{Float64}(undef, 0, 0))
    end

    # Find intersection of all iteration sets
    common_iterations = Set(history_data[1][1])
    for (iterations, _) in history_data[2:end]
        common_iterations = intersect(common_iterations, Set(iterations))
    end

    if isempty(common_iterations)
        @warn "No common iterations found across histories"
        return (Float64[], Matrix{Float64}(undef, 0, length(history_data)))
    end

    # Sort common iterations
    common_iterations = sort(collect(common_iterations))

    # Create lookup dictionaries for each history
    lookup_dicts = [Dict(iter => val for (iter, val) in zip(iterations, values)) for (iterations, values) in history_data]

    # Build aligned matrix
    n_iterations = length(common_iterations)
    n_histories = length(history_data)
    aligned_matrix = Matrix{Float64}(undef, n_iterations, n_histories)

    for (col_idx, lookup_dict) in enumerate(lookup_dicts)
        for (row_idx, iter) in enumerate(common_iterations)
            aligned_matrix[row_idx, col_idx] = get(lookup_dict, iter, NaN)
        end
    end

    return (common_iterations, aligned_matrix)
end

"""
    aggregate_aligned_values(aligned_matrix::Matrix) -> Tuple{Vector, Vector}

Compute mean and std for aligned values across seeds.

# Arguments
- `aligned_matrix::Matrix`: Matrix with rows = iterations, cols = seeds

# Returns
- `Tuple{Vector, Vector}`: (mean_values, std_values)
"""
function aggregate_aligned_values(aligned_matrix::Matrix)
    n_iterations, n_seeds = size(aligned_matrix)
    mean_values = Vector{Float64}(undef, n_iterations)
    std_values = Vector{Float64}(undef, n_iterations)

    for i in 1:n_iterations
        row = aligned_matrix[i, :]
        # Filter out NaN values
        valid_values = filter(!isnan, row)
        if isempty(valid_values)
            mean_values[i] = NaN
            std_values[i] = NaN
        elseif length(valid_values) == 1
            mean_values[i] = valid_values[1]
            std_values[i] = 0.0
        else
            mean_values[i] = mean(valid_values)
            std_values[i] = std(valid_values)
        end
    end

    return (mean_values, std_values)
end

"""
    aggregate_histories_by_seed(
        df::DataFrame,
        key::String;
        group_cols::Vector{<:Union{Symbol, String}} = String[],
        logs_dir::Union{String, Nothing} = nothing,
        seed_col::Union{Symbol, String, Nothing} = nothing,
        get_log_dir_func::Function = default_get_log_dir,
        transform_func::Union{Function, Nothing} = nothing,
    ) -> Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}}

Aggregate training histories across seeds by grouping runs with identical configuration
(except seed) and computing mean and std for each iteration.

# Arguments
- `df::DataFrame`: DataFrame of results (e.g., from `collect_results`)
- `key::String`: Key to extract from MVHistory (e.g., "train/ep_rew_mean")
- `group_cols::Vector{<:Union{Symbol, String}}`: Columns to group by (excluding seed)
- `logs_dir::Union{String, Nothing}`: Base directory containing log directories. If `nothing`,
  will try to infer from DataFrame `path` column
- `seed_col::Union{Symbol, String, Nothing}`: Seed column name. If `nothing`, will auto-detect
- `get_log_dir_func::Function`: Function to convert result path to log directory path
- `transform_func::Union{Function, Nothing}`: Optional function to transform values before aggregation.
  Function signature: `(key::String, iterations::Vector, values::Vector, config::NamedTuple, row::DataFrameRow) -> (iterations::Vector, transformed_values::Vector)`.
  If `nothing`, no transformation is applied.

# Returns
- `Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}}`:
  Dictionary mapping configuration tuples to (mean_history, std_history)

# Examples
```julia
full_res = collect_results!(results_dir)
logs_dir = datadir("studies", experiment_name, "logs")

# Basic usage
aggregated = aggregate_histories_by_seed(
    full_res,
    "train/ep_rew_mean";
    group_cols = ["dt", "target_shock_count"],
    logs_dir = logs_dir,
)

# With transformation (e.g., scale by dt)
aggregated = aggregate_histories_by_seed(
    full_res,
    "env/ep_rew_mean";
    group_cols = ["dt", "target_shock_count"],
    logs_dir = logs_dir,
    transform_func = (key, iters, vals, config, row) -> begin
        if key == "env/ep_rew_mean" && haskey(config, :dt)
            return (iters, vals .* config.dt)
        else
            return (iters, vals)
        end
    end,
)

# Access aggregated histories
config = (dt = 0.1, target_shock_count = 3)
mean_hist = aggregated[config].mean
std_hist = aggregated[config].std
```
"""
function aggregate_histories_by_seed(
        df::DataFrame,
        key::String;
        group_cols::Vector{<:Union{Symbol, String}} = String[],
        logs_dir::Union{String, Nothing} = nothing,
        seed_col::Union{Symbol, String, Nothing} = nothing,
        get_log_dir_func::Function = default_get_log_dir,
        transform_func::Union{Function, Nothing} = nothing,
    )
    # Work with a copy to avoid modifying the original
    result_df = copy(df)

    # Convert group_cols to Symbols
    group_cols = Symbol.(group_cols)

    # Determine or create seed column (similar to group_by_seed)
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

    # Infer logs_dir from path if not provided
    if isnothing(logs_dir)
        if :path in Symbol.(names(result_df))
            # Try to infer from first path
            first_path = result_df[1, :path]
            if isa(first_path, String)
                # Assume logs are in a "logs" directory at the same level as "results"
                path_dir = dirname(first_path)
                parent_dir = dirname(path_dir)
                logs_dir = joinpath(parent_dir, "logs")
            else
                error("Cannot infer logs_dir from path column. Please provide logs_dir explicitly.")
            end
        else
            error("Cannot infer logs_dir. Please provide logs_dir explicitly or ensure DataFrame has 'path' column.")
        end
    end

    # Ensure path column exists
    if !(:path in Symbol.(names(result_df)))
        error("DataFrame must have a 'path' column to locate log directories")
    end

    # Group by configuration columns (excluding seed)
    if isempty(group_cols)
        # If no group columns specified, group all runs together
        grouped = [result_df]
        group_keys = [NamedTuple()]
    else
        # Filter to only valid grouping columns
        valid_group_cols = filter(c -> c in Symbol.(names(result_df)), group_cols)
        if isempty(valid_group_cols)
            @warn "No valid group columns found, grouping all runs together"
            grouped = [result_df]
            group_keys = [NamedTuple()]
        else
            grouped_df = groupby(result_df, valid_group_cols)
            grouped = [g for g in grouped_df]
            # Create NamedTuple keys for each group
            group_keys = [
                NamedTuple{Tuple(valid_group_cols)}([g[1, col] for col in valid_group_cols]) for g in grouped
            ]
        end
    end

    # Process each group
    result = Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}}()

    for (group, group_key) in zip(grouped, group_keys)
        # Get log directories for all runs in this group (and keep row association)
        log_path_row_pairs = Tuple{String, DataFrameRow}[]
        for row in eachrow(group)
            path = row.path
            if isa(path, String)
                log_path = get_log_dir_func(path, logs_dir)
                if isdir(log_path)
                    push!(log_path_row_pairs, (log_path, row))
                else
                    @warn "Log directory does not exist: $log_path"
                end
            end
        end

        if isempty(log_path_row_pairs)
            @warn "No valid log directories found for group $group_key, skipping"
            continue
        end

        # Load MVHistory objects
        histories = VH.MVHistory[]
        rows = DataFrameRow[]
        for (log_path, row) in log_path_row_pairs
            try
                hist = get_train_log_mvhist(log_path)
                push!(histories, hist)
                push!(rows, row)
            catch e
                @warn "Failed to load history from $log_path: $e"
            end
        end

        if isempty(histories)
            @warn "No valid histories loaded for group $group_key, skipping"
            continue
        end

        # Extract data for the specified key
        key_sym = Symbol(key)
        history_data = Tuple{Vector, Vector}[]

        # Debug: List available keys from first history
        if !isempty(histories)
            first_hist = histories[1]
            try
                # Try to get available keys using Base.keys
                if hasmethod(keys, (typeof(first_hist),))
                    available_keys = collect(keys(first_hist))
                    @debug "Available keys in first history for group $group_key: $available_keys"
                else
                    @debug "Could not list keys via keys() method for history type $(typeof(first_hist))"
                end
            catch debug_e
                @debug "Error while trying to inspect history structure: $debug_e"
            end
        end

        for (hist_idx, (hist, row)) in enumerate(zip(histories, rows))
            try
                data = get(hist, key_sym)
                if !isnothing(data)
                    iterations, values = data

                    # Apply transformation if provided
                    if !isnothing(transform_func)
                        iterations, values = transform_func(key, iterations, values, group_key, row)
                    end

                    push!(history_data, (iterations, values))
                    @debug "Successfully extracted key '$key' from history $hist_idx"
                else
                    @debug "Key '$key' returned nothing from history $hist_idx"
                end
            catch e
                @warn "Key '$key' not found in history $hist_idx: $e"
                # Try to list what keys are actually available
                try
                    if hasmethod(keys, (typeof(hist),))
                        available = collect(keys(hist))
                        @debug "Available keys in history $hist_idx: $available"
                    end
                catch inspect_e
                    @debug "Could not inspect history $hist_idx structure: $inspect_e"
                end
            end
        end

        if isempty(history_data)
            @warn "No data found for key '$key' in group $group_key, skipping"
            # Show available keys from first history if possible
            if !isempty(histories)
                try
                    first_hist = histories[1]
                    if hasmethod(keys, (typeof(first_hist),))
                        available = collect(keys(first_hist))
                        @debug "Available keys in first history of group $group_key: $available"
                    end
                catch
                end
            end
            continue
        end

        # Align histories to common iterations
        common_iterations, aligned_matrix = align_histories(history_data)

        if isempty(common_iterations)
            @warn "No common iterations found for group $group_key, skipping"
            continue
        end

        # Compute mean and std
        mean_values, std_values = aggregate_aligned_values(aligned_matrix)

        # Create new MVHistory objects with mean and std
        mean_hist = VH.MVHistory()
        std_hist = VH.MVHistory()

        for (iter, mean_val, std_val) in zip(common_iterations, mean_values, std_values)
            if !isnan(mean_val)
                push!(mean_hist, key_sym, iter, mean_val)
            end
            if !isnan(std_val)
                push!(std_hist, Symbol("$(key)_std"), iter, std_val)
            end
        end

        result[group_key] = (mean = mean_hist, std = std_hist)
    end

    # Clean up temporary columns if we created them
    if :_seed in Symbol.(names(result_df))
        select!(result_df, Not(:_seed))
    end

    return result
end

"""
    _is_continuous_colormap(colormap) -> Bool

Check if colormap should be treated as continuous (for colorbar) vs categorical (for legend).
"""
function _is_continuous_colormap(colormap)
    if isnothing(colormap)
        return false
    end
    # If it's a Symbol (like :viridis), treat as continuous
    if isa(colormap, Symbol)
        return true
    end
    # If it's a string that looks like a colormap name, treat as continuous
    if isa(colormap, String)
        return true
    end
    # Otherwise, assume categorical
    return false
end

"""
    _get_color_mapping(
        configs::Vector{NamedTuple},
        color_by::Symbol,
        colormap,
        colorscale;
        color_range = nothing,
    ) -> (colors::Vector, colorbar_info::Union{Nothing, NamedTuple})

Get color mapping for configurations based on color_by parameter and colormap.
Returns colors vector and optional colorbar info (min, max, colormap, scale) for continuous colormaps.
"""
function _get_color_mapping(
        configs::Vector{NamedTuple},
        color_by::Symbol,
        colormap,
        colorscale;
        color_range = nothing,
    )
    if isnothing(color_by) || !haskey(first(configs), color_by)
        # Fall back to default behavior
        wong_colors = Makie.wong_colors()
        colors = [wong_colors[mod1(i, length(wong_colors))] for i in 1:length(configs)]
        return (colors, nothing)
    end

    # Extract values for color_by parameter (we've already checked haskey above)
    color_values = [config[color_by] for config in configs]

    if _is_continuous_colormap(colormap)
        # Continuous colormap - map values to colors
        if isnothing(color_range)
            color_min = minimum(color_values)
            color_max = maximum(color_values)
        else
            color_min, color_max = color_range
        end
        cmap_vec = Makie.to_colormap(colormap)
        cmap_length = length(cmap_vec)

        # Handle edge case where all values are the same
        if color_min == color_max
            single_color = cmap_vec[1]
            colors = fill(single_color, length(color_values))
            colorbar_info = (
                min = color_min,
                max = color_max,
                colormap = colormap,
                label = string(color_by),
                scale = colorscale,
            )
            return (colors, colorbar_info)
        end

        scaled_min = colorscale(Float64(color_min))
        scaled_max = colorscale(Float64(color_max))

        # Normalize values to [0, 1] in scaled space, clamp, then map to colormap indices
        colors = [
            begin
                    scaled_val = colorscale(Float64(val))
                    normalized = clamp((scaled_val - scaled_min) / (scaled_max - scaled_min), 0.0, 1.0)
                    # Map normalized [0, 1] to colormap index [1, cmap_length]
                    idx = 1 + round(Int, normalized * (cmap_length - 1))
                    cmap_vec[idx]
                end
                for val in color_values
        ]

        colorbar_info = (
            min = color_min,
            max = color_max,
            colormap = colormap,
            label = string(color_by),
            scale = colorscale,
        )
        return (colors, colorbar_info)
    else
        # Categorical - use discrete colors
        unique_values = unique(color_values)
        wong_colors = Makie.wong_colors()
        value_to_color = Dict(
            val => wong_colors[mod1(i, length(wong_colors))]
                for (i, val) in enumerate(unique_values)
        )
        colors = [value_to_color[val] for val in color_values]
        # Return color legend info for categorical colors
        color_legend = Dict(val => value_to_color[val] for val in unique_values)
        return (colors, (legend = color_legend,))
    end
end

"""
    _get_marker_mapping(configs::Vector{NamedTuple}, marker_by::Union{Symbol, Nothing}, markers::Union{Vector, Nothing}) -> (markers::Vector, marker_legend::Dict)

Get marker mapping for configurations based on marker_by parameter.
Returns markers vector and dictionary mapping values to marker styles for legend.
"""
function _get_marker_mapping(configs::Vector{NamedTuple}, marker_by::Union{Symbol, Nothing}, markers::Union{Vector, Nothing})
    if isnothing(marker_by) || !haskey(first(configs), marker_by)
        return (fill(:circle, length(configs)), Dict())
    end

    # Extract values for marker_by parameter (we've already checked haskey above)
    marker_values = [config[marker_by] for config in configs]
    unique_values = unique(marker_values)

    # Default marker set
    default_markers = [:circle, :rect, :utriangle, :dtriangle, :diamond, :pentagon, :hexagon, :star4, :star5, :star8]

    # Use provided markers or default set
    marker_set = isnothing(markers) ? default_markers : markers

    # Map unique values to markers
    value_to_marker = Dict(
        val => marker_set[mod1(i, length(marker_set))]
            for (i, val) in enumerate(unique_values)
    )

    markers_vec = [value_to_marker[val] for val in marker_values]
    marker_legend = Dict(val => marker_set[mod1(i, length(marker_set))] for (i, val) in enumerate(unique_values))

    return (markers_vec, marker_legend)
end

"""
    plot_aggregated_histories!(
        ax,
        aggregated::Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}},
        key::String;
        config_filter = nothing,
        colors = nothing,
        label_func = nothing,
        color_by = nothing,
        marker_by = nothing,
        colormap = nothing,
        colorscale = identity,
        markers = nothing,
        kwargs...,
    )

Plot aggregated histories (mean ± std) into an existing Axis.

# Arguments
- `ax`: Makie Axis to plot into
- `aggregated::Dict`: Dictionary returned by `aggregate_histories_by_seed`
- `key::String`: Key to plot (e.g., "train/ep_rew_mean")
- `config_filter`: Optional function to filter which configurations to plot
- `colors`: Optional vector of colors. If `nothing`, uses color mapping based on `color_by` or defaults to `Makie.wong_colors()`
- `label_func`: Optional function to generate labels from configuration tuples
- `color_by::Union{Symbol, Nothing}`: Config parameter to use for color mapping (e.g., `:dt`)
- `marker_by::Union{Symbol, Nothing}`: Config parameter to use for marker mapping (e.g., `:target_shock_count`)
- `colormap`: Colormap to use. If `nothing`, uses categorical colors. If `:viridis` or similar, uses continuous colormap with colorbar
- `colorscale`: Transform for continuous color mapping (e.g., `log10`, `Makie.pseudolog10`)
- `color_range`: Optional `(min, max)` override for continuous color mapping
- `markers`: Optional vector of marker styles for discrete marker mapping
- `show_legend::Bool`: Whether to add a legend to the axis
- `return_legend::Bool`: Whether to return legend elements and labels
- `legend_kwargs`: Keyword arguments forwarded to `axislegend` when `show_legend = true`
- `kwargs...`: Additional keyword arguments passed to `lines!` for the mean line

# Returns
- `ax`: The Axis (for method chaining)
- `colorbar_info::Union{Nothing, NamedTuple}`: Colorbar info if continuous colormap was used, `nothing` otherwise
  (fields: `min`, `max`, `colormap`, `label`, `scale`)
- `legend_info::Union{Nothing, NamedTuple}`: Legend elements and labels if `return_legend = true`
"""
function plot_aggregated_histories!(
        ax,
        aggregated::Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}},
        key::String;
        plot_band = true,
        config_filter = nothing,
        colors = nothing,
        label_func = nothing,
        color_by = nothing,
        marker_by = nothing,
        colormap = nothing,
        colorscale = identity,
        color_range = nothing,
        markers = nothing,
        show_legend = true,
        return_legend = false,
        legend_kwargs = (;),
        kwargs...,
    )
    key_sym = Symbol(key)
    std_key_sym = Symbol("$(key)_std")

    # Filter configurations if filter function provided
    configs_to_plot = if isnothing(config_filter)
        collect(keys(aggregated))
    else
        filter(config_filter, collect(keys(aggregated)))
    end

    if isempty(configs_to_plot)
        @warn "No configurations to plot after filtering"
        return (ax, nothing)
    end

    # Get color mapping
    colorbar_info = nothing
    color_legend_dict = nothing
    if colors isa AbstractDict
        if isnothing(color_by)
            @warn "colors provided as a mapping but color_by is nothing; falling back to default colors"
            wong_colors = Makie.wong_colors()
            colors = [wong_colors[mod1(i, length(wong_colors))] for i in 1:length(configs_to_plot)]
        else
            colors_map = colors
            colors = Vector{Any}(undef, length(configs_to_plot))
            wong_colors = Makie.wong_colors()
            for (idx, config) in enumerate(configs_to_plot)
                if haskey(config, color_by)
                    value = config[color_by]
                    if haskey(colors_map, value)
                        colors[idx] = colors_map[value]
                    else
                        @warn "Color mapping missing for $(color_by) = $(value); using default color"
                        colors[idx] = wong_colors[mod1(idx, length(wong_colors))]
                    end
                else
                    @warn "Config missing $(color_by); using default color"
                    colors[idx] = wong_colors[mod1(idx, length(wong_colors))]
                end
            end
            color_legend_dict = Dict(colors_map)
        end
    elseif isnothing(colors)
        if !isnothing(color_by)
            colors, colorbar_info = _get_color_mapping(
                configs_to_plot,
                color_by,
                colormap,
                colorscale;
                color_range = color_range,
            )
            # Extract color legend if categorical (colorbar_info will be NamedTuple with :legend key)
            if isa(colorbar_info, NamedTuple) && haskey(colorbar_info, :legend)
                color_legend_dict = colorbar_info.legend
                colorbar_info = nothing  # Reset since it's categorical, not continuous
            end
        else
            wong_colors = Makie.wong_colors()
            colors = [wong_colors[mod1(i, length(wong_colors))] for i in 1:length(configs_to_plot)]
        end
    elseif length(colors) < length(configs_to_plot)
        @warn "Not enough colors provided, cycling through available colors"
        colors = [colors[mod1(i, length(colors))] for i in 1:length(configs_to_plot)]
    end

    # Get marker mapping
    markers_vec, marker_legend = _get_marker_mapping(configs_to_plot, marker_by, markers)

    # Get default marker size from kwargs for later use in legend
    default_marker_size = haskey(kwargs, :markersize) ? kwargs[:markersize] : 8

    # Default label function
    if isnothing(label_func)
        if !isnothing(color_by) && !isnothing(marker_by)
            # Combine both in label
            label_func = config -> begin
                color_val = haskey(config, color_by) ? config[color_by] : "?"
                marker_val = haskey(config, marker_by) ? config[marker_by] : "?"
                return "$(color_by)=$color_val, $(marker_by)=$marker_val"
            end
        elseif !isnothing(color_by)
            label_func = config -> begin
                val = haskey(config, color_by) ? config[color_by] : "?"
                return "$(color_by)=$val"
            end
        elseif !isnothing(marker_by)
            label_func = config -> begin
                val = haskey(config, marker_by) ? config[marker_by] : "?"
                return "$(marker_by)=$val"
            end
        else
            label_func = config -> string(config)
        end
    end

    # Plot each configuration
    for (idx, config) in enumerate(configs_to_plot)
        hists = aggregated[config]
        mean_hist = hists.mean
        std_hist = hists.std

        # Extract mean data
        mean_data = get(mean_hist, key_sym)
        if isnothing(mean_data)
            @warn "Key '$key' not found in mean history for config $config, skipping"
            continue
        end
        mean_iterations, mean_values = mean_data

        # Extract std data
        std_data = get(std_hist, std_key_sym)
        if isnothing(std_data)
            @warn "Key '$(std_key_sym)' not found in std history for config $config, skipping"
            continue
        end
        std_iterations, std_values = std_data

        # Align std to mean iterations (should be the same, but be safe)
        if std_iterations != mean_iterations
            # Create lookup for std values
            std_lookup = Dict(iter => val for (iter, val) in zip(std_iterations, std_values))
            std_values_aligned = [get(std_lookup, iter, 0.0) for iter in mean_iterations]
        else
            std_values_aligned = std_values
        end

        # Compute upper and lower bounds
        upper = mean_values .+ std_values_aligned
        lower = mean_values .- std_values_aligned

        # Plot error band
        if plot_band
            band!(
                ax,
                mean_iterations,
                lower,
                upper,
                color = (colors[idx], 0.2),
                label = nothing,
            )
        end

        # Plot mean line
        plot_kwargs = Dict{Symbol, Any}(kwargs)
        plot_kwargs[:color] = colors[idx]

        # Use scatterlines! if markers are requested, otherwise lines!
        if !isnothing(marker_by)
            marker_size = get(plot_kwargs, :markersize, 8)
            marker_shape = markers_vec[idx]

            # Downsample data for markers to avoid overcrowding
            n_points = length(mean_iterations)
            marker_interval = max(1, n_points ÷ 50)  # Show marker every ~50 points
            marker_indices = 1:marker_interval:n_points

            # Plot line for all points (no label - we'll create separate legends)
            lines!(
                ax,
                mean_iterations,
                mean_values,
                color = colors[idx],
                label = nothing,
            )

            # Plot markers at sampled points (no label - we'll create separate legends)
            scatter!(
                ax,
                mean_iterations[marker_indices],
                mean_values[marker_indices],
                color = colors[idx],
                marker = marker_shape,
                markersize = marker_size,
                label = nothing,
            )
        else
            lines!(
                ax,
                mean_iterations,
                mean_values;
                plot_kwargs...,
            )
        end
    end

    # Create separate legends for colors and markers
    # Collect all legend elements
    all_legend_elements = []
    all_legend_labels = []

    # Color legend (only for categorical colors, not continuous - colorbar handles continuous)
    if !isnothing(color_legend_dict)
        # Get a reference point from the data for placing legend elements
        if !isempty(configs_to_plot)
            first_config = configs_to_plot[1]
            first_hists = aggregated[first_config]
            first_mean_data = get(first_hists.mean, key_sym)
            if !isnothing(first_mean_data)
                ref_iter, ref_val = first_mean_data
                if !isempty(ref_iter)
                    ref_x = ref_iter[1]
                    ref_y = ref_val[1]

                    legend_values = collect(keys(color_legend_dict))
                    if all(val -> val isa Number, legend_values) ||
                            all(val -> val isa AbstractString, legend_values)
                        legend_values = sort(legend_values)
                    else
                        legend_values = sort(legend_values, by = val -> string(val))
                    end

                    for val in legend_values
                        color = color_legend_dict[val]
                        # Create invisible line for legend
                        line = lines!(
                            ax,
                            [ref_x, ref_x],
                            [ref_y, ref_y],
                            color = color,
                            visible = false,
                            label = string(val),
                        )
                        push!(all_legend_elements, line)
                        push!(all_legend_labels, string(val))
                    end
                end
            end
        end
    end

    # Marker legend
    if !isnothing(marker_by) && !isempty(marker_legend)
        marker_size = default_marker_size

        # Get a reference point from the data for placing legend elements
        if !isempty(configs_to_plot)
            first_config = configs_to_plot[1]
            first_hists = aggregated[first_config]
            first_mean_data = get(first_hists.mean, key_sym)
            if !isnothing(first_mean_data)
                ref_iter, ref_val = first_mean_data
                if !isempty(ref_iter)
                    ref_x = ref_iter[1]
                    ref_y = ref_val[1]

                    for (val, marker_shape) in marker_legend
                        # Create invisible scatter for legend
                        sc = scatter!(
                            ax,
                            [ref_x],
                            [ref_y],
                            marker = marker_shape,
                            markersize = marker_size,
                            color = :gray,
                            visible = false,
                            label = "$(marker_by) = $val",
                        )
                        push!(all_legend_elements, sc)
                        push!(all_legend_labels, "$(marker_by) = $val")
                    end
                end
            end
        end
    end

    legend_info = nothing
    if !isempty(all_legend_elements)
        if show_legend
            axislegend(
                ax,
                all_legend_elements,
                all_legend_labels;
                position = :rt,
                tellwidth = false,
                tellheight = false,
                legend_kwargs...,
            )
        end
        if return_legend
            legend_info = (elements = all_legend_elements, labels = all_legend_labels)
        end
    end

    if return_legend
        return (ax, colorbar_info, legend_info)
    end

    return (ax, colorbar_info)
end

"""
    plot_aggregated_histories(
        aggregated::Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}},
        key::String;
        kwargs...,
    ) -> Union{Figure, Tuple{Figure, Colorbar}}

Plot aggregated histories (mean ± std) in a new Figure.

# Arguments
- `aggregated::Dict`: Dictionary returned by `aggregate_histories_by_seed`
- `key::String`: Key to plot (e.g., "train/ep_rew_mean")
- `kwargs...`: Keyword arguments passed to `Axis` constructor and `plot_aggregated_histories!`
  (including `color_by`, `marker_by`, `colormap`, etc.)

# Returns
- `Figure`: Makie figure containing the plot (if no continuous colormap)
- `Tuple{Figure, Colorbar}`: Figure and Colorbar (if continuous colormap is used)

# Examples
```julia
aggregated = aggregate_histories_by_seed(full_res, "train/ep_rew_mean", group_cols=["dt", "target_shock_count"])

# Plot all configurations with discrete colors and markers
fig = plot_aggregated_histories(
    aggregated, "train/ep_rew_mean";
    color_by = :dt,
    marker_by = :target_shock_count,
)

# Plot with continuous colormap
fig, cbar = plot_aggregated_histories(
    aggregated, "env/ep_rew_mean";
    color_by = :dt,
    marker_by = :target_shock_count,
    colormap = :viridis,
)
```
"""
function plot_aggregated_histories(
        aggregated::Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}},
        key::String;
        size = (1200, 1000),
        kwargs...,
    )
    fig = Figure(size = size)

    # Extract axis-specific kwargs
    axis_kwargs = Dict{Symbol, Any}()
    plot_kwargs = Dict{Symbol, Any}()

    axis_only_kwargs = [:xlabel, :ylabel, :title, :xscale, :yscale, :xlims, :ylims, :xticks, :yticks]
    for (k, v) in kwargs
        if k in axis_only_kwargs
            axis_kwargs[k] = v
        else
            plot_kwargs[k] = v
        end
    end

    # Set defaults
    if !haskey(axis_kwargs, :xlabel)
        axis_kwargs[:xlabel] = "Iteration"
    end
    if !haskey(axis_kwargs, :ylabel)
        axis_kwargs[:ylabel] = key
    end

    # Create axis
    ax = Axis(fig[1, 1]; axis_kwargs...)

    # Plot
    ax_result, colorbar_info = plot_aggregated_histories!(ax, aggregated, key; plot_kwargs...)

    # Create colorbar if needed
    if !isnothing(colorbar_info)
        # Create colorbar directly from colormap
        cbar = Colorbar(
            fig[1, 2],
            colormap = colorbar_info.colormap,
            limits = (colorbar_info.min, colorbar_info.max),
            label = colorbar_info.label,
            scale = colorbar_info.scale,
        )
        return (fig, cbar)
    end

    return fig
end

"""
    plot_aggregated_histories_grid(
        aggregated::Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}},
        keys::AbstractVector{<:AbstractString};
        nrows = nothing,
        ncols = nothing,
        figure_kwargs = (;),
        axis_kwargs = (;),
        plot_kwargs = (;),
        legend_kwargs = (;),
        colorbar_kwargs = (;),
    ) -> NamedTuple

Create a multi-panel figure that compares multiple history keys from the same dataset,
with a shared legend and colorbar.

# Arguments
- `aggregated::Dict`: Dictionary returned by `aggregate_histories_by_seed`
- `keys::AbstractVector{<:AbstractString}`: Keys to plot in separate axes
- `nrows`, `ncols`: Layout specification. If one is `nothing`, it is inferred.
- `figure_kwargs`: Keyword arguments forwarded to `Figure`
- `axis_kwargs`: Keyword arguments forwarded to each `Axis`
- `plot_kwargs`: Keyword arguments forwarded to `plot_aggregated_histories!`
- `legend_kwargs`: Keyword arguments forwarded to `Legend`
- `colorbar_kwargs`: Keyword arguments forwarded to `Colorbar`

# Returns
- `NamedTuple`: `(fig = fig, legend = legend, colorbar = colorbar)`
"""
function plot_aggregated_histories_grid(
        aggregated::Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}},
        keys::AbstractVector{<:AbstractString};
        plot_band = true,
        nrows = nothing,
        ncols = nothing,
        figure_kwargs = (;),
        axis_kwargs = (;),
        plot_kwargs = (;),
        legend_kwargs = (;),
        colorbar_kwargs = (;),
    )
    n_plots = length(keys)
    if n_plots == 0
        @warn "No keys provided for plot_aggregated_histories_grid"
        return (fig = Figure(; figure_kwargs...), legend = nothing, colorbar = nothing)
    end

    if isnothing(nrows) && isnothing(ncols)
        ncols = ceil(Int, sqrt(n_plots))
        nrows = ceil(Int, n_plots / ncols)
    elseif isnothing(nrows)
        nrows = ceil(Int, n_plots / ncols)
    elseif isnothing(ncols)
        ncols = ceil(Int, n_plots / nrows)
    end

    fig = Figure(; figure_kwargs...)
    legend_info = nothing
    colorbar_info = nothing
    axes = Axis[]

    for (idx, key) in enumerate(keys)
        row = div(idx - 1, ncols) + 1
        col = mod1(idx, ncols)

        axis_local_kwargs = Dict{Symbol, Any}(axis_kwargs)
        if !haskey(axis_local_kwargs, :xlabel)
            axis_local_kwargs[:xlabel] = "Iteration"
        end
        if !haskey(axis_local_kwargs, :ylabel)
            axis_local_kwargs[:ylabel] = key
        end

        ax = Axis(fig[row, col]; axis_local_kwargs...)
        push!(axes, ax)

        if idx == 1
            _, plot_colorbar_info, plot_legend_info = plot_aggregated_histories!(
                ax,
                aggregated,
                key;
                show_legend = false,
                return_legend = true,
                plot_band = plot_band,
                plot_kwargs...,
            )
            legend_info = plot_legend_info
            colorbar_info = plot_colorbar_info
        else
            _, plot_colorbar_info = plot_aggregated_histories!(
                ax,
                aggregated,
                key;
                show_legend = false,
                return_legend = false,
                plot_band = plot_band,
                plot_kwargs...,
            )
            if !isnothing(plot_colorbar_info)
                if isnothing(colorbar_info)
                    colorbar_info = plot_colorbar_info
                else
                    if plot_colorbar_info.colormap != colorbar_info.colormap ||
                            plot_colorbar_info.label != colorbar_info.label ||
                            plot_colorbar_info.scale != colorbar_info.scale
                        @warn "Colorbar settings differ across plots; using the first one."
                    end
                    colorbar_info = (
                        min = min(colorbar_info.min, plot_colorbar_info.min),
                        max = max(colorbar_info.max, plot_colorbar_info.max),
                        colormap = colorbar_info.colormap,
                        label = colorbar_info.label,
                        scale = colorbar_info.scale,
                    )
                end
            end
        end
    end

    if length(axes) > 1
        linkyaxes!(axes...)
    end

    legend = nothing
    if !isnothing(legend_info) && !isempty(legend_info.elements)
        legend_title = nothing
        if !isnothing(color_by) && !_is_continuous_colormap(colormap)
            legend_title = string(color_by)
        end
        legend = Legend(
            fig[1:nrows, ncols + 1],
            legend_info.elements,
            legend_info.labels,
            legend_title;
            legend_kwargs...,
        )
    end

    colorbar = nothing
    if !isnothing(colorbar_info)
        colorbar = Colorbar(
            fig[1:nrows, ncols + 2],
            colormap = colorbar_info.colormap,
            limits = (colorbar_info.min, colorbar_info.max),
            label = colorbar_info.label,
            scale = colorbar_info.scale,
            colorbar_kwargs...,
        )
    end

    return (fig = fig, legend = legend, colorbar = colorbar)
end

"""
    plot_aggregated_histories_grid(
        aggregated_list::AbstractVector{<:Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}}},
        key::AbstractString;
        labels = nothing,
        nrows = nothing,
        ncols = nothing,
        figure_kwargs = (;),
        axis_kwargs = (;),
        plot_kwargs = (;),
        legend_kwargs = (;),
        colorbar_kwargs = (;),
    ) -> NamedTuple

Create a multi-panel figure that compares the same history key across multiple datasets,
with a shared legend and colorbar.

# Arguments
- `aggregated_list::AbstractVector{<:Dict}`: Collection of dictionaries returned by `aggregate_histories_by_seed`
- `key::AbstractString`: Key to plot in every axis
- `labels`: Optional labels for each dataset (used as axis titles)
- `nrows`, `ncols`: Layout specification. If one is `nothing`, it is inferred.
- `figure_kwargs`: Keyword arguments forwarded to `Figure`
- `axis_kwargs`: Keyword arguments forwarded to each `Axis`
- `plot_kwargs`: Keyword arguments forwarded to `plot_aggregated_histories!`
- `legend_kwargs`: Keyword arguments forwarded to `Legend`
- `colorbar_kwargs`: Keyword arguments forwarded to `Colorbar`

# Returns
- `NamedTuple`: `(fig = fig, legend = legend, colorbar = colorbar)`
"""
function plot_aggregated_histories_grid(
        aggregated_list::AbstractVector{<:Dict{NamedTuple, NamedTuple{(:mean, :std), Tuple{VH.MVHistory, VH.MVHistory}}}},
        key::AbstractString;
        plot_band = true,
        labels = nothing,
        nrows = nothing,
        ncols = nothing,
        figure_kwargs = (;),
        axis_kwargs = (;),
        plot_kwargs = (;),
        legend_kwargs = (;),
        colorbar_kwargs = (;),
    )
    n_plots = length(aggregated_list)
    if n_plots == 0
        @warn "No datasets provided for plot_aggregated_histories_grid"
        return (fig = Figure(; figure_kwargs...), legend = nothing, colorbar = nothing)
    end

    if !isnothing(labels) && length(labels) != n_plots
        @warn "Provided labels length does not match datasets; ignoring labels."
        labels = nothing
    end

    if isnothing(nrows) && isnothing(ncols)
        ncols = ceil(Int, sqrt(n_plots))
        nrows = ceil(Int, n_plots / ncols)
    elseif isnothing(nrows)
        nrows = ceil(Int, n_plots / ncols)
    elseif isnothing(ncols)
        ncols = ceil(Int, n_plots / nrows)
    end

    fig = Figure(; figure_kwargs...)
    legend_info = nothing
    colorbar_info = nothing

    plot_kwargs_local = plot_kwargs
    color_by = get(plot_kwargs, :color_by, nothing)
    colors = get(plot_kwargs, :colors, nothing)
    colormap = get(plot_kwargs, :colormap, nothing)
    color_range = get(plot_kwargs, :color_range, nothing)
    config_filter = get(plot_kwargs, :config_filter, nothing)

    if isnothing(colors) && !isnothing(color_by) && _is_continuous_colormap(colormap) &&
            isnothing(color_range)
        all_configs = NamedTuple[]
        for aggregated in aggregated_list
            configs = collect(keys(aggregated))
            if !isnothing(config_filter)
                configs = filter(config_filter, configs)
            end
            append!(all_configs, configs)
        end
        configs_with_color = filter(config -> haskey(config, color_by), all_configs)
        if isempty(configs_with_color)
            @warn "No configs have $(color_by); falling back to default colors"
        else
            color_values = [config[color_by] for config in configs_with_color]
            color_min = minimum(color_values)
            color_max = maximum(color_values)
            plot_kwargs_local = merge(plot_kwargs, (color_range = (color_min, color_max),))
        end
    elseif isnothing(colors) && !isnothing(color_by) && !_is_continuous_colormap(colormap)
        all_configs = NamedTuple[]
        for aggregated in aggregated_list
            configs = collect(keys(aggregated))
            if !isnothing(config_filter)
                configs = filter(config_filter, configs)
            end
            append!(all_configs, configs)
        end
        configs_with_color = filter(config -> haskey(config, color_by), all_configs)
        if !isempty(configs_with_color)
            color_values = [config[color_by] for config in configs_with_color]
            unique_values = unique(color_values)
            if all(v -> v isa Number, unique_values) || all(v -> v isa AbstractString, unique_values)
                unique_values = sort(unique_values)
            end
            wong_colors = Makie.wong_colors()
            value_to_color = Dict(
                val => wong_colors[mod1(i, length(wong_colors))]
                    for (i, val) in enumerate(unique_values)
            )
            plot_kwargs_local = merge(plot_kwargs, (colors = value_to_color,))
        else
            @warn "No configs have $(color_by); falling back to default colors"
        end
    end

    axes = Axis[]

    for (idx, aggregated) in enumerate(aggregated_list)
        row = div(idx - 1, ncols) + 1
        col = mod1(idx, ncols)

        axis_local_kwargs = Dict{Symbol, Any}(axis_kwargs)
        if !haskey(axis_local_kwargs, :xlabel)
            axis_local_kwargs[:xlabel] = "Iteration"
        end
        if !haskey(axis_local_kwargs, :ylabel)
            axis_local_kwargs[:ylabel] = key
        end
        if !haskey(axis_local_kwargs, :title)
            axis_local_kwargs[:title] = isnothing(labels) ? "Dataset $(idx)" : labels[idx]
        end

        ax = Axis(fig[row, col]; axis_local_kwargs...)
        push!(axes, ax)
        if idx == 1
            _, plot_colorbar_info, plot_legend_info = plot_aggregated_histories!(
                ax,
                aggregated,
                key;
                show_legend = false,
                return_legend = true,
                plot_band = plot_band,
                plot_kwargs_local...,
            )
            legend_info = plot_legend_info
            colorbar_info = plot_colorbar_info
        else
            _, plot_colorbar_info = plot_aggregated_histories!(
                ax,
                aggregated,
                key;
                show_legend = false,
                return_legend = false,
                plot_band = plot_band,
                plot_kwargs_local...,
            )
            if !isnothing(plot_colorbar_info)
                if isnothing(colorbar_info)
                    colorbar_info = plot_colorbar_info
                else
                    if plot_colorbar_info.colormap != colorbar_info.colormap ||
                            plot_colorbar_info.label != colorbar_info.label ||
                            plot_colorbar_info.scale != colorbar_info.scale
                        @warn "Colorbar settings differ across plots; using the first one."
                    end
                    colorbar_info = (
                        min = min(colorbar_info.min, plot_colorbar_info.min),
                        max = max(colorbar_info.max, plot_colorbar_info.max),
                        colormap = colorbar_info.colormap,
                        label = colorbar_info.label,
                        scale = colorbar_info.scale,
                    )
                end
            end
        end
    end

    if length(axes) > 1
        linkyaxes!(axes...)
        linkxaxes!(axes...)
    end

    legend = nothing
    if !isnothing(legend_info) && !isempty(legend_info.elements)
        legend_title = nothing
        if !isnothing(color_by) && !_is_continuous_colormap(colormap)
            legend_title = string(color_by)
        end
        legend = Legend(
            fig[1:nrows, ncols + 1],
            legend_info.elements,
            legend_info.labels,
            legend_title;
            legend_kwargs...,
        )
    end

    colorbar = nothing
    if !isnothing(colorbar_info)
        colorbar = Colorbar(
            fig[1:nrows, ncols + 2],
            colormap = colorbar_info.colormap,
            limits = (colorbar_info.min, colorbar_info.max),
            label = colorbar_info.label,
            scale = colorbar_info.scale,
            colorbar_kwargs...,
        )
    end

    return (fig = fig, legend = legend, colorbar = colorbar)
end
