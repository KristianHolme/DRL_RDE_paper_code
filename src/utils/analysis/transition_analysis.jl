"""
Functions for analyzing and plotting transition experiment results.
"""

"""
    plot_best_transitions(res, shock_u_ps, transition_plot_dir; show_plots=false, parallel=false)

Plot the best (fastest) transition for each transition type.

# Arguments
- `res`: DataFrame with transition results
- `shock_u_ps`: Array of u_p values for each shock count
- `transition_plot_dir`: Directory to save plots
- `show_plots`: Whether to display plots (default: false for better performance)
- `parallel`: Whether to process in parallel (default: false, experimental)

# Example
```julia
# Fast mode (recommended) - saves plots but doesn't display them
plot_best_transitions(res, shock_u_ps, transition_plot_dir, show_plots=false)

# If you want to see plots as they're generated
plot_best_transitions(res, shock_u_ps, transition_plot_dir, show_plots=true)

# Experimental parallel mode (use with caution)
plot_best_transitions(res, shock_u_ps, transition_plot_dir, show_plots=false, parallel=true)
```
"""
function plot_best_transitions(res, shock_u_ps, transition_plot_dir; show_plots = false, parallel = false)
    println("\nPlotting best transitions:")
    println("=======================")

    # Collect all transition tasks
    transition_tasks = []

    for start_shocks in sort(unique(res.start_shocks))
        for target_shocks in sort(unique(res.target_shocks))
            if start_shocks != target_shocks
                subset = res[(res.start_shocks .== start_shocks) .& (res.target_shocks .== target_shocks) .& (.!ismissing.(res.transition_time)), :]
                if !isempty(subset)
                    best_idx = argmin(subset.transition_time)
                    best = subset[best_idx, :]

                    push!(
                        transition_tasks, (
                            start_shocks = start_shocks,
                            target_shocks = target_shocks,
                            best = best,
                        )
                    )
                end
            end
        end
    end

    # Process transitions
    if parallel && length(transition_tasks) > 1
        println("Processing $(length(transition_tasks)) transitions in parallel...")
        # Note: Parallel processing might have issues with Makie plots
        @warn "Parallel plotting is experimental and may cause issues with Makie"
        Threads.@threads for task in transition_tasks
            sleep(rand()) #to avoid printing races
            process_single_transition(task, shock_u_ps, transition_plot_dir, show_plots)
        end
    else
        println("Processing $(length(transition_tasks)) transitions sequentially...")
        for task in transition_tasks
            process_single_transition(task, shock_u_ps, transition_plot_dir, show_plots)
        end
    end

    return println("\nCompleted plotting all best transitions!")
end

"""
    process_single_transition(task, shock_u_ps, transition_plot_dir, show_plots)

Process a single transition task for plotting.

# Arguments
- `task`: NamedTuple with transition parameters (start_shocks, target_shocks, best)
- `shock_u_ps`: Array of u_p values for each shock count
- `transition_plot_dir`: Directory to save plots
- `show_plots`: Whether to display the plot after creation

# Returns
- `fig`: The created Makie figure
"""
function process_single_transition(task, shock_u_ps, transition_plot_dir, show_plots)
    start_shocks = task.start_shocks
    target_shocks = task.target_shocks
    best = task.best

    println("\nFastest $start_shocks → $target_shocks transition:")
    println("  Transition Time: ", round(best.transition_time, digits = 2))
    println("  Intermediate u_p: ", round(best.intermediate_u_p, digits = 3))
    println("  Intermediate Duration: ", round(best.intermediate_duration, digits = 2))

    # Run and plot the best transition
    fig, _ = run_transition(
        Dict(
            :start_shocks => start_shocks,
            :target_shocks => target_shocks,
            :intermediate_u_p => best.intermediate_u_p,
            :intermediate_duration => best.intermediate_duration
        ), shock_u_ps, save_result = false
    )

    # Save the plot
    plot_path = joinpath(transition_plot_dir, "transition_$(start_shocks)_to_$(target_shocks).png")
    safesave(plot_path, fig)

    # Only display if requested (can be slow and block execution)
    if show_plots
        display(fig)
    end

    return fig
end

const _TRANSITION_TIME_BLACK_LIST = ["transition_success_matrix", "transition_time_matrix"]

"""
    get_best_configs_per_target(res::DataFrame) -> DataFrame

Per-target aggregated selection: one best config per target shock count.
Groups by `target_shocks`, `intermediate_u_p`, `intermediate_duration`; computes
success rate and mean transition time; for each target selects the config with
maximum success rate, then minimum mean time.

Returns a DataFrame with one row per target (columns include `target_shocks`,
`intermediate_u_p`, `intermediate_duration`, `success_rate`, `mean_time`).
"""
function get_best_configs_per_target(res::DataFrame)
    group_cols = [:target_shocks, :intermediate_u_p, :intermediate_duration]
    agg = combine(
        groupby(res, group_cols),
        :transition_time => (tt -> mean(.!ismissing.(tt))) => :success_rate,
        :transition_time => (
            tt -> begin
                vals = collect(skipmissing(tt))
                return isempty(vals) ? NaN : mean(vals)
            end
        ) => :mean_time,
    )
    best_configs = DataFrame()
    for t in sort(unique(res.target_shocks))
        cand = agg[agg.target_shocks .== t, :]
        if nrow(cand) == 0
            continue
        end
        max_success = maximum(cand.success_rate)
        cand2 = cand[cand.success_rate .== max_success, :]
        best_idx = argmin(cand2.mean_time)
        push!(best_configs, cand2[best_idx, :])
    end
    return best_configs
end

"""
    get_best_configs_per_target(; data_dir = datadir("two_step_transition_time"), black_list = _TRANSITION_TIME_BLACK_LIST) -> DataFrame

Load transition results from `data_dir` and return best configs per target.
Use this when you do not already have a results DataFrame (e.g. in figure scripts).
"""
function get_best_configs_per_target(; data_dir = datadir("two_step_transition_time"), black_list = _TRANSITION_TIME_BLACK_LIST)
    res = DrWatson.collect_results(data_dir; black_list)
    return get_best_configs_per_target(res)
end

"""
    transition_time_matrix_per_target(res::DataFrame, best_configs::DataFrame) -> Matrix

Build 4×4 matrix of median transition times using the selected config per target.
`time_matrix[init, target]` is the median transition time from `init` to `target` shocks
using the best config for that target; diagonal is 0, missing entries are NaN.
"""
function transition_time_matrix_per_target(res::DataFrame, best_configs::DataFrame)
    time_matrix_per_target = fill(NaN, 4, 4)
    for i in 1:4
        time_matrix_per_target[i, i] = 0.0
    end
    for row in eachrow(best_configs)
        target = row.target_shocks
        cfg_subset = res[(res.target_shocks .== target) .& (res.intermediate_u_p .== row.intermediate_u_p) .& (res.intermediate_duration .== row.intermediate_duration), :]
        for init in setdiff(1:4, target)
            df_start = cfg_subset[cfg_subset.start_shocks .== init, :]
            if !isempty(df_start)
                tt = collect(skipmissing(df_start.transition_time))
                if !isempty(tt)
                    time_matrix_per_target[init, target] = median(tt)
                end
            end
        end
    end
    return time_matrix_per_target
end

"""
    get_best_config_for_target(best_configs::DataFrame, target_shocks::Int) -> DataFrameRow

Return the single row of `best_configs` for the given target shock count.
Use this instead of indexing by row number when using `best_configs` in scripts.
"""
function get_best_config_for_target(best_configs::DataFrame, target_shocks::Int)
    idx = findfirst(==(target_shocks), best_configs.target_shocks)
    if idx === nothing
        throw(ArgumentError("No best config for target_shocks = $target_shocks in best_configs"))
    end
    return best_configs[idx, :]
end

"""
    print_transition_time_summary(res::DataFrame)

Print per start→target summary: success rate, mean ± std, min and max transition time.
"""
function print_transition_time_summary(res::DataFrame)
    println("\nTransition Time Analysis:")
    println("=======================")
    for start_shocks in sort(unique(res.start_shocks))
        for target_shocks in sort(unique(res.target_shocks))
            if start_shocks != target_shocks
                subset = res[(res.start_shocks .== start_shocks) .& (res.target_shocks .== target_shocks), :]
                if !isempty(subset)
                    println("\nTransition from $start_shocks to $target_shocks shocks:")
                    successful = subset[.!ismissing.(subset.transition_time), :]
                    if !isempty(successful)
                        println("  Success Rate: ", round(nrow(successful) / nrow(subset) * 100, digits = 1), "%")
                        println(
                            "  Mean Transition Time: ", round(mean(successful.transition_time), digits = 2), " ± ",
                            round(std(successful.transition_time), digits = 2)
                        )
                        println("  Min Transition Time: ", round(minimum(successful.transition_time), digits = 2))
                        println("  Max Transition Time: ", round(maximum(successful.transition_time), digits = 2))
                    else
                        println("  No successful transitions")
                    end
                end
            end
        end
    end
    return nothing
end

"""
    transition_overview_matrices(res::DataFrame) -> (success_matrix::Matrix, time_matrix::Matrix)

Compute 4×4 success rate (percent) and mean transition time matrices for overview heatmaps.
Indices are [start_shocks, target_shocks]; off-diagonal only.
"""
function transition_overview_matrices(res::DataFrame)
    success_matrix = zeros(4, 4)
    time_matrix = zeros(4, 4)
    for (i, start) in enumerate(1:4)
        for (j, target) in enumerate(1:4)
            if start != target
                subset = res[(res.start_shocks .== start) .& (res.target_shocks .== target), :]
                if !isempty(subset)
                    success_matrix[i, j] = mean(.!ismissing.(subset.transition_time)) * 100
                    successful = subset[.!ismissing.(subset.transition_time), :]
                    if !isempty(successful)
                        time_matrix[i, j] = mean(successful.transition_time)
                    else
                        time_matrix[i, j] = NaN
                    end
                end
            end
        end
    end
    return success_matrix, time_matrix
end
