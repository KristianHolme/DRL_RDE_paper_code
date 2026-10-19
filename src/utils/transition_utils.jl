const DELAY_TIME = 100.0f0
const DEFAULT_TRANSITION_TMAX = 600.0f0

transition_targets(context_strat::FixedTargetShockCountStrategy) = [context_strat.target_shock_count]
transition_targets(context_strat::RandomTargetShockCountStrategy) = 1:4
transition_targets(context_strat) = error("Unsupported context strategy: $(typeof(context_strat))")

function _detect_transition_legacy(data, prob, target_shocks, reward_stability_length, reward_threshold)
    us, = RDE.split_sol(data.states)
    #FIXME: not ideal, as dt can change for some action types
    dt = data.action_ts[2] - data.action_ts[1]
    #TODO this breaks if terminated, should we not eval if terminated? yes i think so
    saves_per_action = calculate_saves_per_action(data)
    if length(1:saves_per_action:lastindex(data.state_ts)) == length(data.action_ts) && !all(data.state_ts[1:saves_per_action:end] .≈ data.action_ts)
        @warn "state_ts and action_ts are not aligned, possible solver failure termination? Not detecting transition"
        return false, nothing, nothing
    end
    stability_length = floor(Int, reward_stability_length / dt)
    #this is not optimal, this allows for wrong shock count diffs in the ministeps
    counts = RDE.count_shocks.(us[1:saves_per_action:end], prob.x[2] - prob.x[1])
    end_shocks = counts[end]
    shock_diffs = diff(counts)


    if any(shock_diffs .!= 0) && end_shocks == target_shocks
        nonzero_inds = findall(!iszero, shock_diffs)
        acuired_final_shocks = last(nonzero_inds)
        found_transition = false
        search_idx = acuired_final_shocks

        while !found_transition && search_idx < length(data.rewards) - stability_length
            num_steps = length(data.rewards)
            search_endpoint = min(search_idx + stability_length, num_steps)
            rewards = data.rewards[search_idx:search_endpoint]
            # @info rewards
            if eltype(rewards) <: Vector
                @warn "Vector of vectors!!"
                rewards = minimum.(rewards)
            end
            if minimum(rewards) > reward_threshold
                return true, search_idx, search_endpoint #search_idx is index in rewards array, not in sync with action_ts
            else
                search_idx += 1
            end
        end
    else
        @debug "no schock change or target shock count reached"
    end

    return false, nothing, nothing
end

function detect_transition(data::PolicyRunData, prob::RDEProblem, target_shocks, reward_stability_length, reward_threshold)
    rewards = data.rewards
    n_rewards = length(rewards)
    @assert length(rewards) == length(data.action_ts)
    reward_times = vcat(data.action_ts[2:end], data.state_ts[end])
    for ix in 1:n_rewards
        current_time = reward_times[ix]
        forward_rewards = @view rewards[ix:end]
        forward_reward_times = @view reward_times[ix:end]
        # TODO: should it be findlast, ..., <,... instead? to just check indices inside window, if dt is large, the endpoint maybe outside window
        search_end_ix = findlast(rt -> rt < current_time + reward_stability_length, forward_reward_times)
        # search_end_ix = findfirst(rt -> rt > current_time + reward_stability_length, forward_reward_times)
        if isnothing(search_end_ix) || search_end_ix == 1 #dont allow single point window
            #should never happen, as we have at least one reward
            isnothing(search_end_ix) && @warn "No search end index found, continuing"
            continue
        end
        rewards_in_window = @view forward_rewards[1:search_end_ix]
        if minimum(rewards_in_window) > reward_threshold
            return true, ix, ix + search_end_ix - 1
        end
    end

    return false, nothing, nothing
end


"""
    detect_transition(data, target_shocks; reward_stability_length=20, reward_threshold=0.9f0)

Detect when a stable transition to a target number of shocks occurs. Default params intended for StabilityReward.
A transition is detected when
# Arguments
- `data`: RunData object containing simulation results
- `target_shocks`: Target number of shocks to detect transition to
- `reward_stability_length`: Number of timesteps required with high reward to confirm stability
- `reward_threshold`: Minimum reward value required for stability

# Returns
- `found`: Boolean indicating if a stable transition was found
- `transition_idx`: Index where the transition was found (if found)
- `stability_end_idx`: Index where the stability window ends (if found)
"""
function detect_transition(
        data::PolicyRunData,
        prob::RDEProblem,
        target_shocks;
        reward_stability_length = 20,
        reward_threshold = 0.9f0,
        mode::Symbol = :new
    )
    if mode == :legacy
        return _detect_transition_legacy(data, prob, target_shocks, reward_stability_length, reward_threshold)
    elseif mode == :new
        return detect_transition(data, prob, target_shocks, reward_stability_length, reward_threshold)
    else
        error("Invalid mode: $mode")
    end
end

"""
    run_transition(config, shock_u_ps; save_result=true, make_plot=true)

Run a transition experiment with the given configuration.

# Arguments
- `config`: Dictionary with keys :start_shocks, :target_shocks, :intermediate_u_p, :intermediate_duration
- `shock_u_ps`: Array of u_p values for each shock count
- `save_result`: Whether to save the result to a file
- `make_plot`: Whether to create and return a plot

# Returns
- If make_plot=true: (figure, result)
- If make_plot=false: result
"""
function run_transition(config, shock_u_ps; save_result = true, make_plot = true, mode::Symbol = :new)
    start_shocks = config[:start_shocks]
    target_shocks = config[:target_shocks]
    intermediate_u_p = config[:intermediate_u_p]
    intermediate_duration = config[:intermediate_duration]

    @debug "Running transition" start_shocks = start_shocks target_shocks = target_shocks intermediate_u_p = intermediate_u_p intermediate_duration = intermediate_duration

    u_p_start = shock_u_ps[start_shocks]
    u_p_end = shock_u_ps[target_shocks]
    transition_start = DELAY_TIME
    params = RDEParam(tmax = DEFAULT_TRANSITION_TMAX)
    env = RDEEnv(
        params, dt = 0.5f0, action_strat = DirectScalarPressureAction(),
        observation_strat = StateObservation(),
        reset_strategy = NShock(start_shocks),
        reward_strat = StabilityReward(1.0f0),
        context_strat = FixedTargetShockCountStrategy(target_shocks)
    )
    policy = StepwiseRDEPolicy(
        env, [0.0f0, transition_start, transition_start + intermediate_duration],
        [u_p_start, intermediate_u_p, u_p_end]
    )
    data = run_policy(policy, env)
    reward_times = vcat(data.action_ts[2:end], data.state_ts[end])
    # Create the plot if requested
    fig = nothing
    if make_plot
        fig = plot_shifted_history(data, env.prob.x)
    end

    # Detect transition
    found, transition_idx, stability_end_idx = detect_transition(data, env.prob, target_shocks; mode)
    if found
        transition_time = reward_times[transition_idx] - transition_start
        @debug "Transition time: $transition_time"
        result = Dict(
            "start_shocks" => start_shocks,
            "target_shocks" => target_shocks,
            "transition_time" => transition_time,
            "intermediate_u_p" => intermediate_u_p,
            "intermediate_duration" => intermediate_duration
        )

        # Add transition markers to plot if created
        if make_plot
            for content in fig.content
                if content isa Axis
                    vlines!(content, reward_times[transition_idx], color = :red, linestyle = :dash)
                    vlines!(content, reward_times[stability_end_idx], color = :red, linestyle = :dash, alpha = 0.5)
                end
            end
        end
    else
        @debug "No transition found"
        result = Dict(
            "start_shocks" => start_shocks,
            "target_shocks" => target_shocks,
            "transition_time" => missing,
            "intermediate_u_p" => intermediate_u_p,
            "intermediate_duration" => intermediate_duration
        )
    end

    if save_result
        safesave(datadir("two_step_transition_time", experiment_savename(config, "jld2")), result)
    end

    return make_plot ? (fig, result) : result
end


function evaluate_transition(
        experiment::AbstractExperiment, model_path::String,
        start_shocks::Int,
        target_shocks::Int,
        n_trials::Int = 5;
        mode::Symbol = :new,
        reward_stability_length::Float32 = 20.0f0,
        reward_threshold::Float32 = 0.9f0,
        lock_dt::Union{Nothing, AbstractFloat} = nothing,
    )
    jl_policy = get_jl_policy(experiment, model_path)
    transition_times = Float32[]
    success_count = 0
    control_dt = experiment.env_config.dt
    @reset experiment.env_config.reward_strat = StabilityReward(1.0f0)
    @reset experiment.env_config.context_strat = FixedTargetShockCountStrategy(target_shocks)
    @reset experiment.env_config.reset_strategy = ShiftReset(NShock(start_shocks))
    if lock_dt !== nothing
        @reset experiment.env_config.dt = Float32(lock_dt)
    end

    for _ in 1:n_trials
        env = create_env(experiment.env_config)
        # @info env.reward_strat, env.reward

        # Run policy
        delayed_policy = DelayedPolicy(jl_policy, DELAY_TIME, env)
        policy = if lock_dt !== nothing && Float32(lock_dt) < Float32(control_dt)
            HeldActionPolicy(delayed_policy, env, control_dt)
        else
            delayed_policy
        end
        data = run_policy(policy, env)
        # Detect transition
        found, transition_idx, _ = detect_transition(
            data, env.prob, target_shocks;
            reward_stability_length, reward_threshold, mode
        )
        if found
            reward_times = vcat(data.action_ts[2:end], data.state_ts[end])
            transition_time = reward_times[transition_idx] - DELAY_TIME
            push!(transition_times, transition_time)
            success_count += 1
        end
    end

    if lock_dt !== nothing
        @reset experiment.env_config.dt = control_dt
    end

    success_rate = success_count / n_trials
    mean_time = isempty(transition_times) ? missing : mean(transition_times)
    std_time = isempty(transition_times) ? missing : std(transition_times)

    return (
        success_rate = success_rate, mean_time = mean_time, std_time = std_time,
        transition_times = transition_times,
    )
end


function evaluate_transition_matrices(
        experiment::AbstractExperiment,
        model_path::String;
        targets = 1:4,
        mode::Symbol = :new,
        kwargs...
    )
    success_matrix = zeros(4, 4)
    time_matrix = zeros(4, 4)
    results_by_pair = Dict{Tuple{Int, Int}, NamedTuple}()
    for target_shocks in targets
        for start_shocks in setdiff(1:4, target_shocks)
            results = evaluate_transition(
                experiment, model_path,
                start_shocks,
                target_shocks;
                mode,
                kwargs...
            )
            success_matrix[start_shocks, target_shocks] = results.success_rate * 100
            time_matrix[start_shocks, target_shocks] = ismissing(results.mean_time) ? NaN : results.mean_time
            results_by_pair[(start_shocks, target_shocks)] = results
        end
    end
    return success_matrix, time_matrix, results_by_pair
end


function transition_matrices(
        experiment::AbstractExperiment,
        model_path::String;
        tmax = DEFAULT_TRANSITION_TMAX,
        kwargs...
    )
    @reset experiment.env_config.tmax = tmax
    @reset experiment.eval_config.env_config.tmax = tmax
    return evaluate_transition_matrices(
        experiment, model_path;
        kwargs...
    )
end


function plot_transition_matrices(success_matrix, time_matrix)
    fig = Figure(size = (1200, 800))

    # Success rate heatmap
    ax1 = Axis(
        fig[1, 1],
        title = "Transition Success Rate (RL)",
        xlabel = "Target Shock Count",
        ylabel = "Start Shock Count"
    )

    hm1 = heatmap!(
        ax1, 1:4, 1:4, success_matrix',
        colormap = :viridis,
        colorrange = (0, 100)
    )
    Colorbar(fig[1, 2], hm1, label = "Success Rate (%)")

    # Mean transition time heatmap
    ax2 = Axis(
        fig[2, 1],
        title = "Mean Transition Time (RL)",
        xlabel = "Target Shock Count",
        ylabel = "Start Shock Count"
    )

    hm2 = heatmap!(
        ax2, 1:4, 1:4, time_matrix',
        colormap = :viridis,
        nan_color = :black
    )
    Colorbar(fig[2, 2], hm2, label = "Mean Transition Time")

    return fig
end

# Utility function to abbreviate words in a string
function abbreviate_words(s::AbstractString)
    # Replace each word (sequence of letters) with its first letter
    # This regex matches sequences of letters and replaces them with just the first letter
    return replace(s, r"[A-Za-z][A-Za-z]+" => m -> string(first(m)))
end

function plot_and_save_transition(env, data, start_shocks, target_shocks, transition_plot_dir; title = "", mode::Symbol = :new)
    # Create the plot
    fig = plot_shifted_history(
        data,
        env.prob.x,
        target_shock_count = target_shocks,
        title = "Transition from $start_shocks to $target_shocks, $title"
    )
    # Add transition markers if found
    found, transition_idx, stability_end_idx = detect_transition(data, env.prob, target_shocks; mode)
    if found
        for content in fig.content
            if content isa Axis
                reward_times = vcat(data.action_ts[2:end], data.state_ts[end])
                vlines!(content, reward_times[transition_idx], color = :red, linestyle = :dash)
                vlines!(content, reward_times[stability_end_idx], color = :red, linestyle = :dash, alpha = 0.5)
            end
        end
    end
    # Check if file already exists
    save_path = joinpath(transition_plot_dir, "transition_$(start_shocks)_to_$(target_shocks).png")
    if isfile(save_path)
        # Get timestamp for old file
        timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        # Rename old file with timestamp
        new_name = joinpath(transition_plot_dir, "transition_$(start_shocks)_to_$(target_shocks)_$timestamp.png")
        mv(save_path, new_name)
    end
    save(joinpath(transition_plot_dir, "transition_$(start_shocks)_to_$(target_shocks).png"), fig)
    return fig
end

function analyze_transitions(
        study_name; reevaluate = false, single_model = false,
        tmax = DEFAULT_TRANSITION_TMAX, mode::Symbol = :new, filter_function = x -> true,
        name_suffix = "", display_plots = false, kwargs...
    )
    results_dir = datadir("studies", study_name, "results")
    res = DrWatson.collect_results!(results_dir)
    res = filter(filter_function, res)
    if reevaluate
        reevaluate_results!(res)
    end

    # Create directory for transition plots if it doesn't exist
    transition_plot_dir = plotsdir(study_name, name_suffix)
    if !isdir(transition_plot_dir)
        mkpath(transition_plot_dir)
    end
    success_matrix = zeros(4, 4)
    time_matrix = zeros(4, 4)
    results_by_pair = Dict{Tuple{Int, Int}, NamedTuple}()
    best_row_by_target = Dict{Int, Any}()
    model_path_by_target = Dict{Int, String}()
    model_basename_by_target = Dict{Int, String}()

    if single_model
        best_idx = argmax(res."eval/ep_rew_mean")
        best_experiment_row = res[best_idx, :]
        saved_model_path = best_experiment_row.model_path
        model_path_basename = basename(saved_model_path)
        model_path = datadir("studies", study_name, "models", model_path_basename)
        experiment = best_experiment_row.experiment
        @reset experiment.env_config.tmax = tmax
        @reset experiment.eval_config.env_config.tmax = tmax
        success_matrix, time_matrix, results_by_pair = evaluate_transition_matrices(
            experiment,
            model_path;
            targets = 1:4,
            mode,
            kwargs...
        )
        for target_shocks in 1:4
            best_row_by_target[target_shocks] = best_experiment_row
            model_path_by_target[target_shocks] = model_path
            model_basename_by_target[target_shocks] = model_path_basename
        end
    else
        ## For each target shock count, find the best model and evaluate it
        for target_shocks in 1:4
            target_results = res[res.target_shock_count .== target_shocks, :]
            if !isempty(target_results)
                best_idx = argmax(target_results."eval/ep_rew_mean")
                best_experiment_row = target_results[best_idx, :]

                # Load the model
                saved_model_path = best_experiment_row.model_path
                model_path_basename = basename(saved_model_path)
                model_path = datadir("studies", study_name, "models", model_path_basename)

                experiment = best_experiment_row.experiment
                @reset experiment.env_config.tmax = tmax
                @reset experiment.eval_config.env_config.tmax = tmax

                target_success_matrix, target_time_matrix, target_results_by_pair = evaluate_transition_matrices(
                    experiment,
                    model_path;
                    targets = [target_shocks],
                    mode,
                    kwargs...
                )
                success_matrix[:, target_shocks] = target_success_matrix[:, target_shocks]
                time_matrix[:, target_shocks] = target_time_matrix[:, target_shocks]
                merge!(results_by_pair, target_results_by_pair)
                best_row_by_target[target_shocks] = best_experiment_row
                model_path_by_target[target_shocks] = model_path
                model_basename_by_target[target_shocks] = model_path_basename
            end
        end
    end

    for target_shocks in 1:4
        if !haskey(best_row_by_target, target_shocks)
            continue
        end
        best_experiment_row = best_row_by_target[target_shocks]
        model_path = model_path_by_target[target_shocks]
        model_path_basename = model_basename_by_target[target_shocks]
        println("\nEvaluating best model for target $target_shocks shocks:")
        println("Model path: $model_path")
        println("eval/ep_rew_mean: $(best_experiment_row."eval/ep_rew_mean")")
        for start_shocks in setdiff(1:4, target_shocks)
            pair_key = (start_shocks, target_shocks)
            if !haskey(results_by_pair, pair_key)
                continue
            end
            results = results_by_pair[pair_key]
            println("\nTransition from $start_shocks to $target_shocks shocks:")
            println("  Success Rate: $(round(results.success_rate * 100, digits = 1))%")
            if !ismissing(results.mean_time)
                println("  Mean Transition Time: $(round(results.mean_time, digits = 2)) ± $(round(results.std_time, digits = 2))")
            end

            # If successful, plot best transition
            if results.success_rate > 0
                display_plots && @info "Plotting transition"
                experiment = best_experiment_row.experiment
                @reset experiment.env_config.reward_strat = StabilityReward(1.0f0)
                @reset experiment.env_config.reset_strategy = ShiftReset(NShock(start_shocks))
                env = create_env(experiment.env_config)
                policy = get_jl_policy(experiment, model_path)
                delayed_policy = DelayedPolicy(policy, DELAY_TIME, env)
                data = run_policy(delayed_policy, env)
                if env.terminated
                    @warn "Environment terminated, aborting transition plot"
                else
                    # Use abbreviated title
                    fig = plot_and_save_transition(
                        env,
                        data,
                        start_shocks,
                        target_shocks,
                        transition_plot_dir;
                        title = "model: " * abbreviate_words(model_path_basename),
                        mode
                    )
                    if display_plots
                        display(fig)
                    end
                end
            end
        end
    end

    ## Create visualization of transition results
    fig = plot_transition_matrices(success_matrix, time_matrix)
    if display_plots
        display(fig)
    end
    save(plotsdir(study_name, "transition_rl_analysis_overview" * name_suffix * ".png"), fig)
    safesave(
        datadir("studies", study_name, "transition_rl_analysis_matrices" * name_suffix * ".jld2"), Dict(
            "transition_success_matrix" => success_matrix,
            "transition_time_matrix" => time_matrix
        )
    )
    return success_matrix, time_matrix
end

function current_host_model_path(row)
    model_path_basename = basename(row.model_path)
    model_path = datadir("studies", row.experiment.study, "models", model_path_basename)
    return model_path
end

"""
    analyze_all_transitions(experiment_name; reevaluate = false, force = false, ...)

Compute and save per-row transition matrices under
`datadir("studies", experiment_name, "transition_analyses", basename(row.path))`.

When `force` is `false` (default), rows whose output file already exist are skipped.
When `force` is `true`, all rows are recomputed and files overwritten.
"""
function analyze_all_transitions(
        experiment_name; reevaluate = false,
        force::Bool = false,
        filter_function = x -> true,
        kwargs...
    )

    @debug "collecting results"
    results_dir = datadir("studies", experiment_name, "results")
    res = DrWatson.collect_results!(results_dir)
    @debug "filtering results, rows before filtering: $(nrow(res))"
    res = filter(filter_function, res)
    @debug "rows after filtering: $(nrow(res))"

    if nrow(res) == 0
        @info "No result rows in experiment; skipping transition analysis" experiment_name
        return res
    end

    if reevaluate
        reevaluate_results!(res)
    end

    save_dir = datadir("studies", experiment_name, "transition_analyses")
    if !isdir(save_dir)
        mkpath(save_dir)
    end

    rows_vec = collect(eachrow(res))
    rows_to_process = if force
        rows_vec
    else
        filter(row -> !isfile(joinpath(save_dir, basename(row.path))), rows_vec)
    end
    if !force
        n_existing = length(rows_vec) - length(rows_to_process)
        if n_existing > 0
            @info "Skipping existing transition analysis files" experiment_name n_existing
        end
        if isempty(rows_to_process)
            @info "All transition analysis files already exist; nothing to compute" experiment_name
            return res
        end
    end

    @showprogress pmap(rows_to_process) do row
        @debug "Analyzing transition for row: $(row.path), thread: $(myid())"
        success_matrix, time_matrix, _ = transition_matrices(
            row.experiment,
            current_host_model_path(row);
            targets = transition_targets(row.experiment.env_config.context_strat),
            kwargs...
        )
        @debug "transition matrices computed, thread: $(myid())"
        target_shock_count = :target_shock_count in propertynames(row) ? row.target_shock_count : missing
        save_path = joinpath(save_dir, basename(row.path))
        @debug "saving transition matrices, thread: $(myid())"
        safesave(
            save_path,
            Dict(
                "transition_success_matrix" => success_matrix,
                "transition_time_matrix" => time_matrix,
                "row_path" => row.path,
                "model_path" => row.model_path,
                "target_shock_count" => target_shock_count
            )
        )
    end

    return res
end


"""
    add_transition_matrices!(df, experiment_name)

Load saved transition matrices for each row in `df` and add two columns:
`transition_success_matrix` and `transition_time_matrix`.

Matrices are loaded from:
`datadir("studies", experiment_name, "transition_analyses", basename(row.path))`.

"""
function add_transition_matrices!(
        df::AbstractDataFrame,
        experiment_name::AbstractString
    )
    save_dir = datadir("studies", experiment_name, "transition_analyses")
    n_rows = nrow(df)
    success_matrices = Vector{Any}(undef, n_rows)
    time_matrices = Vector{Any}(undef, n_rows)

    for (idx, row) in enumerate(eachrow(df))
        save_path = joinpath(save_dir, basename(row.path))
        if !isfile(save_path)
            error("Missing transition analysis file: $save_path")
        end
        data = load(save_path)
        if !haskey(data, "transition_success_matrix") || !haskey(data, "transition_time_matrix")
            error("Missing transition matrices in: $save_path")
        end
        success_matrices[idx] = data["transition_success_matrix"]
        time_matrices[idx] = data["transition_time_matrix"]
    end

    df.transition_success_matrix = success_matrices
    df.transition_time_matrix = time_matrices
    return df
end


"""
    find_best_model_index_for_target(mean_success_rates, mean_times, targets, target_shock_count)

Find the index of the best model for a given target shock count.

Selects the model with the highest mean success rate for the target. If multiple models tie,
selects the one with the lowest mean transition time.

# Arguments
- `mean_success_rates`: Vector of mean success rates for each model
- `mean_times`: Vector of mean transition times for each model
- `targets`: Vector of target shock count vectors, one per model
- `target_shock_count`: The target shock count to find the best model for (1-4)

# Returns
- `Int`: Index into the original vectors for the best model
"""
function best_index_from_rates(
        mean_success_rates::Vector,
        mean_times::Vector
    )
    if isempty(mean_success_rates)
        error("No models available to select from.")
    end

    best_indices = findall(x -> x == maximum(mean_success_rates), mean_success_rates)
    if length(best_indices) > 1
        finite_indices = best_indices[isfinite.(mean_times[best_indices])]
        if isempty(finite_indices)
            best_index = best_indices[1]
        else
            best_index = finite_indices[argmin(mean_times[finite_indices])]
        end
    else
        best_index = best_indices[1]
    end

    return best_index
end

function find_best_model_index_for_target(
        mean_success_rates::Vector,
        mean_times::Vector,
        targets::Vector{Vector{Int}},
        target_shock_count::Int
    )
    qualified_models = findall(x -> target_shock_count in x, targets)
    qualified_success_rates = mean_success_rates[qualified_models]
    qualified_times = mean_times[qualified_models]
    best_index = best_index_from_rates(qualified_success_rates, qualified_times)
    return qualified_models[best_index]
end

"""
    select_best_transition_matrices(success_matrices, time_matrices)

Select the best matrix per entry by prioritizing higher success rate, then lower transition time.
Transition times of 0.0 are treated as infinite (unset entries).
"""
function select_best_transition_matrices(
        success_matrices::Vector{<:AbstractMatrix},
        time_matrices::Vector{<:AbstractMatrix},
        targets::Vector{Vector{Int}}
    )
    @assert length(success_matrices) == length(time_matrices)
    @assert !isempty(success_matrices)

    mean_success_rates = map(eachindex(success_matrices)) do k
        model_targets = targets[k]
        sum(success_matrices[k][:, model_targets]) / (length(model_targets) * 3)
    end
    mean_times = map(eachindex(time_matrices)) do k
        model_targets = targets[k]
        sum(time_matrices[k][:, model_targets]) / (length(model_targets) * 3)
    end
    n_rows, n_cols = size(success_matrices[1])
    best_success = zeros(n_rows, n_cols)
    best_time = zeros(n_rows, n_cols)
    best_indices = Vector{Int}(undef, 4)
    all_targets = reduce(union, targets)
    for target in all_targets
        best_model_index = find_best_model_index_for_target(
            mean_success_rates, mean_times, targets, target
        )
        best_indices[target] = best_model_index
        best_success[:, target] = success_matrices[best_model_index][:, target]
        best_time[:, target] = time_matrices[best_model_index][:, target]
    end
    return best_success, best_time, best_indices
end


"""
    aggregate_transition_matrices(experiment_name, name; filter_function = x -> true)

Collect transition matrices for rows matching `filter_function`, aggregate them using
`select_best_transition_matrices`, and save to
`datadir("studies", experiment_name, "transition_analyses", name * ".jld2")`.
"""
function aggregate_transition_matrices(
        experiment_name::AbstractString,
        name::AbstractString;
        filter_function = x -> true
    )
    results_dir = datadir("studies", experiment_name, "results")
    res = DrWatson.collect_results!(results_dir)
    res = filter(filter_function, res)

    if isempty(res)
        error("No results found for experiment: $experiment_name")
    end

    save_dir = datadir("studies", experiment_name, "transition_analyses")
    success_matrices = Vector{Matrix{Float64}}()
    time_matrices = Vector{Matrix{Float64}}()

    for row in eachrow(res)
        save_path = joinpath(save_dir, basename(row.path))
        if !isfile(save_path)
            error("Missing transition analysis file: $save_path")
        end
        data = load(save_path)
        if !haskey(data, "transition_success_matrix") || !haskey(data, "transition_time_matrix")
            error("Missing transition matrices in: $save_path")
        end
        push!(success_matrices, data["transition_success_matrix"])
        push!(time_matrices, data["transition_time_matrix"])
    end

    if isempty(success_matrices)
        error("No transition matrices available to aggregate for: $experiment_name")
    end
    targets_per_model = map(eachrow(res)) do row
        transition_targets(row.experiment.env_config.context_strat)
    end
    best_success, best_time, _ = select_best_transition_matrices(success_matrices, time_matrices, targets_per_model)
    save_path = joinpath(save_dir, name * ".jld2")
    safesave(
        save_path,
        Dict(
            "transition_success_matrix" => best_success,
            "transition_time_matrix" => best_time
        )
    )
    return best_success, best_time
end


"""
inputs are time matrices
analyse the different matrices.
make barplot comparing times, 4 barplots for each target shock
grouped by starting shock on x axis, with different colors for each dataset
common legend for all barplots
"""
function transition_comparison(;
        matrices::Vector{<:AbstractMatrix},
        labels::Vector{<:AbstractString},
        colors::Vector{Any},
        num_banks = nothing,
        use_title = false,
        size = (20cm, 15cm)
    )
    # Validate inputs
    @assert(
        length(matrices) == length(labels) == length(colors),
        "Number of matrices, labels, and colors must match." *
            "Lengths are $(length(matrices)), $(length(labels)), $(length(colors))"
    )

    # Create figure with proper layout from the start
    fig = Figure(; size)

    # Add title
    if use_title
        Label(fig[1, 1:2], "Transition Time Comparison")
    end
    # Create a common legend for all subplots
    axes = []
    elements = []
    failure_scatters = []

    # Loop through each target shock count
    for (idx, target) in enumerate(1:4)
        # Create axis for this target shock count
        start_shocks = setdiff(1:4, target)
        ax = Axis(
            fig[div((idx - 1), 2) + 1, mod(idx - 1, 2) + 1],
            title = "Target wave count: $target",
            xlabel = idx == 3 || idx == 4 ? "Initial wave count" : "",
            ylabel = idx == 1 || idx == 3 ? "Transition time" : "",
            xticks = (1:3, string.(start_shocks))
        )
        idx % 2 == 0 && hideydecorations!(ax, ticks = false, grid = false)
        push!(axes, ax)

        n_datasets = length(matrices)
        total_width = 0.9
        gap = 0.3
        bar_width = total_width / n_datasets

        for (k, matrix) in enumerate(matrices)
            positions = collect(1:3) .- total_width ./ 2 .+ (k - 1 / 2) .* bar_width
            heights = [matrix[start, target] for start in start_shocks]
            bp = barplot!(ax, positions, heights, color = colors[k], width = bar_width, gap = gap)

            # Add cross marks for failed entries (NaN values)
            failed_indices = findall(isnan, heights)
            if !isempty(failed_indices)
                failed_positions = positions[failed_indices]
                # Use a small fixed y-offset that will be visible at the base of where bars would be
                # Calculate as 1% of the maximum successful bar height, or use a minimum of 0.5
                successful_heights = filter(!isnan, heights)
                y_offset = isempty(successful_heights) ? 0.5 : max(0.5, maximum(successful_heights) * 0.01)
                s = scatter!(
                    ax,
                    failed_positions,
                    fill(y_offset, length(failed_positions)),
                    marker = :x,
                    # color = colors[k],
                    color = :red,
                    markersize = 16
                )
                push!(failure_scatters, s)
                # scatter!(
                #     ax,
                #     failed_positions,
                #     fill(y_offset, length(failed_positions)),
                #     marker = :circle,
                #     color = colors[k],
                #     markersize = 10
                # )

            end

            idx == 1 && push!(elements, bp)
        end

    end

    # Link y-axes for consistent scale across plots
    linkyaxes!(axes...)
    # Add legend at the top
    num_banks = isnothing(num_banks) ? (length(labels) > 5 ? 2 : 1) : num_banks
    legend_elements = isempty(failure_scatters) ? elements : vcat(elements, failure_scatters[1])
    legend_labels = isempty(failure_scatters) ? labels : vcat(labels, "Failure")
    Legend(fig[3, 1:2], legend_elements, legend_labels, orientation = :horizontal, nbanks = num_banks)

    # Create directory for saving if it doesn't exist
    plot_dir = plotsdir("transition_comparison")
    if !isdir(plot_dir)
        mkpath(plot_dir)
    end

    # Generate a timestamp for the filename
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

    # Save the figure with timestamp
    save(joinpath(plot_dir, "transition_comparison_$(timestamp).png"), fig)

    return fig
end

function load_matrices(experiment_name, suffix = ""; mode = :best_reward)
    if mode == :best_reward
        if experiment_name == "two_step_transition_time"
            file_path = datadir(experiment_name, "transition_time_matrices.jld2")
        elseif experiment_name == "two_step_transition_time_per_target"
            file_path = datadir(experiment_name, "transition_time_matrices.jld2")
        else
            file_path = datadir("studies", experiment_name, "transition_rl_analysis_matrices" * suffix * ".jld2")
        end
        if !isfile(file_path)
            error("File $file_path does not exist")
        end
    elseif mode == :best_transition
        file_path = datadir("studies", experiment_name, "transition_analyses", experiment_name * suffix * ".jld2")
    else
        error("Invalid mode: $mode")
    end
    data = load(file_path)
    return data["transition_success_matrix"], data["transition_time_matrix"]
end
