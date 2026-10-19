function find_nearest_shock(shock_locations)
    N = length(shock_locations)
    # Find first shock location
    if sum(shock_locations) == 0
        return false
    end
    shocks = zeros(2)
    first_shock = findfirst(shock_locations)
    shocks[1] = first_shock
    # Find last shock location
    last_shock = findlast(shock_locations)
    shocks[2] = last_shock
    chosen_shock = shocks[argmin([shocks[1], abs(N - shocks[2])])]
    return Int(chosen_shock)
end


function get_jl_policy(result_row::DataFrameRow, models_dir::String; name::String = "", kwargs...)
    experiment = result_row.experiment
    model_path = result_row.model_path
    model_path_basename = basename(model_path)
    full_path = joinpath(models_dir, model_path_basename)
    return get_jl_policy(experiment, full_path; kwargs...)
    # model = load_model(result_row, models_dir)
    # policy = SBPolicy(model.policy; name=name, kwargs...)
    # return policy
end

# ==============================================================================
# Study Directory Utilities for Converted Data
# ==============================================================================

"""
    get_study_dir(study_name::String; use_converted::Bool = true)

Get the path to a study directory, optionally using the converted version.

# Arguments
- `study_name::String`: Name of the study
- `use_converted::Bool`: Whether to use the converted version (default: true)

# Returns
- `String`: Path to the study directory

# Examples
```julia
# Use converted data (recommended after migration)
res = collect_results!(joinpath(get_study_dir("transition_SA_9"), "results"))

# Use original data
res = collect_results!(joinpath(get_study_dir("transition_SA_9", use_converted=false), "results"))
```
"""
function get_study_dir(study_name::String; use_converted::Bool = true)
    suffix = use_converted ? "_converted" : ""
    return datadir("studies", study_name * suffix)
end

"""
    get_study_results_dir(study_name::String; use_converted::Bool = true)

Get the path to a study's results directory.

# Arguments
- `study_name::String`: Name of the study
- `use_converted::Bool`: Whether to use the converted version (default: true)

# Returns
- `String`: Path to the study's results directory

# Examples
```julia
res = collect_results!(get_study_results_dir("transition_SA_9"))
```
"""
function get_study_results_dir(study_name::String; use_converted::Bool = true)
    return joinpath(get_study_dir(study_name; use_converted = use_converted), "results")
end

"""
    get_study_models_dir(study_name::String; use_converted::Bool = true)

Get the path to a study's models directory.

# Arguments
- `study_name::String`: Name of the study
- `use_converted::Bool`: Whether to use the converted version (default: true)

# Returns
- `String`: Path to the study's models directory
"""
function get_study_models_dir(study_name::String; use_converted::Bool = true)
    return joinpath(get_study_dir(study_name; use_converted = use_converted), "models")
end

function calculate_saves_per_action(n_states::Int, n_actions::Int)
    return (n_states - 1) / n_actions |> round |> Int
end

calculate_saves_per_action(data::PolicyRunData) = calculate_saves_per_action(length(data.states), length(data.action_ts))
function calculate_saves_per_action(vec::Vector, vec2::Vector)
    n_1 = length(vec)
    n_2 = length(vec2)
    if n_1 == n_2
        return 1
    elseif n_1 > n_2
        return calculate_saves_per_action(n_1, n_2)
    else
        return calculate_saves_per_action(n_2, n_1)
    end
end

#TODO: reasonableness?
function get_reasonable_time_scale(L::T, n_sections = 1; average_wave_speed::T = 1.71f0) where {T <: AbstractFloat}
    section_length = L / n_sections
    avg_time_in_section = section_length / average_wave_speed
    return avg_time_in_section
end

@kwdef mutable struct EvalCallback <: Drill.AbstractCallback
    eval_config::EvalConfig
    dril_config::DrillConfig
    eval_freq::Int # parallell env steps, so 1 means after all envs have taken a step
    n_calls::Int = 0 #number of times the callback has been called
    spatial_kernel_width::Int = 0
    progress_task::Union{ProgressTask, Nothing} = nothing
end

function Drill.on_step(callback::EvalCallback, locals::Dict)
    callback.n_calls += 1
    if callback.n_calls % callback.eval_freq == 0
        maybe_update_progress!(callback.progress_task, "Evaluating...")
        eval_env = make_eval_env(
            callback.eval_config.env_config,
            callback.dril_config,
            locals[:env];
            spatial_kernel_width = callback.spatial_kernel_width
        )
        mean_reward, std_reward, mean_length, std_length = evaluate_agent(locals[:agent], eval_env; n_eval_episodes = callback.eval_config.n_eval_episodes)
        Drill.log_scalar!(locals[:agent].logger, "eval/ep_rew_mean", mean_reward)
        Drill.log_scalar!(locals[:agent].logger, "eval/ep_rew_std", std_reward)
        Drill.log_scalar!(locals[:agent].logger, "eval/ep_len_mean", mean_length)
        Drill.log_scalar!(locals[:agent].logger, "eval/ep_len_std", std_length)
    end
    return true
end

@kwdef mutable struct SaveCallback <: Drill.AbstractCallback
    save_freq::Int # parallell env steps, so 1 means after all envs have taken a step
    n_calls::Int = 0 #number of times the callback has been called
    model_path::String
    progress_task::Union{ProgressTask, Nothing} = nothing
end

function Drill.on_step(callback::SaveCallback, locals::Dict)
    callback.n_calls += 1
    if callback.n_calls % callback.save_freq == 0
        maybe_update_progress!(callback.progress_task, "Saving...")
        current_model_path = replace(callback.model_path, ".jld2" => "_$(callback.n_calls).jld2")
        jldsave(current_model_path; agent = locals[:agent], env = locals[:env])
    end
    return true
end


function send_ntfy(
        topic::String, message::String;
        title::String = "",
        priority::Int = 3,
        tags::String = ""
    )
    headers = Dict{String, String}()

    !isempty(title) && (headers["Title"] = title)
    priority != 3 && (headers["Priority"] = string(priority))
    !isempty(tags) && (headers["Tags"] = tags)

    # NOTE: HTTP.request is not marked public in HTTP; keep until public alternative exists.
    return HTTP.request("POST", "https://ntfy.sh/$topic", headers, message)
end

const NTFY_TOPIC = let
    filepath = projectdir(".secrets", "NTFY_TOPIC")
    if isfile(filepath)
        topic = strip(read(filepath, String)) |> String
        isempty(topic) ? nothing : topic
    else
        nothing
    end
end

function get_best_policy(
        experiment_name;
        filter_function = r -> true,
        initial_shock::Union{Int, Nothing} = nothing,
        target_shock_count::Union{Int, Nothing} = nothing,
        delay = 0.0f0,
        tmax = 330.0f0,
        stability_reward_factor = 1.0f0,
        wrap_delay = true
    )
    results_dir = datadir("studies", experiment_name, "results")
    models_dir = datadir("studies", experiment_name, "models")
    res = DrWatson.collect_results!(results_dir)
    res = filter(filter_function, res)
    add_transition_matrices!(res, experiment_name)

    targets = map(eachrow(res)) do row
        if :experiment in propertynames(row) &&
                !ismissing(row.experiment) &&
                hasproperty(row.experiment, :env_config) &&
                hasproperty(row.experiment.env_config, :context_strat)
            return transition_targets(row.experiment.env_config.context_strat)
        elseif :target_shock_count in propertynames(row) && !ismissing(row.target_shock_count)
            return [Int(row.target_shock_count)]
        else
            error("No targets found for row: $row")
        end
    end

    mean_success_rates = map(eachindex(targets)) do idx
        model_targets = targets[idx]
        sum(res.transition_success_matrix[idx][:, model_targets]) / (length(model_targets) * 3)
    end
    mean_times = map(eachindex(targets)) do idx
        model_targets = targets[idx]
        sum(res.transition_time_matrix[idx][:, model_targets]) / (length(model_targets) * 3)
    end

    if !isnothing(target_shock_count)
        qualified_models = findall(x -> target_shock_count in x, targets)
        if isempty(qualified_models)
            error("No models found for target shock count: $target_shock_count")
        end
    else
        qualified_models = eachindex(targets)
    end
    qualified_success_rates = mean_success_rates[qualified_models]
    qualified_times = mean_times[qualified_models]
    best_local_idx = best_index_from_rates(qualified_success_rates, qualified_times)
    chosen_idx = qualified_models[best_local_idx]

    chosen_row = res[chosen_idx, :]
    chosen_policy = get_jl_policy(chosen_row, models_dir)
    experiment = chosen_row.experiment
    @reset experiment.env_config.tmax = tmax
    @reset experiment.env_config.reward_strat = StabilityReward(stability_reward_factor)
    if !isnothing(initial_shock)
        @reset experiment.env_config.reset_strategy = ShiftReset(NShock(initial_shock))
    end
    env = create_env(experiment.env_config)
    if wrap_delay
        policy = DelayedPolicy(chosen_policy, delay, env)
    else
        policy = chosen_policy
    end
    return policy, env, chosen_row
end
