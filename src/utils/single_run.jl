function single_run(
        config::Dict,
        experiment::DrillExperiment;
        ignore_existing::Bool = false,
        progress_task::Union{ProgressTask, Nothing} = nothing,
        continue_training::Bool = false,
        params_dir::String = "",
        save::Bool = true,
        logging::Bool = true,
        n_evals::Int = 15,
    )

    if nthreads() == 1
        @warn "Running with single thread, this is not recommended"
    end

    experiment_dir = datadir("studies", experiment.study)
    trial_name = experiment_savename(config)
    log_dir = joinpath(experiment_dir, "logs", trial_name)
    # @info "TESTING LOGGGING FROM $(myid())"

    if isdir(log_dir) && !ignore_existing && !continue_training
        @info "log_dir already exists, skipping training" log_dir
        if !isnothing(progress_task)
            finish!(progress_task)
        end
        return Dict()
    end

    model_dir = joinpath(experiment_dir, "models")
    isdir(model_dir) || mkpath(model_dir)
    model_path = joinpath(model_dir, experiment_savename(config, "jld2"))


    seed = experiment.seed
    dril_config = experiment.dril_config

    algorithm = dril_config.algorithm
    learn_config = dril_config.learn_config

    spatial_kernel_width = get(config, "spatial_kernel_width", 0)
    train_env = make_train_env(experiment.env_config, dril_config; spatial_kernel_width = spatial_kernel_width)
    DrillInterface.reset!(train_env)


    observation_space = DrillInterface.observation_space(train_env)
    action_space = DrillInterface.action_space(train_env)

    !logging && (log_dir = nothing)

    agent = make_agent(observation_space, action_space, dril_config; log_dir, verbose = learn_config.verbose)

    hparams = get_hparams(algorithm)
    hparams = merge(hparams, filter_for_hparams(config))
    metrics = [
        "env/ep_rew_mean",
        "env/ep_len_mean",
        "train/loss",
        "eval/ep_rew_mean",
        "eval/ep_rew_std",
        "eval/ep_len_mean",
        "eval/ep_len_std",
    ]
    if algorithm isa PPO
        push!(metrics, "train/explained_variance")
    end
    Drill.log_hparams!(agent.logger, hparams, metrics)

    # Set random seed
    if seed isa Int
        Random.seed!(agent, seed)
        Random.seed!(train_env, seed)
    end


    if isfile(model_path) && continue_training
        @error "Loading model from $model_path is not implemented"
    end
    callbacks = Vector{Drill.AbstractCallback}()
    if !isnothing(progress_task)
        # Keep task ownership local so the full run can report evaluation/post-processing
        # before emitting a single terminal finish event.
        push!(callbacks, create_dril_progress_callback(progress_task))
    end
    if n_evals > 0
        n_envs = number_of_envs(train_env)
        eval_freq = (learn_config.max_timesteps ÷ n_envs) ÷ n_evals
        eval_callback = EvalCallback(
            eval_config = experiment.eval_config,
            dril_config = dril_config,
            eval_freq = eval_freq,
            spatial_kernel_width = spatial_kernel_width,
            progress_task = progress_task,
        )
        push!(callbacks, eval_callback)

        save_callback = SaveCallback(
            model_path = model_path,
            save_freq = eval_freq,
            progress_task = progress_task,
        )
        push!(callbacks, save_callback)
    end

    @logmsg LogLevel(-500) "Training model"
    t_start = time()
    train!(
        agent, train_env, algorithm, learn_config.max_timesteps;
        callbacks, ad_type = Lux.AutoEnzyme()
    )
    training_time = time() - t_start
    @logmsg LogLevel(-500) "Learning done, starting evaluation and saving"

    # Evaluate
    maybe_update_progress!(progress_task, "Evaluating...")
    eval_config = experiment.eval_config
    train_mean_reward, train_std_reward, train_mean_length, train_std_length = evaluate_agent(agent, train_env; n_eval_episodes = eval_config.n_eval_episodes)


    eval_env = make_eval_env(
        experiment.eval_config.env_config,
        dril_config,
        train_env;
        spatial_kernel_width = spatial_kernel_width
    )
    mean_reward, std_reward, mean_length, std_length = evaluate_agent(agent, eval_env; n_eval_episodes = eval_config.n_eval_episodes)
    Drill.log_scalar!(agent.logger, "eval/ep_rew_mean", mean_reward)
    Drill.log_scalar!(agent.logger, "eval/ep_rew_std", std_reward)
    Drill.log_scalar!(agent.logger, "eval/ep_len_mean", mean_length)
    Drill.log_scalar!(agent.logger, "eval/ep_len_std", std_length)

    # scaled_mean_reward = mean_reward * experiment.env_config.dt / experiment.env_config.tmax
    # scaled_std_reward = std_reward * experiment.env_config.dt / experiment.env_config.tmax

    save && jldsave(model_path; agent, train_env)

    transition_success_matrix = nothing
    transition_time_matrix = nothing
    if save
        maybe_update_progress!(progress_task, "Computing transition matrices...")
        targets = transition_targets(experiment.env_config.context_strat)
        transition_success_matrix, transition_time_matrix, _ = transition_matrices(
            experiment,
            model_path;
            targets
        )
        transition_dir = datadir("studies", experiment.study, "transition_analyses")
        if !isdir(transition_dir)
            mkpath(transition_dir)
        end
        safesave(
            joinpath(transition_dir, experiment_savename(config, "jld2")),
            Dict(
                "transition_success_matrix" => transition_success_matrix,
                "transition_time_matrix" => transition_time_matrix
            )
        )
        maybe_update_progress!(progress_task, "Transition matrices saved")
    end

    result = merge(
        config, Dict(
            "experiment" => experiment,
            #this should maybe be called "test" instead of "eval"?
            "eval/ep_rew_mean" => mean_reward,
            "eval/ep_rew_std" => std_reward,
            "eval/ep_len_mean" => mean_length,
            "eval/ep_len_std" => std_length,
            # "scaled_mean_reward" => scaled_mean_reward,
            # "scaled_std_reward" => scaled_std_reward,
            "training_time" => training_time,
            "total_timesteps" => steps_taken(agent),
            "model_path" => model_path,
            "timestamp" => now(),
            "computer" => gethostname(),
            "framework" => "Drill",
            "algorithm" => string(typeof(algorithm)),
            "dt" => experiment.env_config.dt,
            "tmax" => experiment.env_config.tmax,
            # these should maybe be called "eval" instead of "train"?
            "train/ep_rew_mean" => train_mean_reward,
            "train/ep_rew_std" => train_std_reward,
            "train/ep_len_mean" => train_mean_length,
            "train/ep_len_std" => train_std_length,
            "transition_success_matrix" => transition_success_matrix,
            "transition_time_matrix" => transition_time_matrix,
        )
    )

    result = merge(result, hparams)
    if !isnothing(progress_task)
        finish!(progress_task)
    end
    @logmsg LogLevel(-1000) "Evaluation and saving done, returning result"
    if save
        safesave(joinpath(experiment_dir, "results", experiment_savename(config, "jld2")), result)
    end
    return result
end


function make_agent(observation_space, action_space, dril_config; log_dir = nothing, verbose = 1)
    policy = setup_policy(
        observation_space, action_space, dril_config.algorithm, dril_config.policy_config
    )
    logger = isnothing(log_dir) ? Drill.NoTrainingLogger() : TB.TBLogger(log_dir)
    agent = Agent(policy, dril_config.algorithm; logger, verbose)
    return agent
end

function setup_policy(observation_space, action_space, ::PPO, policy_config::LayerKwargs)
    policy = ActorCriticLayer(observation_space, action_space; policy_config.kwargs...)
    return policy
end

function setup_policy(observation_space, action_space, ::SAC, policy_config::LayerKwargs)
    policy = SACLayer(observation_space, action_space; policy_config.kwargs...)
    return policy
end
