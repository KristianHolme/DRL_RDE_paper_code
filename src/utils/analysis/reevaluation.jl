using Base.Threads
function reevaluate_results!(res)
    @showprogress desc = "Reevaluating results" showspeed = true @threads for row in eachrow(res)
        experiment = row.experiment
        experiment_name = experiment.study
        dril_config = experiment.dril_config
        model_path = row.model_path
        base_path = basename(model_path)
        path = datadir("studies", experiment_name, "models", base_path)
        agent = get_agent(experiment, path)
        train_env = get_train_env(experiment, path)
        eval_env = make_eval_env(experiment.eval_config.env_config, dril_config, train_env)
        mean_reward, std_reward, mean_length, std_length = evaluate_agent(agent, eval_env; n_eval_episodes = experiment.eval_config.n_eval_episodes)
        row."eval/ep_rew_mean" = mean_reward
        row."eval/ep_rew_std" = std_reward
        row."eval/ep_len_mean" = mean_length
        row."eval/ep_len_std" = std_length
    end
    return nothing
end
