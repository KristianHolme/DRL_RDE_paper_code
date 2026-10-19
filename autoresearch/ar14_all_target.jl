function sample_cfgs(rng::AbstractRNG, n_samples::Int)
    cfg = Dict{String, Any}(
        "experiment_name" => "ar14_all_target",
        "action_strat" => "linear",
        "recipe" => ["us4", "ar4_tv"],
        "obs" => DrWatson.Derived(
            ["recipe"],
            (recipe) -> recipe == "us4" ? "fourier8" : "fourier8_up",
        ),
        "reward_strat" => DrWatson.Derived(
            ["recipe"],
            (recipe) -> recipe == "us4" ? "stabilitytarget" : "stabilitytarget_tv_lock",
        ),
        "target_shock_count" => [1, 2, 3, 4],
        "experiment_seed" => [98, 84, 15, 51, 44, 1],
        "n_sections" => 1,
        "n_agents" => 1,
        "movingframe" => false,
        "n_envs" => 1,
        "dt" => [10.0f0],
        "lock_bonus" => 150.0f0,
        "learning_rate" => [3.0f-4],
        "gamma" => DrWatson.Derived(["dt"], (dt) -> 1.0f0 - dt / 100.0f0),
        "ent_coef" => [0.01f0],
        "epochs" => [4],
        "target_kl" => [0.03f0],
        "name_keys" => [["recipe", "target_shock_count", "experiment_seed", "run"]],
    )
    cfg_list = DrWatson.dict_list(cfg)
    return cfg_list
end
