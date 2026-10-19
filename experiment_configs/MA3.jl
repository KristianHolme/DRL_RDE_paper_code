function sample_cfgs(rng::AbstractRNG, n_samples::Int)
    cfg = Dict{String, Any}(
        "experiment_name" => "MA3",
        "experiment_seed" => 827,
        "movingframe" => [false, true],
        "n_sections" => 16,
        "n_envs" => 1,
        "target_shock_count" => [1, 2, 3, 4],
        "dt" => 2.0f0 .^ collect(-2:3),
        "gamma" => DrWatson.Derived(["dt"], (dt) -> 1.0f0 - dt / 10.0f0),
    )
    cfg["n_agents"] = cfg["n_sections"]
    cfg_list = DrWatson.dict_list(cfg)
    return cfg_list
end
