struct LinearPressureChangeConfig
    init_shock_count::Int
    init_pressure::Float32
    final_pressure::Float32
    pressure_change_duration::Float32
    run::Int
end

struct LinearPressureChangeExperiment
    env_config::RDEEnvConfig
    pressure_change_config::LinearPressureChangeConfig
end

struct LinearPressureChangeResults
    ts::Vector{Float32}
    u_ps::Vector{Float32}
    shocks::Vector{Int}
end

function get_linear_pressure_change_configs(n, runs_per_config = 3)
    pressure_change_durations = LinRange(1.0f0, 100.0f0, n)
    shock_pressures = RDE.SHOCK_PRESSURES
    configs = LinearPressureChangeExperiment[]

    for init_shock_count in 1:4
        init_pressure = shock_pressures[init_shock_count]
        for final_target_shock_count in setdiff(1:4, init_shock_count)
            final_pressure = shock_pressures[final_target_shock_count]
            for duration in pressure_change_durations
                env_config = RDEEnvConfig(
                    tmax = 800.0f0,
                    reset_strategy = ShiftReset(NShock(init_shock_count)),
                    reward_strat = PeriodicityReward()
                )
                for i in 1:runs_per_config
                    config = LinearPressureChangeConfig(init_shock_count, init_pressure, final_pressure, duration, i)
                    push!(configs, LinearPressureChangeExperiment(deepcopy(env_config), config))
                end
            end
        end
    end
    return configs
end


function run_single_pressure_change_experiment(config::LinearPressureChangeExperiment)
    env = create_env(config.env_config)
    policy = LinearPolicy(
        env,
        [0.0f0, config.pressure_change_config.pressure_change_duration],
        [config.pressure_change_config.init_pressure, config.pressure_change_config.final_pressure]
    )
    data = run_policy(policy, env)
    us, _ = RDE.split_sol(data.states)
    saves_per_action = calculate_saves_per_action(data)
    shocks = RDE.count_shocks.(us[1:saves_per_action:end], env.prob.x[2] - env.prob.x[1])
    result = LinearPressureChangeResults(data.action_ts, data.u_ps, shocks)
    return result, data
end

function experiment_to_string(config::LinearPressureChangeExperiment)
    return "init_u_p=$(config.pressure_change_config.init_pressure), final_u_p=$(config.pressure_change_config.final_pressure), duration=$(config.pressure_change_config.pressure_change_duration)"
end

function run_experiments(experiments::Vector{LinearPressureChangeExperiment}, study_name::String = "main")
    @showprogress @threads for experiment in experiments
        @produce_or_load(
            experiment,
            datadir("fast_and_slow_transitions", study_name),
            loadfile = false,
            verbose = false,
            filename = experiment_to_string(experiment)
        ) do experiment
            result, data = run_single_pressure_change_experiment(experiment)
            #Not saving data so save space
            output = Dict("result" => result, "experiment" => experiment)
            return output
        end
    end
    @info "Experiment running finished!"
    return nothing
end

#To run: run_experiments(get_linear_pressure_change_configs(10), "test")
# run_experiments(get_linear_pressure_change_configs(100), "main")
