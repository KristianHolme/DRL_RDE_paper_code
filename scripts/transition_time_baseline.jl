using DrWatson
@quickactivate :DRL_RDE
using RDE
using RDE_Env

using CairoMakie
CairoMakie.activate!()

# Constants
shock_u_ps = copy(RDE.SHOCK_PRESSURES)

# Parameter configurations for experiments
params = Dict(
    :start_shocks => [1, 2, 3, 4],
    :target_shocks => [1, 2, 3, 4],
    :intermediate_u_p => LinRange(0.1f0, 0.98f0, 10) |> collect,
    :intermediate_duration => LinRange(5.0f0, 60.0f0, 10) |> collect,
)

configs = dict_list(params)

## Run experiments
using ProgressMeter
@showprogress Threads.@threads for config in configs
    produce_or_load(config, datadir("two_step_transition_time")) do cfg
        run_transition(cfg, shock_u_ps; save_result = true, make_plot = false)
    end
end
# to run:
# include(joinpath("scripts", "studies", "transition", "transition_time_baseline.jl"))
