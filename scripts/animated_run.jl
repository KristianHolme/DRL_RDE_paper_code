using DrWatson
@quickactivate :DRL_RDE
using GLMakie
GLMakie.activate!()
using ProgressMeter
##
dt = 1.0f0
tmax = 700.0f0
target_shock_count = 3
N = 512
## Create environment
env = RDEEnv(;
    dt = dt,
    params = RDEParam(; tmax, N),
    observation_strat = SectionedStateObservation(),
    action_strat = DirectScalarPressureAction(),
    reward_strat = PeriodMinimumReward(),
    context_strat = FixedTargetShockCountStrategy(target_shock_count)
);
_reset!(env)
time_points = [50.0f0, 140.0f0, 300.0f0, 450.0f0, 530.0f0]
policy = StepwiseRDEPolicy(
    env, time_points,
    [0.65f0, 0.85f0, 0.64f0, 0.98f0, 1.0f0]
);
@time run_data = run_policy(policy, env);
#
time_idx = Observable(1)
fig = plot_policy_data(run_data, env; fig_size = (1400, 900), live_control = true, rewards_and_shocks = false, control_history = false)
## moving frame at t = 90, speedup to 2
switch_idx = findfirst(run_data.state_ts .> 90.0f0)
moving_frame = Observable(true)
# on(time_idx) do ti
#     if run_data.state_ts[ti] > 90.0f0
#         moving_frame[] = true
#     else
#         moving_frame[] = false
#     end
# end

time_steps = length(run_data.state_ts)
iter = [collect(1:1:switch_idx); collect(switch_idx:2:time_steps)]
fig = plot_policy_data(
    run_data, env; time_idx, player_controls = false,
    show_mouse_vlines = false, rewards_and_shocks = false, control_history = false,
    moving_frame = true,
    live_control = true,
    fig_size = (1000, 450)
)
stationary_layout = fig[1, 2] = GridLayout()
RDE_Env.plot_policy_data!(
    stationary_layout, run_data, env; time_idx, player_controls = false,
    show_mouse_vlines = false, rewards_and_shocks = false, control_history = false,
    live_control = true,
    moving_frame = false,
)

path = joinpath(plotsdir("animations"), "sample-run.mp4")
if !isdir(dirname(path))
    mkpath(plotsdir("animations"))
end
p = Progress(length(iter), desc = "Recording animation...")
record(fig, path, iter; framerate = 60, px_per_unit = 300 / inch) do i
    time_idx[] = i
    next!(p)
end
