using DrWatson
@quickactivate :DRL_RDE
using GLMakie
GLMakie.activate!()
using Accessors
## Create environment
experiment_name = "SAS3"
policy, env, row = get_best_policy(
    experiment_name;
    filter_function = r -> r.movingframe == true && r.target_shock_count == 2,
    wrap_delay = false,
    initial_shock = 3,
    target_shock_count = 2,
    tmax = 50.0f0,
)
@reset env.prob.reset_strategy = NShock(3)
@reset env.prob.params.tmax = 10.0f0
##
data = run_policy(policy, env)
params = env.prob.params
time_idx = Observable(25)

## Static figure for saving (no title, no controls)
R = 3.1f0
fig = with_theme(theme_drl_rde()) do
    plot_rde_data_3d(
        data, params; show_title = false,
        show_controls = false,
        size = (8cm, 10cm),
        time_idx = time_idx,
        injection_pie_z = -0.5f0,
        wave_arrow_positions = [1.2, 3.5, 4.8],
        wave_arrow_heights = [0.7, 0.6, 0.2],
        eyeposition = Point3f(-R, -R, 2.2f0),
        lookat = Point3f([0, 0, 0.7])
    )
end
# zoom!(fig.content[1].scene, 0.82)
##
dpi = 300
wsave(plotsdir("figures", "rde_diagram.png"), fig; px_per_unit = dpi / inch)

## Interactive figure with title and controls
time_idx = Observable(1)
fig = plot_rde_data_3d(
    data,
    params;
    time_idx = time_idx,
    show_title = true,
    show_controls = true,
    size = (12cm, 12cm),
    wave_arrows = false,
    control_arrows = false,
)
fig
