using DrWatson
@quickactivate :DRL_RDE
using CairoMakie
CairoMakie.activate!()
using MakieExtra
using RDE
using RDE_Env: get_plotting_speed_adjustments, plot_shifted_history!
using Accessors
##
dt = 1.0f0
tmax = 700.0f0
target_shock_count = 3
N = 512
env = RDEEnv(;
    dt = dt,
    params = RDEParam(; tmax, N),
    observation_strat = SectionedStateObservation(),
    action_strat = DirectScalarPressureAction(),
    reward_strat = PeriodMinimumReward(),
    context_strat = FixedTargetShockCountStrategy(target_shock_count),
);
_reset!(env)
time_points = [50.0f0, 200.0f0, 360.0f0]
second_to_last = 520.0f0
last = second_to_last + 80.0f0
push!(time_points, second_to_last)
push!(time_points, last)
policy = StepwiseRDEPolicy(
    env,
    time_points,
    [0.64f0, 0.85f0, 0.64f0, 0.98f0, 1.0f0],
);
@time run_data = run_policy(policy, env);

##
# Time window for the two top (zoom) axes
t_win = (390.0f0, 440.0f0)
us, = RDE.split_sol(run_data.states)
x = env.prob.x
dx = x[2] - x[1]
ts = run_data.state_ts
c_auto = get_plotting_speed_adjustments(run_data, dx; max_speed = 4.6f0)
c_zero = 0.0f0
extrema_us = extrema(stack(us))
colorrange = [extrema_us[1], extrema_us[2]]

# Indices for window
i1 = searchsortedfirst(ts, t_win[1])
i2 = searchsortedlast(ts, t_win[2])
inds = i1:i2
ts_win = ts[inds]
us_win = us[inds]
# c_auto has length length(ts)-1 (per interval)
c_auto_win = c_auto[inds[1]:(inds[end] - 1)]
u_ax_kwargs = (;
    yticks = ([0, π / 2, π, 3π / 2, 2π], [L"0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"]),
    title = "",
    ylabel = "",
    xlabel = "",
    limits = (nothing, (0.0f0, 2.0f0 * π)),
)
shared_kwargs = (;
    rewards = nothing,
    plot_shocks = false,
    u_ps = nothing,
    action_ts = run_data.action_ts,
    control_shifts = run_data.control_shifts,
)
u_ax_kwargs_moving = merge(
    u_ax_kwargs, (;
        title = L"u(\psi, t) \text{ (Moving frame)}",
        ylabel = L"\psi",
    )
)
u_ax_kwargs_stationary = merge(
    u_ax_kwargs, (;
        title = L"A \qquad \text{Stationary frame}", titlealign = :left,
        ylabel = L"x",
    )
)

dpi = 300
u_hm_kwargs = (; rasterize = ceil(Int, dpi * pt / inch))
u_ax_kwargs_main = (
    xticks = time_points,
    yticks = (
        [0, π / 2, π, 3π / 2, 2π],
        [L"0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"],
    ),
    limits = (nothing, (0.0f0, 2.0f0 * π)),
    ylabel = L"x - \psi(t) \,\mathrm{mod}\, 2\pi",
    title = L"B \qquad \text{Moving frame}",
    titlealign = :left,
)
u_p_ax_kwargs = (;
    title = L"C",
    titlealign = :left,
)
fig = with_theme(theme_drl_rde()) do
    fig = Figure(size = (16cm, 12cm))
    # Top: two zoom axes (moving, then stationary) for time window (360, 480)
    top_gl = fig[1, 1] = GridLayout()
    gl_stationary = top_gl[1, 1] = GridLayout()
    # gl_moving = top_gl[1, 2] = GridLayout()
    plot_shifted_history!(
        gl_stationary,
        us_win,
        x,
        ts_win,
        c_zero;
        u_ax_kwargs = u_ax_kwargs_stationary,
        colorrange,
        colorbar_label = L"u(x, t)",
        u_hm_kwargs,
        shared_kwargs...,
    )
    # plot_shifted_history!(
    #     gl_moving,
    #     us_win,
    #     x,
    #     ts_win,
    #     c_auto_win;
    #     u_ax_kwargs = u_ax_kwargs_moving,
    #     shared_kwargs...,
    # )
    # Get inset axis for zoom_lines! (use moving-frame zoom)
    # @show gl_stationary.content
    # ax_inset_moving = contents(gl_moving)[1]
    ax_inset_stationary = contents(gl_stationary)[1]

    # Bottom: full run (sample-run style)
    main_gl = fig[2, 1] = GridLayout()
    plot_shifted_history!(
        main_gl,
        run_data,
        x;
        use_rewards = false,
        plot_shocks = false,
        u_ax_kwargs = u_ax_kwargs_main,
        colorrange,
        u_p_ax_kwargs,
        u_hm_kwargs,
    )
    ax_main = contents(main_gl)[1]
    # Lines on main axis showing the zoom window (360, 480)
    # zoom_lines!(ax_main, ax_inset_moving)
    zoom_lines!(ax_main, ax_inset_stationary)

    # Row sizes: top section smaller, main section larger
    rowsize!(fig.layout, 1, Relative(0.35))
    rowsize!(fig.layout, 2, Relative(0.65))
    # Within main panel: heatmap 70%, u_p row 30% (match sample-run)
    rowsize!(main_gl, 2, Relative(0.3))
    # hidexdecorations!(ax_inset_moving, ticks = false)
    hidexdecorations!(ax_main, ticks = false)
    rowgap!(fig.layout, 0)
    resize_to_layout!(fig)
    return fig
end
##
foreach([".png", ".pdf"]) do ext
    wsave(plotsdir("figures", "sample-run-with-timescales$ext"), fig)
end
