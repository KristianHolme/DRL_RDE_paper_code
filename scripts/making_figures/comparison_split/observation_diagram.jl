# Two-panel observation diagram: Fourier magnitude (SU) and sectioned state (segmented).
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using CairoMakie
CairoMakie.activate!()
using FFTW
include(joinpath(@__DIR__, "plot_knobs.jl"))
##
dt = 1.0f0
tmax = 700.0f0
target_shock_count = 3
N = 512
##
env = RDEEnv(;
    dt = dt,
    params = RDEParam(; tmax, N),
    observation_strat = SectionedStateObservation(),
    action_strat = DirectScalarPressureAction(),
    reward_strat = PeriodMinimumReward(),
    context_strat = FixedTargetShockCountStrategy(target_shock_count),
)
_reset!(env)
time_points = [50.0f0, 140.0f0, 300.0f0, 450.0f0, 530.0f0]
policy = StepwiseRDEPolicy(
    env, time_points,
    [0.65f0, 0.85f0, 0.64f0, 0.98f0, 1.0f0],
)
run_data = run_policy(policy, env)
##
time = 210.0f0
timeidx_sparse = findfirst(run_data.action_ts .>= time)
time_exact = run_data.action_ts[timeidx_sparse]
timeidx_dense = findfirst(run_data.state_ts .≈ time_exact)
state = run_data.states[timeidx_dense]
u, λ = RDE.split_sol(state)
observation = run_data.observations[timeidx_sparse]
scaled_u = u ./ 2
N_mini = (length(observation) - 2) ÷ 2
n_modes = 8
scale = inv(Float32(N))
u_hat = abs.(rfft(u)) .* scale
λ_hat = abs.(rfft(λ)) .* scale
fourier_obs = vcat(u_hat[1:n_modes], λ_hat[1:n_modes])
##
figsize = (16cm, 9cm)
##
fourier_scale = fourier_bar_headroom / maximum(fourier_obs)
fourier_plot = fourier_obs .* fourier_scale
fig = with_theme(comparison_theme()) do
    fig = Figure(; size = figsize, figure_padding = 4, backgroundcolor = bgcolor)
    ax_f = Axis(fig[1, 1]; title = "$(panel_letter_a)  Fourier magnitude")
    xu = 2π / n_modes .* ((1:n_modes) .- 0.5)
    xλ = 2π / n_modes .* (((n_modes + 1):2n_modes) .- 0.5)
    barplot!(ax_f, xu, fourier_plot[1:n_modes]; color = ucolor_sub)
    barplot!(ax_f, xλ, fourier_plot[(n_modes + 1):2n_modes]; color = lambdacolor_sub)
    bracket!(
        ax_f, Point2(0, 1.0), Point2(2π, 1.0);
        color = ucolor_main, text = L"| \hat{u} |_{0:7}", linewidth = 2,
    )
    bracket!(
        ax_f, Point2(2π, 1.0), Point2(4π, 1.0);
        color = lambdacolor_main, text = L"| \hat{\lambda} |_{0:7}", linewidth = 2,
    )
    hidedecorations!(ax_f)
    hidespines!(ax_f)
    xlims!(ax_f, -0.05, 4π)
    ylims!(ax_f, -0.15, 1.5)

    ax = Axis(fig[2, 1]; title = "$(panel_letter_b)  Sectioned state")
    barplot!(ax, 2π / N_mini .* ((1:N_mini) .- 0.5), observation[1:N_mini]; color = ucolor_sub)
    barplot!(
        ax, 2π / N_mini .* (((N_mini + 1):2N_mini) .- 0.5),
        observation[(N_mini + 1):2N_mini]; color = lambdacolor_sub,
    )
    lines!(
        ax, LinRange(0.0, 2π, 512), scaled_u;
        color = ucolor_main, linewidth = 2, linestyle = (:dot, :dense),
    )
    lines!(
        ax, LinRange(2π, 4π, 512), λ;
        color = lambdacolor_main, linewidth = 2, linestyle = (:dot, :dense),
    )
    barplot!(
        ax, 2π / N_mini .* (((2N_mini + 1):(2N_mini + 2)) .- 0.5),
        observation[(2N_mini + 1):(2N_mini + 2)]; color = wave_color,
    )
    bracket!(
        ax, Point2(0, 1.0), Point2(2π, 1.0);
        color = ucolor_main, text = L"u(x; t_i) (\text{scaled})", linewidth = 2,
    )
    bracket!(
        ax, Point2(2π, 1.0), Point2(4π, 1.0);
        color = lambdacolor_main, text = L"\lambda(x; t_i)", linewidth = 2,
    )
    arrow1_heigth = 1.1
    arrows2d!(ax, Point2(2π / 32 * (N_mini * 2 + 1 - 0.5), arrow1_heigth), Point2(0, -0.57))
    text!(
        ax, "detected: 2";
        position = Point2(2π / 32 * (N_mini * 2 + 1 - 0.5), arrow1_heigth),
        align = (:right, :bottom),
    )
    arrow2_heigth = 1.25
    arrows2d!(ax, Point2(2π / 32 * (N_mini * 2 + 2 - 0.5), arrow2_heigth), Point2(0, -0.47))
    text!(
        ax, "target: 3";
        position = Point2(2π / 32 * (N_mini * 2 + 2 - 0.5), arrow2_heigth),
        align = (:right, :bottom),
    )
    bracket!(
        ax, Point2(2π, 0.0), Point2(0, 0.0);
        color = :black, text = L"x \in [0, 2\pi]", orientation = :down, linewidth = 2,
    )
    bracket!(
        ax, Point2(4π, 0.0), Point2(2π, 0.0);
        color = :black, text = L"x \in [0, 2\pi]", orientation = :down, linewidth = 2,
    )
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, -0.05, 4π + 1)
    ylims!(ax, -0.4, 1.4)
    rowgap!(fig.layout, 8)
    return fig
end
##
out_dir = mkpath(plotsdir("comparison_split"))
foreach((".png", ".pdf")) do suffix
    wsave(joinpath(out_dir, "observation_diagram" * suffix), fig; px_per_unit = 300 / inch)
end
println("saved observation_diagram to ", out_dir)
