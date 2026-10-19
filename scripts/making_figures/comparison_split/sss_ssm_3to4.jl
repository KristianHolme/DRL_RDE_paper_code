# SSS vs SSM lab-frame snapshot for the 3→4 transition.
# Recreates the paper figure (no original script in the repo).
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using CairoMakie
CairoMakie.activate!()
include(joinpath(@__DIR__, "plot_knobs.jl"))
##
function snapshot_at(data, t)
    i = findfirst(≥(t), data.state_ts)
    isnothing(i) && error("No state at or after t=$t (tmax=$(data.state_ts[end]))")
    u, λ = RDE.split_sol(data.states[i])
    return u, λ, data.u_ps[i], data.state_ts[i]
end

function plot_ulambda_up!(ax, x, u, λ, control; n_sections, u_color, lambda_color, control_color)
    up = control isa AbstractVector ? collect(control) : [control]
    bars = length(up) == n_sections ? up : section_midpoint_values(up, n_sections)
    barplot!(ax, (2π / n_sections) .* ((1:n_sections) .- 0.5), bars; color = control_color, gap = 0.05)
    lines!(ax, x, u; color = u_color)
    lines!(ax, x, λ; color = lambda_color)
    return ax
end
##
delay = 20.0f0
after_activation = 85.0f0
snapshot_t = delay + after_activation
tmax = 200.0f0
initial_shock = 3
target_shock_count = 4
u_color = categorical_colors[1]
lambda_color = :tomato
control_color = categorical_colors[3]
##
sss_policy, sss_env, sss_row = get_best_policy(
    "SAS3";
    filter_function = r -> r.movingframe == false && r.target_shock_count == 4 && r.dt ≈ 0.25f0,
    initial_shock,
    target_shock_count,
    delay,
    tmax,
)
ssm_policy, ssm_env, ssm_row = get_best_policy(
    "SAS3";
    filter_function = r -> r.movingframe == true && r.target_shock_count == 4 && r.dt ≈ 4.0f0,
    initial_shock,
    target_shock_count,
    delay,
    tmax,
)
@info "SSS row" sss_row.dt sss_row.movingframe sss_row.target_shock_count
@info "SSM row" ssm_row.dt ssm_row.movingframe ssm_row.target_shock_count
sss_data = run_policy(sss_policy, sss_env)
ssm_data = run_policy(ssm_policy, ssm_env)
##
sss_u, sss_λ, sss_up, sss_t = snapshot_at(sss_data, snapshot_t)
ssm_u, ssm_λ, ssm_up, ssm_t = snapshot_at(ssm_data, snapshot_t)
@info "Snapshot times" sss_t ssm_t snapshot_t
##
xticks = ([0, π / 2, π, 3π / 2, 2π], [L"0", L"π/2", L"π", L"3π/2", L"2π"])
fig = with_theme(comparison_theme()) do
    fig = Figure(; size = (16cm, 10cm), backgroundcolor = bgcolor)
    ax_sss = Axis(
        fig[1, 1];
        title = "SSS",
        ylabel = L"u,\;\lambda,\;u_p",
        xticks,
        limits = ((0, 2π), nothing),
    )
    ax_ssm = Axis(
        fig[2, 1];
        title = "SSM",
        xlabel = L"x",
        ylabel = L"u,\;\lambda,\;u_p",
        xticks,
        limits = ((0, 2π), nothing),
    )
    hidexdecorations!(ax_sss; ticks = false, grid = false)
    plot_ulambda_up!(
        ax_sss, sss_env.prob.x, sss_u, sss_λ, sss_up;
        n_sections = sss_env.action_strat.n_sections,
        u_color, lambda_color, control_color,
    )
    plot_ulambda_up!(
        ax_ssm, ssm_env.prob.x, ssm_u, ssm_λ, ssm_up;
        n_sections = ssm_env.action_strat.n_sections,
        u_color, lambda_color, control_color,
    )
    linkyaxes!(ax_sss, ax_ssm)
    lu = LineElement(; color = u_color)
    ll = LineElement(; color = lambda_color)
    p = PolyElement(; color = control_color)
    Legend(
        fig[0, 1],
        [lu, ll, p],
        [L"u(x,t)", L"λ(x,t)", L"u_p(x,t)"];
        orientation = :horizontal,
        tellwidth = false,
    )
    rowgap!(fig.layout, 8)
    return fig
end
##
out_dir = mkpath(plotsdir("comparison_split"))
foreach((".png", ".pdf")) do suffix
    wsave(joinpath(out_dir, "sss_ssm_3to4" * suffix), fig; px_per_unit = dpi / inch)
end
println("saved sss_ssm_3to4 to ", out_dir)
