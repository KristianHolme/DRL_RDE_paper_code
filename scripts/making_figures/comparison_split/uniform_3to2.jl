# SU-only trajectory for the 3→2 transition, matching Fig. 6 (two-step-example) style.
# Uses cached representative SU run (target 2, seed 84, 3→2 trial) so no retraining/rerun needed.
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using CairoMakie
CairoMakie.activate!()
using JLD2

##
cache_path = datadir("studies", "ar14_all_target", "fair_clock", "uniform_3to2_trajectory_t380.jld2")
su_data, su_x = load(cache_path, "su_data", "su_x")
@info "SU trajectory" length(su_data.state_ts) su_data.state_ts[end]

##
u_ax_kwargs = (
    yticks = ([0, π / 2, π, 3π / 2, 2π], [L"0", L"\pi/2", L"\pi", L"3\pi/2", L"2\pi"]),
    limits = (nothing, (0.0f0, 2.0f0 * π)),
    ylabel = L"x - \psi(t) \,\mathrm{mod}\, 2\pi",
)
dpi = 300
u_hm_kwargs = (; rasterize = ceil(Int, dpi * pt / inch))
fig = with_theme(theme_drl_rde()) do
    fig = plot_shifted_history(
        su_data, su_x;
        use_rewards = false,
        plot_shocks = false,
        u_ax_kwargs,
        fig_kwargs = (;
            size = (16cm, 9cm),
            figure_padding = (16, 16, 16, 32),
        ),
        u_hm_kwargs,
    )
    rowsize!(content(fig[1, 1]), 2, Relative(0.3))
    rowgap!(content(fig[1, 1]), 30)
    hidexdecorations!(fig.content[1], ticks = false)
    fig
end
display(fig)

##
out_dir = mkpath(plotsdir("comparison_split"))
foreach((".png", ".pdf")) do suffix
    wsave(joinpath(out_dir, "uniform_3to2" * suffix), fig; px_per_unit = dpi / inch)
end
println("saved uniform_3to2 (SU-only) to ", out_dir)
