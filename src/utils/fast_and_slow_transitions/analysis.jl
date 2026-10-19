function plot_linear_pressure_change_phase_space(results::Vector{LinearPressureChangeResults})
    fig = Figure()
    ax = Axis(fig[1, 1])
    for result in results
        scatter!(ax, result.ts, result.u_ps, color = result.shocks)
    end
    return fig
end
