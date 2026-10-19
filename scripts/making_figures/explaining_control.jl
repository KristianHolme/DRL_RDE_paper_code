using DrWatson
@quickactivate :DRL_RDE
##
using CairoMakie
CairoMakie.activate!()
using Statistics
##
# using GLMakie
# GLMakie.activate!()
##

"""
    moving_frame_state(data::PolicyRunData, x::AbstractVector)

Shift lab-frame state from policy run data into the **agent's** moving reference frame.

Uses only `data.control_shifts` (MovingFrameControlShift) and `RDE.get_control_shift` to get
the frame position at each time. This is the same frame the agent uses for observations and
control; it is **not** the plotting/detected frame from `get_plotting_speed_adjustments`.

# Arguments
- `data::PolicyRunData`: Data from `run_policy(π, env)` with moving-frame control (control_shifts)
- `x::AbstractVector`: Spatial grid (e.g. `env.prob.x`)

# Returns
- `(us, lambdas, u_ps)`: Each is a `Vector{Vector{T}}` of length `length(data.state_ts)` with
  velocity, reaction progress and injection-pressure profile in the agent's moving frame.

# Throws
- Requires non-empty `data.control_shifts` with first element `MovingFrameControlShift`.
"""
function moving_frame_state(data::PolicyRunData, dx)
    if isempty(data.control_shifts) || !(data.control_shifts[1] isa MovingFrameControlShift)
        throw(
            ArgumentError(
                "moving_frame_state requires data with agent moving frame: " *
                    "non-empty control_shifts and first element MovingFrameControlShift"
            )
        )
    end
    state_ts = data.state_ts
    control_ts = getproperty.(data.control_shifts, :t_last)
    control_shift_positions = map(enumerate(state_ts)) do (i, t)
        control_ix = findlast(ct -> ct <= t, control_ts)
        isnothing(control_ix) && return zero(eltype(state_ts))
        return RDE.get_control_shift(data.control_shifts[control_ix], Float32[], t)
    end
    us, lambdas = RDE.split_sol(data.states)
    shifts = [-Int(round(p / dx)) for p in control_shift_positions]
    shifted_us = [circshift(us[i], shifts[i]) for i in eachindex(us)]
    shifted_lambdas = [circshift(lambdas[i], shifts[i]) for i in eachindex(lambdas)]
    shifted_u_ps = [circshift(data.u_ps[i], shifts[i]) for i in eachindex(data.u_ps)]
    return (shifted_us, shifted_lambdas, shifted_u_ps)
end

##
delay = 15.0f0
tmax = 100.0f0
target_shock_count = 2
initial_shock = 3
##
experiment_name = "SAS3"
policy, env, _ = get_best_policy(
    experiment_name;
    filter_function = r -> r.movingframe == true && r.target_shock_count == 2,
    wrap_delay = false,
    initial_shock = 3,
    target_shock_count = 2,
    tmax = 50.0f0,
)
data = run_policy(policy, env)
shifted_us, shifted_lambdas, shifted_u_ps = moving_frame_state(data, env.prob.x[2] - env.prob.x[1])
n_sections = 16
# Mean injection pressure per control section (agent frame)
section_u_p(profile) = vec(mean(reshape(profile, :, n_sections); dims = 1))
##
function add_axis!(gl::GridLayout, time_point::Float32)
    ax = Axis(
        gl[1, 1], xticks = ([0, π / 2, π, 3π / 2, 2π], [L"0", L"π/2", L"π", L"3π/2", L"2π"]),
        xlabel = L"x - \Psi(t) \, \mathrm{mod} \,2\pi",
        title = "t = $(round(time_point, digits = 2))",
        titlegap = -20,
        titlealign = :center,
        limits = ((0, 2π), nothing)
    )
    return ax
end
function plot_step!(ax, u, lambda, control, colors)
    xs = env.prob.x
    control_centers = 2π / n_sections .* ((1:n_sections) .- 0.5)
    barplot!(ax, control_centers, control, color = colors.control, gap = 0.05)
    lines!(ax, xs, u, color = colors.u)
    lines!(ax, xs, lambda, color = colors.lambda)
    return ax
end
##
const wong = Makie.wong_colors()
fig = Figure(size = (16cm, 20cm))
time_points = [0.0f0, 0.5f0, 1.0f0, 1.5f0, 2.0f0, 2.5f0, 3.0f0, 3.5f0]
axes = Axis[]
u_color = wong[1]
lambda_color = :tomato
control_color = wong[3]
num_rows = (length(time_points) + 1) ÷ 2
for (i, time_point) in enumerate(time_points)
    column = ((i - 1) % 2) + 1
    row = (i + 1) ÷ 2
    u_idx = findfirst(data.state_ts .>= time_point)
    control = section_u_p(shifted_u_ps[u_idx])
    u = shifted_us[u_idx]
    lambda = shifted_lambdas[u_idx]
    gl = fig[row, column] = GridLayout()
    ax = add_axis!(gl, time_point)
    if row !== num_rows
        hidexdecorations!(ax, ticks = false, grid = false)
    end
    if column == 2
        hideydecorations!(ax, ticks = false, grid = false)
    end
    plot_step!(ax, u, lambda, control, (; u = u_color, lambda = lambda_color, control = control_color))
    tightlimits!(ax)
    push!(axes, ax)
end
linkyaxes!(fig.content...)
lu = LineElement(color = u_color)
ll = LineElement(color = lambda_color)
p = PolyElement(color = control_color)
Legend(fig[0, :], [lu, ll, p], [L"u(x,t)", L"λ(x,t)", L"u_p(x,t)"], framevisible = true, orientation = :horizontal)
rowgap!(fig.layout, 7)
##
dpi = 300
foreach([".svg", ".pdf"]) do ext
    wsave(plotsdir("figures", "explaining_control", "explaining_control$ext"), fig)
end
