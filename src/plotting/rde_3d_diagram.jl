using Makie: Figure, GridLayout, Label, Observable, Toggle, connect!, lift, on,
    surface!, arrows3d!, Outside, LScene, update_cam!
using Makie: @lift
using GeometryBasics: Point3f
using Printf: @sprintf

"""
    plot_rde_data_3d!(parent, u, control_heights; kwargs...) -> sc

Plot the RDE 3D diagram (annular cylinder, shock surface, exhaust arrows, injection pie,
wave and injection arrows) into `parent`.

`parent` can be an `LScene` (plot into it) or a `GridPosition` (e.g. `fig[i,j]`); in the
latter case an `LScene` is created in that cell with no axis.

`u` is the state variable whose normalized values define the shock surface height in the
annulus. It can be an `AbstractVector{<:Real}` or an `Observable{<:AbstractVector{<:Real}}`;
when it is an Observable, the shock surface updates when the observable changes.
`length(u)` must equal `n_angular`.

`control_heights` are the extrusion heights of each pie segment (one value per section).
It can be an `AbstractVector{<:Real}` or an `Observable` of such; when Observable, the
pie heights update reactively. `length(control_heights)` must equal `n_sections`.

# Keywords
- `inner_radius = 1.0`, `outer_radius = 1.1`, `cylinder_height = 2.0`: geometry
- `n_angular = 512`: number of angular points (cylinder and shock); must match `length(u)`
- `n_sections = 16`: number of pie segments; must match `length(control_heights)`
- `injection_pie_z = -1.0`: z position of the injection pie
- `pie_gap`: gap between pie slices in radians (default `0.1 * (2π / n_sections)`)
- `pie_color = :blue`: color of the injection pie
- `n_exhaust_arrows = 10`, `exhaust_arrow_lengthscale = 0.4`: exhaust arrows at top
- `wave_arrow_lengthscale = 0.2`, `wave_arrow_heights = 0.7`: red wave arrow scale and z-height(s)
- `injection_arrow_lengthscale = 0.2`: injection arrow scale
- `shock_color = :red`, `shock_alpha = 1.0`: shock surface appearance
- `cylinder_alpha = 0.3`: cylinder surface alpha
- `pie_offset = 0.0`: angular offset of the first pie segment in radians (e.g. for moving frame). Can be scalar or `Observable`.
"""
function plot_rde_data_3d!(
        parent,
        u,
        control_heights;
        inner_radius = 1.0f0,
        outer_radius = 1.1f0,
        cylinder_height = 2.0f0,
        cylinder_color = :grey,
        n_angular = 512,
        n_sections = 16,
        injection_pie_z = -1.0f0,
        pie_gap = 0.1f0 * (2π / 16),
        pie_offset = 0.0f0,
        pie_color = :blue,
        n_exhaust_arrows = 10,
        exhaust_arrow_lengthscale = 0.4f0,
        wave_arrow_lengthscale = 0.2f0,
        wave_arrow_heights = 0.7f0,
        wave_arrow_positions = [0.0, 2π / 3, 4π / 3] .- 2π / 7,
        injection_arrow_lengthscale = 0.2f0,
        shock_color = :red,
        shock_alpha = 1.0f0,
        cylinder_alpha = 0.3f0,
        wave_arrows = true,
        exhaust_arrows = true,
        control_arrows = true,
        exhaust_color = :orange,
        eyeposition = Point3f(-1.9f0, -1.9f0, 1.5f0),
        lookat = Point3f(0.0f0, 0.0f0, 0.5f0),
    )
    sc = parent isa LScene ? parent : LScene(parent[1, 1]; show_axis = false)
    u_obs = u isa Observable ? u : Observable(collect(Float32, u))
    control_heights_obs =
        control_heights isa Observable ? control_heights : Observable(collect(Float32, control_heights))
    pie_offset_obs =
        pie_offset isa Observable ? pie_offset : Observable(Float32(pie_offset))


    middle_radius = (inner_radius + outer_radius) / 2
    θs = LinRange(0.0f0, 2π, n_angular)
    inner_xs = inner_radius .* cos.(θs)
    inner_ys = inner_radius .* sin.(θs)
    outer_xs = outer_radius .* cos.(θs)
    outer_ys = outer_radius .* sin.(θs)
    lower_zs = zeros(Float32, n_angular)
    upper_zs = cylinder_height .* ones(Float32, n_angular)

    # Cylinder surfaces (static)
    for (xs, ys) in ((inner_xs, inner_ys), (outer_xs, outer_ys))
        surface!(
            sc,
            hcat(xs, xs),
            hcat(ys, ys),
            hcat(lower_zs, upper_zs);
            color = fill(cylinder_color, n_angular, 2),
            alpha = cylinder_alpha,
            ssao = true,
            transparency = true,
        )
    end

    # Shock surface (reactive): normalize u to [0,1], avoid division by zero
    u_norm_obs = lift(u_obs) do u_val
        u_vec = collect(Float32, u_val)
        min_u = minimum(u_vec)
        max_u = maximum(u_vec)
        if max_u > min_u
            return (u_vec .- min_u) ./ (max_u - min_u)
        else
            return fill(0.5f0, length(u_vec))
        end
    end
    z_shock_obs = lift(u_norm_obs) do v
        hcat(v, v)
    end
    x_shock = hcat(inner_xs, outer_xs)
    y_shock = hcat(inner_ys, outer_ys)
    surface!(
        sc,
        x_shock,
        y_shock,
        z_shock_obs;
        color = fill(shock_color, n_angular, 2),
        alpha = shock_alpha,
        ssao = true,
        transparency = true,
    )

    # Exhaust arrows (static)
    arrow_θs = LinRange(0.0, 2π, n_exhaust_arrows + 1)[1:(end - 1)]
    arrow_xs = middle_radius .* cos.(arrow_θs)
    arrow_ys = middle_radius .* sin.(arrow_θs)
    arrow_zs = cylinder_height .* ones(n_exhaust_arrows)
    arrows3d!(
        sc,
        arrow_xs,
        arrow_ys,
        arrow_zs,
        zeros(n_exhaust_arrows),
        zeros(n_exhaust_arrows),
        ones(n_exhaust_arrows);
        lengthscale = exhaust_arrow_lengthscale,
        color = exhaust_color,
        visible = exhaust_arrows,
    )

    # Injection pie (reactive heights and offset)
    pie3d!(
        sc,
        0.0f0,
        0.0f0,
        injection_pie_z,
        ones(Float32, n_sections);
        color = pie_color,
        inner_radius = inner_radius,
        radius = outer_radius,
        gap = pie_gap,
        offset = pie_offset_obs,
        height = control_heights_obs,
    )
    # Wave direction arrows (static)
    angle_unit_vec(θ) = Point3f(-sin(θ), cos(θ), 0)
    arrow_u_θs = wave_arrow_positions
    arrow_u_z = wave_arrow_heights
    arrow_u_positions =
        Point3f.(middle_radius .* cos.(arrow_u_θs), middle_radius .* sin.(arrow_u_θs), arrow_u_z)
    arrows3d!(
        sc,
        arrow_u_positions,
        angle_unit_vec.(arrow_u_θs);
        lengthscale = wave_arrow_lengthscale,
        color = :red,
        visible = wave_arrows,
    )
    # Injection rotation arrows (static)
    n_arrows_inj = 4
    arrow_inj_θs = LinRange(0.0, 2π, n_arrows_inj + 1)[1:(end - 1)]
    arrow_inj_radius = outer_radius * 1.1f0
    arrow_inj_positions = Point3f.(
        arrow_inj_radius .* cos.(arrow_inj_θs),
        arrow_inj_radius .* sin.(arrow_inj_θs),
        injection_pie_z .* ones(n_arrows_inj),
    )
    arrow_inj_directions = angle_unit_vec.(arrow_inj_θs)
    arrows3d!(
        sc,
        arrow_inj_positions,
        arrow_inj_directions;
        lengthscale = injection_arrow_lengthscale,
        color = pie_color,
        visible = control_arrows,
    )

    # Camera: view from front (default LScene often shows the back).
    # center = false so that we keep this view; center!(scene) would otherwise
    # overwrite eyeposition/lookat and fit the full data bbox (large white space).
    cam3d!(sc; center = false)
    update_cam!(
        sc.scene,
        eyeposition,
        lookat,
    )
    return sc
end

function _dense_to_sparse_ind(dense_time::Vector, sparse_time::Vector, dense_ind::Int)
    dense_val = dense_time[dense_ind]
    return coalesce(findlast(x -> x <= dense_val, sparse_time), 1)
end

"""
    plot_rde_data_3d!(parent, data, params; time_idx, show_title, show_controls, kwargs...) -> parent

Plot the RDE 3D diagram from `PolicyRunData` into `parent` (a `GridLayout`).

Builds observables for state and control from `data`, using `params::RDE.RDEParam` for
geometry (N = `params.N`, dx = `params.L / params.N`). Optional title row and controls
(playback + moving-frame toggle) can be enabled with `show_title` and `show_controls`.
Remaining kwargs are passed to the inner low-level `plot_rde_data_3d!`.

# Keywords
- `time_idx = Observable(1)`: observable time index into `data.state_ts`
- `show_title = true`: whether to show the time + moving-frame label above the diagram
- `show_controls = true`: whether to show playback and frame toggle below the diagram
- Remaining kwargs (e.g. `wave_arrows`, `control_arrows`) are forwarded to the 3D diagram
"""
function plot_rde_data_3d!(
        parent,
        data::PolicyRunData,
        params::RDE.RDEParam;
        time_idx = Observable(1),
        show_title = true,
        show_controls = true,
        max_speed = 4.6f0,
        injection_pie_z = -1.0f0,
        kwargs...,
    )
    N = params.N
    dx = params.L / params.N
    n_field = eltype(data.u_ps) <: AbstractVector ? length(data.u_ps[1]) : 1
    n_sections = n_field == N ? 16 : n_field
    section_size = N ÷ n_sections

    has_moving_frame =
        !isempty(data.control_shifts) && data.control_shifts[1] isa MovingFrameControlShift
    c_vector = get_plotting_speed_adjustments(data, dx; max_speed = max_speed)
    plot_frame_pos = [zero(eltype(c_vector)); cumsum(c_vector .* diff(data.state_ts))]
    control_shift_positions = if has_moving_frame
        control_ts = getproperty.(data.control_shifts, :t_last)
        map(enumerate(data.state_ts)) do (i, t)
            control_ix = findlast(ct -> ct <= t, control_ts)
            isnothing(control_ix) && return zero(eltype(data.state_ts))
            mod2pi(RDE.get_control_shift(data.control_shifts[control_ix], Float32[], t))
        end
    else
        zeros(eltype(data.state_ts), length(data.state_ts))
    end

    control_frame_position = @lift(control_shift_positions[$time_idx])
    lab_to_plot_shift = @lift(Int(round(plot_frame_pos[$time_idx] / dx)))
    moving_frame_toggle = Observable(true)
    plot_shift = @lift($moving_frame_toggle ? $lab_to_plot_shift : 0)
    plot_position = @lift($moving_frame_toggle ? plot_frame_pos[$time_idx] : 0.0f0)

    raw_u = @lift(data.states[$time_idx][1:N])
    u = @lift(circshift($raw_u, -$plot_shift))

    sparse_time_idx =
        @lift(coalesce(_dense_to_sparse_ind(data.state_ts, data.action_ts, $time_idx), 1))

    if eltype(data.u_ps) <: AbstractVector
        pie_offset = @lift($control_frame_position - $plot_position)
        control = @lift begin
            field = data.u_ps[min(length(data.u_ps), $time_idx)]
            if length(field) == N
                RDE_Env.section_midpoint_values(field, n_sections)
            else
                field
            end
        end
    else
        pie_offset = Observable(0.0f0)
        control = @lift([data.u_ps[min(length(data.u_ps), $time_idx)]])
    end

    max_ups = maximum(eltype(data.u_ps) <: AbstractVector ? maximum.(data.u_ps) : data.u_ps)
    scaled_control = @lift($control / max_ups * abs(injection_pie_z))

    gl_rd = parent[1, 1] = GridLayout()
    plot_rde_data_3d!(
        gl_rd,
        u,
        scaled_control;
        n_angular = N,
        n_sections = n_sections,
        pie_offset = pie_offset,
        injection_pie_z = injection_pie_z,
        kwargs...,
    )

    if show_title
        function _update_time_label!(time_label_obs, d, ti_obs, mf_obs)
            ti = ti_obs[]
            mf = mf_obs[]
            t_str = @sprintf("%.2f", d.state_ts[ti])
            time_label_obs[] = "Time: $(t_str)" * (mf ? " [Moving Frame]" : "")
            return nothing
        end
        time_label_text = Observable("")
        on(_ -> _update_time_label!(time_label_text, data, time_idx, moving_frame_toggle), time_idx)
        on(
            _ -> _update_time_label!(time_label_text, data, time_idx, moving_frame_toggle),
            moving_frame_toggle,
        )
        _update_time_label!(time_label_text, data, time_idx, moving_frame_toggle)
        Label(parent[0, 1], text = time_label_text, tellwidth = false)
    end

    if show_controls
        gl_ctrl = parent[2, 1] = GridLayout()
        RDE.plot_controls(gl_ctrl, time_idx, length(data.state_ts))
        toggle_layout = gl_ctrl[3, 1] = GridLayout()
        Label(toggle_layout[1, 1], text = "Frame:", halign = :right, tellwidth = false)
        frame_toggle = Toggle(toggle_layout[1, 2], active = moving_frame_toggle[], tellwidth = false)
        Label(
            toggle_layout[1, 3],
            text = lift(a -> a ? "Moving" : "Lab", frame_toggle.active),
            halign = :left,
            tellwidth = false,
        )
        connect!(moving_frame_toggle, frame_toggle.active)
    end

    return parent
end

"""
    plot_rde_data_3d(data, params; time_idx, show_title, show_controls, size, kwargs...) -> fig

Create a new Figure with the RDE 3D diagram from `PolicyRunData`.

Calls `plot_rde_data_3d!(layout, data, params; ...)` on a new single-column layout.
Returns the figure.

# Keywords
- `time_idx = Observable(1)`: observable time index
- `show_title = true`: whether to show the time + moving-frame label
- `show_controls = true`: whether to show playback and frame toggle
- `size = (900, 900)`: figure size in pixels
- Remaining kwargs are passed to the inner 3D diagram
"""
function plot_rde_data_3d(
        data::PolicyRunData,
        params::RDE.RDEParam;
        size = (900, 900),
        backgroundcolor = :white,
        kwargs...,
    )
    fig = Figure(; size = size, figure_padding = 0, backgroundcolor)
    layout = fig[1, 1] = GridLayout()
    plot_rde_data_3d!(
        layout,
        data,
        params;
        kwargs...,
    )
    return fig
end
