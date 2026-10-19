using GeometryBasics: Point3f, GLTriangleFace, Vec3f
using LinearAlgebra: cross, dot, I
import SciMLBase

"""
    pie3d(values; kwargs...)
    pie3d(point, values; kwargs...)
    pie3d(x, y, z, values; kwargs...)

Creates a 3D pie chart with extruded wedges.
"""
Makie.@recipe Pie3D (x, y, z, values) begin
    "If `true`, the sum of all values is normalized to 2π (a full circle)."
    normalize = true
    "Color for each slice. Can be a single color, vector of colors, or colormap symbol."
    color = :Set1_4
    "Outer radius of the pie segments."
    radius = 1.0
    "Inner radius of the pie segments. If > 0, creates a ring/donut shape."
    inner_radius = 0.0
    "Height/depth of the 3D extrusion. Can be scalar or vector per slice."
    height = 0.5
    "Angular offset of the first pie segment from the (1, 0) vector in radians."
    offset = 0.0
    "Controls how many polygon vertices are used for one degree of rotation."
    vertex_per_deg = 1
    "Gap between slices in radians."
    gap = 0.0
    "Radial offset of slices from center. Can be scalar or vector per slice."
    offset_radius = 0.0
    "Normal vector defining pie orientation (which way the top face points). Default is Z-up."
    normal = Vec3f(0, 0, 1)
    # NOTE: Makie.mixin_* are non-public; keep until public alternative exists.
    Makie.mixin_generic_plot_attributes()...
    Makie.mixin_shading_attributes()...
end

# Disambiguate from SciMLBaseMakieExt and Makie stats recipes (no varargs — those caused new ambiguities)
function Makie.convert_arguments(
    PT::Type{<:Pie3D},
    sol::SciMLBase.AbstractTimeseriesSolution;
    plot_analytic = false,
    denseplot = Makie.automatic,
    plotdensity = Makie.automatic,
    plotat = nothing,
    tspan = nothing,
    tscale = :identity,
    vars = nothing,
    idxs = nothing,
    scatter_discretes = true,
)
    error("pie3d does not accept ODE solutions; pass `values::AbstractVector{<:Real}`")
end

# Convert arguments: just values -> default to origin
function Makie.convert_arguments(PT::Type{<:Pie3D}, values::AbstractVector{<:Real})
    return Makie.convert_arguments(PT, 0.0, 0.0, 0.0, values)
end

# Convert arguments: Point3 + values
function Makie.convert_arguments(PT::Type{<:Pie3D}, point::VecTypes{3}, values::AbstractVector{<:Real})
    return Makie.convert_arguments(PT, point[1], point[2], point[3], values)
end

# Convert arguments: x, y, z, values (canonical form)
function Makie.convert_arguments(::Type{<:Pie3D}, x::Real, y::Real, z::Real, values::AbstractVector{<:Real})
    return (Float32(x), Float32(y), Float32(z), float.(values))
end

function Makie.plot!(plot::Pie3D)
    # Compute merged mesh with vertex colors reactively
    Base.map!(
        plot,
        [
            :x,
            :y,
            :z,
            :values,
            :normalize,
            :radius,
            :inner_radius,
            :height,
            :offset,
            :vertex_per_deg,
            :gap,
            :offset_radius,
            :normal,
            :color,
        ],
        :_pie_mesh,
    ) do x, y, z, values, normalize, radius, inner_radius, height, offset, vertex_per_deg, gap, offset_radius, normal, color
        n_slices = length(values)
        center = Point3f(x, y, z)

        # Compute rotation matrix from normal vector
        rotation_matrix = _rotation_matrix_from_normal(Vec3f(normal))

        # Normalize values to angles
        T = eltype(values)
        summed = cumsum([zero(T); values])
        total_gap = gap * n_slices
        available_angle = normalize ? (2π - total_gap) : (summed[end] - total_gap)
        boundaries = if normalize
            summed ./ summed[end] .* available_angle
        else
            summed
        end

        # Handle offset_radius as scalar or vector
        offset_radius_vec = offset_radius isa Real ? fill(Float32(offset_radius), n_slices) : Float32.(offset_radius)

        # Handle height as scalar or vector
        height_vec = height isa Real ? fill(Float32(height), n_slices) : Float32.(height)

        # Compute colors for each slice
        slice_colors = _compute_slice_colors(n_slices, color)

        # Merge all slice meshes into one with vertex colors
        all_vertices = Point3f[]
        all_faces = GLTriangleFace[]
        all_colors = Makie.RGBAf[]
        vertex_offset = 0

        for i in 1:n_slices
            start_angle = boundaries[i] + offset + gap * (i - 1)
            end_angle = boundaries[i + 1] + offset + gap * (i - 1)

            # Calculate radial offset for this slice (in local XY plane, then rotate)
            mid_angle = (start_angle + end_angle) / 2
            local_offset = Vec3f(offset_radius_vec[i] * cos(mid_angle), offset_radius_vec[i] * sin(mid_angle), 0)
            rotated_offset = rotation_matrix * local_offset
            slice_center = center + Point3f(rotated_offset...)

            vertices, faces = _create_pie_wedge_geometry(
                start_angle,
                end_angle,
                Float32(radius),
                Float32(inner_radius),
                height_vec[i],
                slice_center,
                vertex_per_deg,
                rotation_matrix,
            )

            # Add vertices
            append!(all_vertices, vertices)

            # Add faces with offset
            for face in faces
                push!(
                    all_faces, GLTriangleFace(
                        face[1] + vertex_offset,
                        face[2] + vertex_offset,
                        face[3] + vertex_offset,
                    )
                )
            end

            # Add colors for each vertex
            slice_color = Makie.to_color(slice_colors[i])
            append!(all_colors, fill(slice_color, length(vertices)))

            vertex_offset += length(vertices)
        end

        # Return vertices, faces, and colors as a named tuple
        return (; vertices = all_vertices, faces = all_faces, colors = all_colors)
    end

    # Create separate observables for mesh! arguments using lift on the computed attribute
    vertices_obs = Makie.lift(d -> d.vertices, plot._pie_mesh)
    faces_obs = Makie.lift(d -> d.faces, plot._pie_mesh)
    colors_obs = Makie.lift(d -> d.colors, plot._pie_mesh)

    # Plot the merged mesh with shading attributes from mixin
    mesh!(
        plot, vertices_obs, faces_obs;
        color = colors_obs,
        shading = plot.shading,
        diffuse = plot.diffuse,
        specular = plot.specular,
        shininess = plot.shininess,
        backlight = plot.backlight,
        ssao = plot.ssao,
    )

    return plot
end

# Helper to compute slice colors from various color specifications
function _compute_slice_colors(n::Int, color)
    if color isa AbstractVector
        if length(color) >= n
            return color[1:n]
        else
            return repeat(color, ceil(Int, n / length(color)))[1:n]
        end
    elseif color isa Symbol
        # First try as a named color, then as a colormap
        try
            c = Makie.to_color(color)
            return fill(c, n)
        catch
            # Not a named color, try as colormap
            cmap = Makie.to_colormap(color)
            return [cmap[round(Int, 1 + (i - 1) * (length(cmap) - 1) / max(1, n - 1))] for i in 1:n]
        end
    elseif color isa AbstractString
        # String color (named or hex)
        c = Makie.to_color(color)
        return fill(c, n)
    else
        return fill(color, n)
    end
end

# Computes a 3x3 rotation matrix from Z-up to the given normal using Rodrigues' formula.
function _rotation_matrix_from_normal(normal::Vec3f)
    n = LinearAlgebra.normalize(normal)
    z_axis = Vec3f(0, 0, 1)
    dot_product = dot(n, z_axis)

    if dot_product > 0.9999f0
        # Normal is ~= +Z, identity
        return Float32[1 0 0; 0 1 0; 0 0 1]
    elseif dot_product < -0.9999f0
        # Normal is ~= -Z, rotate 180° around X axis to flip upside down
        return Float32[1 0 0; 0 -1 0; 0 0 -1]
    end

    # Rodrigues' rotation formula
    axis = LinearAlgebra.normalize(cross(z_axis, n))
    angle = acos(clamp(dot_product, -1.0f0, 1.0f0))
    K = Float32[
        0 -axis[3] axis[2]
        axis[3] 0 -axis[1]
        -axis[2] axis[1] 0
    ]

    return Float32.(I + sin(angle) * K + (1 - cos(angle)) * K * K)
end

# Creates geometry (vertices, faces) for a single pie wedge/slice.
function _create_pie_wedge_geometry(
        start_angle::Real,
        end_angle::Real,
        radius::Float32,
        inner_radius::Float32,
        height::Float32,
        center::Point3f,
        vertex_per_deg::Real,
        rotation_matrix::Matrix{Float32},
    )
    distance = end_angle - start_angle
    n = max(2, ceil(Int, rad2deg(distance) * vertex_per_deg))
    angles = LinRange(start_angle, end_angle, n)

    has_inner = inner_radius > 0

    # Helper to transform a local point to world coordinates
    function to_world(local_pt::Vec3f)
        rotated = rotation_matrix * local_pt
        return Point3f(center[1] + rotated[1], center[2] + rotated[2], center[3] + rotated[3])
    end

    # Build vertices in local coordinates (Z-up), then transform
    vertices = Point3f[]

    # Outer arc vertices - bottom (indices 1:n)
    for θ in angles
        local_pt = Vec3f(radius * cos(θ), radius * sin(θ), 0)
        push!(vertices, to_world(local_pt))
    end

    # Outer arc vertices - top (indices n+1:2n)
    for θ in angles
        local_pt = Vec3f(radius * cos(θ), radius * sin(θ), height)
        push!(vertices, to_world(local_pt))
    end

    if has_inner
        # Inner arc vertices - bottom (indices 2n+1:3n)
        for θ in angles
            local_pt = Vec3f(inner_radius * cos(θ), inner_radius * sin(θ), 0)
            push!(vertices, to_world(local_pt))
        end

        # Inner arc vertices - top (indices 3n+1:4n)
        for θ in angles
            local_pt = Vec3f(inner_radius * cos(θ), inner_radius * sin(θ), height)
            push!(vertices, to_world(local_pt))
        end
    else
        # Center vertices for solid pie
        push!(vertices, to_world(Vec3f(0, 0, 0)))       # bottom center (index 2n+1)
        push!(vertices, to_world(Vec3f(0, 0, height)))  # top center (index 2n+2)
    end

    faces = GLTriangleFace[]

    # Index helpers
    outer_bottom(i) = i
    outer_top(i) = n + i
    inner_bottom(i) = 2n + i
    inner_top(i) = 3n + i
    center_bottom = 2n + 1
    center_top = 2n + 2

    # === OUTER CURVED SURFACE ===
    for i in 1:(n - 1)
        push!(faces, GLTriangleFace(outer_bottom(i), outer_bottom(i + 1), outer_top(i + 1)))
        push!(faces, GLTriangleFace(outer_bottom(i), outer_top(i + 1), outer_top(i)))
    end

    if has_inner
        # === INNER CURVED SURFACE ===
        for i in 1:(n - 1)
            push!(faces, GLTriangleFace(inner_bottom(i), inner_top(i + 1), inner_bottom(i + 1)))
            push!(faces, GLTriangleFace(inner_bottom(i), inner_top(i), inner_top(i + 1)))
        end

        # === TOP FACE (ring) ===
        for i in 1:(n - 1)
            push!(faces, GLTriangleFace(outer_top(i), outer_top(i + 1), inner_top(i + 1)))
            push!(faces, GLTriangleFace(outer_top(i), inner_top(i + 1), inner_top(i)))
        end

        # === BOTTOM FACE (ring) ===
        for i in 1:(n - 1)
            push!(faces, GLTriangleFace(outer_bottom(i), inner_bottom(i + 1), outer_bottom(i + 1)))
            push!(faces, GLTriangleFace(outer_bottom(i), inner_bottom(i), inner_bottom(i + 1)))
        end

        # === SIDE FACES (radial edges) ===
        push!(faces, GLTriangleFace(outer_bottom(1), outer_top(1), inner_top(1)))
        push!(faces, GLTriangleFace(outer_bottom(1), inner_top(1), inner_bottom(1)))
        push!(faces, GLTriangleFace(outer_bottom(n), inner_top(n), outer_top(n)))
        push!(faces, GLTriangleFace(outer_bottom(n), inner_bottom(n), inner_top(n)))
    else
        # === SOLID PIE (no inner radius) ===

        # === TOP FACE (fan from center) ===
        for i in 1:(n - 1)
            push!(faces, GLTriangleFace(center_top, outer_top(i), outer_top(i + 1)))
        end

        # === BOTTOM FACE (fan from center) ===
        for i in 1:(n - 1)
            push!(faces, GLTriangleFace(center_bottom, outer_bottom(i + 1), outer_bottom(i)))
        end

        # === SIDE FACES ===
        push!(faces, GLTriangleFace(center_bottom, outer_bottom(1), outer_top(1)))
        push!(faces, GLTriangleFace(center_bottom, outer_top(1), center_top))
        push!(faces, GLTriangleFace(center_bottom, outer_top(n), outer_bottom(n)))
        push!(faces, GLTriangleFace(center_bottom, center_top, outer_top(n)))
    end

    return (vertices, faces)
end
