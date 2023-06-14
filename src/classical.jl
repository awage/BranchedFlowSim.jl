# TODO: Figure out if this file should be in potentials.jl
# Currently this contains mostly potential functions and quasi2d simulation
# for branch counting.
using LinearAlgebra
using StaticArrays
using Makie
using Interpolations

export FermiDotPotential, LatticePotential, PeriodicGridPotential, RotatedPotential
export RepeatedPotential
export rotation_matrix, force, force_x, force_y
export quasi2d_num_branches, quasi2d_visualize_rays
export grid_eval

"""
We consider an object V to be a Potential if it is callable with two floats
(like V(x,y)) and has a defined function force(V, x, y) = -∇V. Plain functions
are supported with numerical differentiation, and some of the potentials
defined here can be composed together (LatticePotential, RepeatedPotential)
to create complex yet efficient potential functions.
"""
function ispotential(v)
    return applicable(v, 1.2, 2.3) && applicable(force, v, 1.2, 2.3)
end

struct FermiDotPotential
    radius::Float64
    α::Float64
    inv_α::Float64
    v0::Float64
    function FermiDotPotential(radius, v0, softness=0.2)
        α = softness * radius
        return new(radius, α, 1/α, v0)
    end
end

function (V::FermiDotPotential)(x::Real, y::Real)::Float64
    d = V.radius - sqrt(x^2 + y^2)
    return V.v0 / (1 + exp(-V.inv_α * d))
end

function force(V::FermiDotPotential, x::Real, y::Real)::SVector{2,Float64}
    d = sqrt(x^2 + y^2)
    if d <= 1e-9
        return SVector(0.0, 0.0)
    end
    z = exp(V.inv_α * (-V.radius + d))
    return SVector(x, y) * (V.v0 * z *V.inv_α/ (d * (1 + z)^2))
end

"""
LatticePotential

"""
struct LatticePotential{DotPotential}
    A::SMatrix{2,2,Float64, 4}
    A_inv::SMatrix{2,2,Float64, 4}
    dot_potential::DotPotential
    offset::SVector{2,Float64}
    four_offsets::SVector{4, SVector{2, Float64}}
    function LatticePotential(A, radius, v0; offset=[0, 0], softness=0.2)
        dot = FermiDotPotential(radius, v0, softness)
        fo = [ SVector(0.0, 0.0), A[:, 1], A[:, 2], A[:, 1] + A[:, 2]]
        return new{FermiDotPotential}(
            SMatrix{2,2}(A),
            SMatrix{2,2}(inv(A)),
             dot, SVector{2,Float64}(offset), fo)
    end
end

"""
    rotation_matrix(θ)

Returns a 2D rotation matrix for the given angle θ.
"""
function rotation_matrix(θ)
    return [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]
end

function (V::LatticePotential)(x::Real, y::Real)::Float64
    # Find 4 closest lattice vectors
    r = SVector(x, y) - V.offset
    a::SVector{2, Float64} = V.A_inv * r
    # ind = SVector{2, Float64}(floor(a[1]), floor(a[2]))
    ind::SVector{2, Float64} = floor.(a)
    v = 0.0
    R0 = V.A * ind
    for offset ∈ V.four_offsets
        R = R0 + offset
        v += V.dot_potential(r[1] - R[1], r[2] - R[2])
    end
    return v
end

function force(V::LatticePotential, x::Real, y::Real)::SVector{2,Float64}
    # Find 4 closest lattice vectors
    r = SVector{2, Float64}(x - V.offset[1], y - V.offset[2])
    a = (V.A_inv*r)::SVector{2, Float64}
    # ind = SVector{2, Float64}(floor(a[1]), floor(a[2]))
    ind::SVector{2, Float64} = floor.(a)
    F::SVector{2, Float64} = SVector(0.0, 0.0)
    R0::SVector{2, Float64} = V.A * ind
    for offset ∈ V.four_offsets
        rR::SVector{2,Float64} = r - (R0 + offset)
        F += @inline force(V.dot_potential, rR[1], rR[2])::SVector{2, Float64}
    end
    return F
end

struct PeriodicGridPotential
    itp::AbstractInterpolation
    """
    PeriodicGridPotential(xs, ys, arr)

Note: `arr` is indexed as arr[y,x].
"""
    function PeriodicGridPotential(xs, ys, arr::AbstractMatrix{Float64})
        itp = interpolate(transpose(arr), BSpline(Cubic(Periodic(OnCell()))))
        itp = extrapolate(itp, Periodic(OnCell()))
        itp = scale(itp, xs, ys)
        return new(itp)
    end
end

function (V::PeriodicGridPotential)(x::Real, y::Real)
    return V.itp(x, y)
end

"""
    force(V::PeriodicGridPotential, x::Real, y::Real)

"""
function force(V::PeriodicGridPotential, x::Real, y::Real)
    return -gradient(V.itp, x, y)
end

function force_x(V, x, y)
    return force(V, x, y)[1]
end
function force_y(V, x, y)
    return force(V, x, y)[2]
end

"""
    force_diff(V, x::Real, y::Real)

Generic implementation of `force` function using numeric differentiation.
Should only be used for testing.
"""
function force_diff(V, x::Real, y::Real)::SVector{2, Float64}
    h = 1e-6
    return -SVector(
        (@inline V(x + h, y) - @inline V(x - h, y)) / (2 * h),
        (@inline V(x, y + h) - @inline V(x, y - h)) / (2 * h)
    )
end

"""
    force(V::Function, x::Real, y::Real)

Generic implementation of `force` function for potentials defined as plain
functions. Uses numerical differentiation
"""
function force(V::Function, x::Real, y::Real)::SVector{2, Float64}
    return force_diff(V, x, y)
end

function force_y(V::Function, x::Real, y::Real)::Float64
    h = 1e-6
    return - (@inline V(x, y + h) - @inline V(x, y - h)) / (2 * h)
end

function force_x(V::Function, x::Real, y::Real)::Float64
    h = 1e-6
    return -(@inline V(x + h, y) - @inline V(x - h, y)) / (2 * h)
end

"""
    compare_force_with_diff(p)

Debugging function for comparing `force` implementation with `force_diff`
"""
function compare_force_with_diff(p)
    xs = LinRange(0, 1, 99)
    ys = LinRange(0, 1, 124)
    return [
        norm(force_diff(p, x, y) - force(p, x, y)) for y ∈ ys, x ∈ xs
    ]
    # fig, ax, plt = heatmap(xs, ys, (x, y) -> norm(force_diff(p, x, y) - force(p, x, y)))
    # Colorbar(fig[1, 2], plt)
    # fig
end

function count_zero_crossing(fx)
    c = 0
    for j ∈ 2:length(fx)
        if sign(fx[j-1]) != sign(fx[j])
            c += 1
        end
    end
    return c
end

function get_dynamic_rays(ray_y, ray_py, maxd)
    new_ray_y = [ray_y[1]]
    new_ray_py = [ray_py[1]]
    sizehint!(new_ray_y, length(ray_y))
    sizehint!(new_ray_py, length(ray_py))
    for j ∈ 2:length(ray_y)
        if abs(ray_y[j] - ray_y[j-1]) > maxd
            # Add new ray with mean values
            push!(new_ray_y, (ray_y[j-1] + ray_y[j]) / 2)
            push!(new_ray_py, (ray_py[j-1] + ray_py[j]) / 2)
            #          println("new ray!, t=$x, y=$(new_ray_y[end])")
        end
        push!(new_ray_y, ray_y[j])
        push!(new_ray_py, ray_py[j])
    end
    return new_ray_y, new_ray_py
end

function quasi2d_num_branches(xs, ys, ts, potential)
    dx = xs[2] - xs[1]
    dt = dx
    orig_dy = ys[2] - ys[1]
    ray_y = Vector(ys)
    ray_py = zeros(length(ys))
    num_branches = zeros(length(ts))
    ti = 1
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        if ts[ti] <= x
            # Count branches
            caustics = count_zero_crossing(ray_y[2:end] - ray_y[1:end-1])
            num_branches[ti] = caustics / 2
            ti += 1
        end
    end
    return num_branches
end

"""
    quasi2d_visualize_rays(path, xs, ys, potential;
        dynamic_rays=false,
        triple_y=false,
        height=(if triple_y 3*length(ys) else length(ys) end)
     )

TBW
"""
function quasi2d_visualize_rays(path, xs, ys, potential;
    dynamic_rays=false,
    triple_y=false
)
    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    sim_height = dy * length(ys)
    sim_width = dt * length(xs)
    height = round(Int,
        if triple_y
            3 * length(xs) * sim_height / sim_width
        else
            sim_height * length(xs) / sim_width
        end
    )
    image = zeros(height, length(xs))
    # Height of one pixel in simulation length units
    pixel_h = if triple_y
        3 * sim_height / height
    else
        sim_height / height
    end
    sim_pixels = sim_height / pixel_h
    ray_y = Vector(ys)
    ray_py = zeros(length(ys))
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        # Collect
        for y ∈ ray_y
            yi = 1 + round(Int, y / pixel_h)
            if triple_y
                yi += height ÷ 3
            end
            if yi >= 1 && yi <= height
                image[yi, xi] += 1
            end
        end
        if dynamic_rays && (xi % 10 == 0)
            maxd = 10 * dy
            ray_y, ray_py = get_dynamic_rays(ray_y, ray_py, maxd)
        end
    end
    scene = Scene(camera=campixel!, resolution=size(image'))
    ypixels = LinRange(ys[1], ys[end], height)
    if triple_y
        ypixels = LinRange(ys[1] - sim_height, ys[end] + sim_height, height)
    end
    pot_values = [
        potential(x, y) for x ∈ xs, y ∈ ypixels
    ]
    heatmap!(scene, pot_values; colormap=:Greens_3)
    heatmap!(scene, image'; colormap=[
            RGBA(0, 0, 0, 0), RGBA(0, 0, 0, 1),
        ],
        colorrange=(0, 15 * length(ys) / sim_pixels)
    )
    save(path, scene)
end

function grid_eval(xs, ys, fun)
    return [
        fun(x, y) for y ∈ ys, x ∈ xs
    ]
end

struct RotatedPotential{DotPotential}
    A::SMatrix{2,2,Float64, 4}
    A_inv::SMatrix{2,2,Float64, 4}
    V::DotPotential
    function RotatedPotential(θ::Real, V)
        rot = rotation_matrix(θ)
        return new{typeof(V)}(rot, inv(rot), V)
    end
end

function (V::RotatedPotential)(x::Real, y::Real)::Float64
    x, y = V.A * SVector(x, y)
    return V.V(x, y)
end

function force(V::RotatedPotential, x::Real, y::Real)::SVector{2,Float64}
    x, y = V.A * SVector(x, y)
    F = force(V.V, x, y)
    return V.A_inv * F
end


struct RepeatedPotential{DotPotential}
    dot_potential::DotPotential
    potential_size::Float64
    locations::Matrix{Float64}
    grid_x::Float64
    grid_y::Float64
    grid_locations::Matrix{Vector{SVector{2,Float64}}}
    grid_w::Int64
    grid_h::Int64

    function RepeatedPotential(locations :: AbstractMatrix, 
        dot_potential, potential_size :: Real)
        min_x, max_x = extrema(locations[1, :])
        min_y, max_y = extrema(locations[2, :])
        # XXX: Explain this math
        grid_w = 3 + ceil(Int, (max_x - min_x) / potential_size)
        grid_h = 3 + ceil(Int, (max_y - min_y) / potential_size)
        grid_locations = [
            Vector{SVector{2,Float64}}()
            for y ∈ 1:grid_h, x ∈ 1:grid_w
        ]
        offsets = [
            0 0
            0 1
            1 0
            0 -1
            -1 0
            -1 -1
            -1 1
            1 1
            1 -1
        ]'
        for (x, y) ∈ eachcol(locations)
            ix = 2 + floor(Int, (x - min_x) / potential_size)
            iy = 2 + floor(Int, (y - min_y) / potential_size)
            for (dx, dy) ∈ eachcol(offsets)
                push!(grid_locations[iy+dy, ix+dx], SVector(x, y))
            end
        end
        return new{typeof(dot_potential)}(dot_potential, potential_size,
            locations, min_x, min_y, grid_locations,
            grid_w, grid_h
        )
    end
end

function (V::RepeatedPotential)(x::Real, y::Real)::Float64
    # find index
    ix = 2 + floor(Int, (x - V.grid_x) / V.potential_size)
    iy = 2 + floor(Int, (y - V.grid_y) / V.potential_size)
    if ix < 1 || iy < 1 || ix > V.grid_w || iy > V.grid_h
        return 0.0
    end
    v = 0
    for (rx, ry) ∈ V.grid_locations[iy, ix]
        v += @inline V.dot_potential(x - rx, y - ry)
    end
    return v
end

function force(V::RepeatedPotential, x::Real, y::Real)::SVector{2,Float64}
    # find index
    ix = 2 + floor(Int, (x - V.grid_x) / V.potential_size)
    iy = 2 + floor(Int, (y - V.grid_y) / V.potential_size)
    if ix < 1 || iy < 1 || ix > V.grid_w || iy > V.grid_h
        return SVector(0.0, 0.0)
    end
    F = SVector(0.0, 0.0)
    for (rx, ry) ∈ V.grid_locations[iy, ix]
        F += @inline force(V.dot_potential, x - rx, y - ry)
    end
    return F
end
