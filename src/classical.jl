# TODO: Figure out if this file should be in potentials.jl
# Currently this contains mostly potential functions and quasi2d simulation
# for branch counting.
using LinearAlgebra
using StaticArrays
using Makie

export LatticePotential, PeriodicGridPotential, RotatedPotential
export rotation_matrix, force, force_x, force_y
export quasi2d_num_branches, quasi2d_visualize_rays
export grid_eval

"""
LatticePotential


"""
struct LatticePotential
    A::SMatrix{2,2,Float64}
    A_inv::SMatrix{2,2,Float64}
    radius::Float64
    v0::Float64
    α::Float64
    offset::SVector{2,Float64}
    function LatticePotential(A, radius, v0; offset=[0, 0], softness=0.2)
        return new(A, inv(A), radius, v0, radius * softness, SVector{2,Float64}(offset))
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

const four_offsets = [SVector(0, 0), SVector(0, 1), SVector(1, 0), SVector(1, 1)]
function (V::LatticePotential)(x::Real, y::Real)::Float64
    # Find 4 closest lattice vectors
    r = SVector(x, y) - V.offset
    a = V.A_inv * r
    ind = floor.(a)
    v = 0.0
    for offset ∈ four_offsets
        R = V.A * (ind + offset)
        d = norm(R - r)
        v += fermi_step(V.radius - d, V.α)
    end
    return V.v0 * v
end

function force(V::LatticePotential, x::Real, y::Real)::SVector{2,Float64}
    # Find 4 closest lattice vectors
    r = SVector(x, y) - V.offset
    a = V.A_inv * r
    ind = floor.(a)
    F = SVector(0.0, 0.0)
    for offset ∈ four_offsets
        R = V.A * (ind + offset)
        rR = r - R
        d = norm(rR)
        if d <= 1e-9
            return SVector(0.0, 0.0)
        end
        z = exp((-V.radius + d) / V.α)
        F += rR * (z / (V.α * d * (1 + z)^2))
    end
    return F * V.v0
end

function force_x(V::LatticePotential, x, y)
    return force(V, x, y)[1]
end

function force_y(V::LatticePotential, x, y)::Float64
    return force(V, x, y)[2]
    #    # Find 4 closest lattice vectors
    #    r = SVector(x, y) - V.offset
    #    a = V.A_inv * r
    #    ind = floor.(a)
    #    F = 0.0
    #    for offset ∈ four_offsets
    #        R = V.A * (ind + offset)
    #        rR = r - R
    #        d = norm(rR)
    #        if d <= 1e-9
    #            return 0.0
    #        end
    #        z = exp((-V.radius + d) / V.α)
    #        F += rR[2] * (z / (V.α * d * (1 + z)^2))
    #    end
    #    return F * V.v0
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
function force_diff(V, x::Real, y::Real)
    h = 1e-6
    return -[
        (V(x + h, y) - V(x - h, y)) / (2 * h)
        (V(x, y + h) - V(x, y - h)) / (2 * h)
    ]
end

"""
    force(V::Function, x::Real, y::Real)

TBW
"""
function force(V::Function, x::Real, y::Real)
    return force_diff(V, x, y)
end

"""
    compare_force_with_diff(p)

Debugging function for comparing `force` implementation with `force_diff`
"""
function compare_force_with_diff(p)
    xs = LinRange(0, 3, 99)
    ys = LinRange(0, 3, 124)
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

function quasi2d_num_branches(xs, ys, ts, potential; dynamic_rays=false)
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
            if dynamic_rays
                maxd = 10 * orig_dy
                ray_y, ray_py = get_dynamic_rays(ray_y, ray_py, maxd)
            end
        end
    end
    println("num rays: $(length(ray_y))")
    return num_branches
end

function quasi2d_visualize_rays(path, xs, ys, potential;
     dynamic_rays=false, height=length(ys)
     )
    image = zeros(height, length(xs))
    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    ray_y = Vector(ys)
    ray_py = zeros(length(ys))
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        # Collect
        for y ∈ ray_y
            yi = 1 + round(Int, (y / dy)*height / length(ys))
            if yi >= 1 && yi <= height
                image[yi, xi] += 1
            end
        end
        if dynamic_rays && (xi%10 == 0)
            maxd = 10 * dy
            ray_y, ray_py = get_dynamic_rays(ray_y, ray_py, maxd)
        end
    end
    scene = Scene(camera=campixel!, resolution=size(image'))
    ypixels = LinRange(ys[1], ys[end], height)
    pot_values = [
        potential(x, y) for x ∈ xs, y ∈ ypixels
    ]
    heatmap!(scene, pot_values; colormap=:Greens_3)
    heatmap!(scene, image'; colormap=[
            RGBA(0, 0, 0, 0), RGBA(0, 0, 0, 1),
        ],
        colorrange=(0, 20*length(ys)/height)
    )
    save(path, scene)
end

function grid_eval(xs, ys, fun)
    return [
        fun(x, y) for y ∈ ys, x ∈ xs
    ]
end

struct RotatedPotential
    A::SMatrix{2,2,Float64}
    A_inv::SMatrix{2,2,Float64}
    V::Any
    function RotatedPotential(θ::Real, V)
        rot = rotation_matrix(θ)
        return new(rot, inv(rot), V)
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