# TODO: Figure out if this file should be in potentials.jl
# Currently this contains mostly potential functions and quasi2d simulation
# for branch counting.
using LinearAlgebra
using StaticArrays

export LatticePotential, PeriodicGridPotential, RotatedPotential
export rotation_matrix, force, force_x, force_y
export quasi2d_num_branches
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
    offset::SVector{2, Float64}
    function LatticePotential(A, radius, v0; offset=[0, 0], softness=0.2)
        return new(A, inv(A), radius, v0, radius * softness, SVector{2, Float64}(offset))
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

function force(V::LatticePotential, x::Real, y::Real)::SVector{2, Float64}
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

function force_y(V::LatticePotential, x, y) :: Float64
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
        norm(force_diff(p, x, y) - force(p, x, y)) for y∈ys, x∈xs
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

function quasi2d_num_branches(xs, ys, ts, potential)
    dx = xs[2] - xs[1]
    dt = dx
    num_rays = length(ys)
    # Compute F(x,y) = -∂V(x,y)/∂y
    ray_y = Vector(ys)
    ray_py = zeros(num_rays)
    num_branches = zeros(length(ts))
    ti = 1
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # for i ∈ 1:num_rays
        #     ray_py[i] += dt * force_y(potential, x, ray_y[i])
        # end
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

function grid_eval(xs, ys, fun)
    return [
        fun(x,y) for y ∈ ys, x ∈ xs
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
    x,y = V.A * SVector(x,y)
    return V.V(x, y)
end
function force(V::RotatedPotential, x::Real, y::Real)::SVector{2, Float64}
    x,y = V.A * SVector(x,y)
    F = force(V.V, x,y)
    return V.A_inv * F
end