using BranchedFlowSim
using CairoMakie
using Interpolations
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Makie
using ColorTypes



function pixel_heatmap(path, data; kwargs...)
    scene = Scene(camera=campixel!, resolution=size(data))
    heatmap!(scene, data; kwargs...)
    save(path, scene)
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

function simulate_num_branches(xs, ys, ts, potential)
    dx = xs[2] - xs[1]
    dt = dx
    # Compute F(x,y) = -∂V(x,y)/∂y
    ray_y = Vector(ys)
    ray_py = zeros(num_rays)
    num_branches = zeros(length(ts))
    ti = 1
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* [force(potential, x, y)[2] for y ∈ ray_y]
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

function visualize_sim(path, xs, ys, potential)
    image = zeros(length(ys), length(xs))
    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    ray_y = Vector(ys)
    ray_py = zeros(length(ys))
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* map(f -> f[2], force.(potential, x, ray_y))
        # drift
        ray_y .+= dt .* ray_py
        # Collect
        for y ∈ ray_y
            yi = 1 + round(Int, y / dy)
            if yi >= 1 && yi <= length(ys)
                image[yi, xi] += 1
            end
        end
    end
    scene = Scene(camera=campixel!, resolution=size(image'))
    pot_values = [
        potential(x, y) for x ∈ xs, y ∈ ys
    ]
    heatmap!(scene, pot_values; colormap=:Greens_3)
    heatmap!(scene, image'; colormap=[
            RGBA(0, 0, 0, 0), RGBA(0, 0, 0, 1),
        ],
        colorrange=(0, 20)
    )
    save(path, scene)
end


struct LatticePotential <: Function
    A::Matrix{Float64}
    A_inv::Matrix{Float64}
    radius::Float64
    v0::Float64
    α::Float64
    offset::Vector{Float64}
    function LatticePotential(A, radius, v0; offset=[0, 0], softness=0.2)
        return new(A, inv(A), radius, v0, radius * softness, offset)
    end
end

function rotation_matrix(θ)
    return [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]
end

function (V::LatticePotential)(x::Real, y::Real)
    # Find 4 closest lattice vectors
    r = [x, y] - V.offset
    a = V.A_inv * r
    ind = floor.(a)
    v = 0.0
    for offset ∈ [[0, 0], [0, 1], [1, 0], [1, 1]]
        R = V.A * (ind + offset)
        d = norm(R - r)
        v += fermi_step(V.radius - d, V.α)
    end
    return V.v0 * v
end

function force(V::LatticePotential, x::Real, y::Real)::Vector{Float64}
    # Find 4 closest lattice vectors
    r = [x, y] - V.offset
    a = V.A_inv * r
    ind = floor.(a)
    F = [0.0, 0.0]
    for offset ∈ [[0, 0], [0, 1], [1, 0], [1, 1]]
        R = V.A * (ind + offset)
        rR = r - R
        d = norm(rR)
        if d <= 1e-9
            return [0.0, 0.0]
        end
        z = exp((-V.radius + d) / V.α)
        F += rR * (z / (V.α * d * (1 + z)^2))
    end
    return F * V.v0
end

struct PeriodicGridPotential <: Function
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

function force(V::PeriodicGridPotential, x::Real, y::Real)
    return -gradient(V.itp, x, y)
end

function force_diff(V, x::Real, y::Real)
    h = 1e-6
    return -[
        (V(x + h, y) - V(x - h, y)) / (2 * h)
        (V(x, y + h) - V(x, y - h)) / (2 * h)
    ]
end

"""
    compare_force_with_diff(p)

Debugging function for comparing `force` implementation with `force_diff`
"""
function compare_force_with_diff(p)
    xs = LinRange(0, 3, 99)
    ys = LinRange(0, 3, 124)
    fig, ax, plt = heatmap(xs, ys, (x, y) -> norm(force_diff(p, x, y) - force(p, x, y)))
    Colorbar(fig[1, 2], plt)
    fig
end

path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)
sim_height = 1
sim_width = 2
# num_rays = 768
num_rays = 256
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0 = ϵ_percent * 0.01 * 0.5

num_sims = 50

Ny = num_rays
Nx = round(Int, num_rays * (sim_width / sim_height))
ys = LinRange(0, sim_height, Ny + 1)[1:end-1]
xs = LinRange(0, sim_width, Nx + 1)[1:end-1]

# Time steps to count branches.
ts = LinRange(0, xs[end], 100)

# potential = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
# nbz,m = simulate_num_branches(xs, ys, potential)
# lines(ys[1:end-1], m)

num_branches = zeros(length(ts), num_sims)
Threads.@threads for i ∈ 1:num_sims
    pot_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
    potential = PeriodicGridPotential(xs, ys, pot_arr)
    num_branches[:, i] = simulate_num_branches(xs, ys, ts, potential)
end
nb = vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)

# Periodic potential
lattice_a = 0.1
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5
lines(ts, nb, label="random (mean)", linestyle=:dash,
    axis=(xlabel=L"t", ylabel=L"N_b", title="Number of branches")
)
for deg ∈ [0, 15, 30, 45]
    # lattice_θ = π / 6
    lattice_θ = deg2rad(deg)
    # lattice_θ =0
    potential = LatticePotential(lattice_a * rotation_matrix(lattice_θ),
        dot_radius, dot_v0; offset=[lattice_a / 2, 0])

    grid_nb = simulate_num_branches(xs, ys, ts, potential) / sim_height

    lines!(ts, grid_nb, label="grid $(deg)°")
end

# Compute average over all angles
angles = LinRange(0, π / 2, 51)[1:end-1]
grid_nb = zeros(length(ts), length(angles))
Threads.@threads for di ∈ 1:length(angles)
    θ = angles[di]
    potential = LatticePotential(lattice_a * rotation_matrix(θ),
        dot_radius, dot_v0; offset=[lattice_a / 2, 0])
    grid_nb[:, di] = simulate_num_branches(xs, ys, ts, potential) / sim_height
end
grid_nb_mean = sum(grid_nb, dims=2) / length(angles)
lines!(ts, vec(grid_nb_mean), label="grid (mean)", linestyle=:dot)

axislegend(position=:lt)
display(current_figure())
save(path_prefix * "branches.png", current_figure())
potential = LatticePotential(lattice_a * rotation_matrix(0),
    dot_radius, 0.08 * 0.5, offset=[lattice_a / 2, 0])
pixel_heatmap(path_prefix * "grid_potential.png",
    [potential(x, y) for x ∈ xs, y ∈ ys],
    colormap=:Greens_3
)

rand_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
rand_potential = PeriodicGridPotential(xs, ys, rand_arr)
visualize_sim(path_prefix * "grid_sim.png", xs, ys, potential)
visualize_sim(path_prefix * "rand_sim.png", xs, ys, rand_potential)
