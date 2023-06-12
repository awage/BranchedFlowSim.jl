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

function visualize_sim(path, xs, ys, potential)
    image = zeros(length(ys), length(xs))
    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    ray_y = Vector(ys)
    ray_py = zeros(length(ys))
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* map(f -> f[2], force.(Ref(potential), x, ray_y))
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


path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)
sim_height = 1
sim_width = 2
num_rays = 768
# num_rays = 512
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0 = ϵ_percent * 0.01 * 0.5

num_sims = 100

Ny = num_rays
Nx = round(Int, num_rays * (sim_width / sim_height))
ys = LinRange(0, sim_height, Ny + 1)[1:end-1]
xs = LinRange(0, sim_width, Nx + 1)[1:end-1]

# Time steps to count branches.
ts = LinRange(0, xs[end], 100)

# potential = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
# nbz,m = quasi2d_num_branches(xs, ys, potential)
# lines(ys[1:end-1], m)

num_branches = zeros(length(ts), num_sims)
Threads.@threads for i ∈ 1:num_sims
    pot_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
    potential = PeriodicGridPotential(xs, ys, pot_arr)
    num_branches[:, i] = quasi2d_num_branches(xs, ys, ts, potential)
end
nb = vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)

# Periodic potential
lattice_a = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

fig = Figure(resolution=(600, 600))
ax = Axis(fig[1, 1],
    xlabel=L"t", ylabel=L"N_b",
    title=L"Number of branches, $\epsilon=%$ϵ_percent%%$")

lines!(ax, ts, nb, label=L"random $l_c=%$correlation_scale$ (mean) ", linestyle=:dash)
for deg ∈ [0, 15, 30, 45]
    # lattice_θ = π / 6
    lattice_θ = deg2rad(deg)
    # lattice_θ =0
    potential = LatticePotential(lattice_a * rotation_matrix(lattice_θ),
        dot_radius, dot_v0; offset=[lattice_a / 2, 0])

    grid_nb = quasi2d_num_branches(xs, ys, ts, potential) / sim_height

    lines!(ts, grid_nb, label=L"grid ($a=%$(lattice_a)$) %$(deg)°")
end

# Compute average over all angles
angles = LinRange(0, π / 2, 51)[1:end-1]
grid_nb = zeros(length(ts), length(angles))
Threads.@threads for di ∈ 1:length(angles)
    θ = angles[di]
    potential = LatticePotential(lattice_a * rotation_matrix(θ),
        dot_radius, dot_v0; offset=[lattice_a / 2, 0])
    grid_nb[:, di] = quasi2d_num_branches(xs, ys, ts, potential) / sim_height
end
grid_nb_mean = sum(grid_nb, dims=2) / length(angles)
lines!(ts, vec(grid_nb_mean), label=L"grid ($a=%$(lattice_a)$) (mean)", linestyle=:dot)

axislegend(position=:lt)
display(current_figure())
save(path_prefix * "branches.png", current_figure(), px_per_unit=2)
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