using BranchedFlowSim
using CairoMakie
using Interpolations
using Statistics
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

path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)

sim_height = 1
sim_width = 6
num_rays = 1024
dt = 1 / 512
# num_rays = 512
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.2

num_sims = 100


# Time steps to count branches.
ts = LinRange(0, sim_width, 200)

# Periodic potential
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

function random_potential()
    return correlated_random_potential(sim_width, sim_height, correlation_scale, v0)
end

function get_random_nb()::Vector{Float64}
    # Run simulations to count branches
    num_branches = zeros(length(ts), num_sims)
    Threads.@threads for i ∈ 1:num_sims
        potential = random_potential()
        num_branches[:, i] = quasi2d_num_branches(num_rays, dt, ts, potential)
    end
    return vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)
end

# Compute average over all angles
function get_lattice_mean_nb()::Vector{Float64}
    angles = LinRange(0, π / 2, 51)[1:end-1]
    grid_nb = zeros(length(ts), length(angles))
    Threads.@threads for di ∈ 1:length(angles)
        θ = angles[di]
        potential = LatticePotential(lattice_a * rotation_matrix(θ),
            dot_radius, dot_v0; offset=[lattice_a / 2, 0], softness=softness)
        grid_nb[:, di] = quasi2d_num_branches(num_rays, dt, ts, potential) / sim_height
    end
    return vec(sum(grid_nb, dims=2) / length(angles))
end

# Integrable potential
function integrable_pot(x::Real, y::Real)::Float64
    (v0 / 4) * (2 + cos(2pi * x / lattice_a) + cos(2pi * y / lattice_a))
end

function integrable_pot2(x::Real, y::Real)::Float64
    a0 = 0.5 * 0.4441648435271566
    a1 = 0.2632722982611676
    a2 = 2 * 0.1491129866125936
    k = 2pi / lattice_a
    return v0 * (a0 + a1 * (cos(k * x) + cos(k * y))
                 + a2 * cos(k * x) * cos(k * y))
end

function get_nb_int(V)::Vector{Float64}
    num_angles = 50
    angles = LinRange(0, π / 2, num_angles + 1)[1:end-1]
    int_nb_arr = zeros(length(ts), length(angles))
    Threads.@threads for di ∈ 1:length(angles)
        θ = angles[di]
        potential = RotatedPotential(θ, V)
        # tys = LinRange(0, 1, 2num_rays)
        int_nb_arr[:, di] = quasi2d_num_branches(num_rays, dt, ts, potential) / sim_height
    end
    return vec(sum(int_nb_arr, dims=2) / length(angles))
end

function random_dot_potential()
    y_extra = sim_width
    xmin = -1
    xmax = sim_width + 1
    ymin = -y_extra
    ymax = sim_height + y_extra
    box_dims =
        num_dots = round(Int, (xmax - xmin) * (ymax - ymin) / lattice_a^2)
    locs = zeros(2, num_dots)
    for i ∈ 1:num_dots
        locs[:, i] = [xmin, ymin] + rand(2) .* [xmax - xmin, ymax - ymin]
    end
    return RepeatedPotential(
        locs,
        FermiDotPotential(dot_radius, v0),
        lattice_a
    )
end

function get_nb_rand_dots()::Vector{Float64}
    # Run simulations to count branches
    num_branches = zeros(length(ts), num_sims)
    Threads.@threads for i ∈ 1:num_sims
        potential = random_dot_potential()
        num_branches[:, i] = quasi2d_num_branches(num_rays, dt, ts, potential)
    end
    return vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)
end

## Actually evaluate

@time "rand sim" nb_rand = get_random_nb()
@time "lattice sim" nb_lattice = get_lattice_mean_nb()
@time "int sim" nb_int = get_nb_int(integrable_pot)
@time "int sim 2" nb_int2 = get_nb_int(integrable_pot2)
@time "rand dot sim" nb_rand_dots = get_nb_rand_dots()

## Make a comparison between rand, lattice, and int

fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
    title=LaTeXString("Number of branches"), limits=((0, sim_width), (0, nothing)))

lines!(ax, ts, nb_rand, label=L"Metzger, $l_c=%$correlation_scale$")
lines!(ax, ts, nb_lattice, label=L"periodic lattice, $a=%$(lattice_a)$ (mean)")
lines!(ax, ts, nb_rand_dots, label=LaTeXString("random dots"))
lines!(ax, ts, nb_int,
    label=L"${v_0}(\cos(2\pi x / a)+\cos(2\pi y / a))$, $a=%$(lattice_a)$ (mean)")
lines!(ax, ts, nb_int2,
    label=L"${v_0}(\cos(2\pi x / a)+\cos(2\pi y / a)-(1/3)(\cos(4\pi x / a)+\cos(4\pi y / a)))$, $a=%$(lattice_a)$ (mean)")
axislegend(ax, position=:lt)
save(path_prefix * "branches.png", fig, px_per_unit=2)
display(fig)

## Visualize simulations
rand_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
rand_potential = PeriodicGridPotential(xs, ys, rand_arr)
quasi2d_visualize_rays(path_prefix * "rand_sim.png", num_rays, sim_width, rand_potential)
quasi2d_visualize_rays(path_prefix * "int_sim.png", num_rays, sim_width, integrable_pot)
potential = LatticePotential(lattice_a * rotation_matrix(0),
    dot_radius, v0, offset=[0, 0])
lattice_mat = lattice_a * rotation_matrix(-pi / 10)
rot_lattice = LatticePotential(lattice_mat, dot_radius, v0)
quasi2d_visualize_rays(path_prefix * "lattice_sim.png", num_rays, sim_width,
    potential
)
quasi2d_visualize_rays(path_prefix * "int_rot_sim.png", num_rays, sim_width,
    RotatedPotential(pi / 10, integrable_pot),
    triple_y=true
)
quasi2d_visualize_rays(path_prefix * "int2_rot_sim.png", num_rays, sim_width,
    RotatedPotential(pi / 10, integrable_pot2),
    triple_y=true
)
quasi2d_visualize_rays(path_prefix * "lattice_rot_sim.png", num_rays, sim_width,
    rot_lattice,
    triple_y=true
)

rand_dot_potential = random_dot_potential()

quasi2d_visualize_rays(path_prefix * "rand_dots_sim.png", num_rays, sim_width,
    rand_dot_potential, triple_y=true)