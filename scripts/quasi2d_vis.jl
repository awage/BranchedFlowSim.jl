using BranchedFlowSim
using CairoMakie
using Interpolations
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Makie
using ColorTypes

path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)
sim_height = 1
sim_width = 4
num_rays = 512
# num_rays = 512
correlation_scale = 0.01
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.2
dynamic_rays = false
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a

Ny = num_rays
Nx = round(Int, num_rays * (sim_width / sim_height))
ys = LinRange(0, sim_height, Ny + 1)[1:end-1]
xs = LinRange(0, sim_width, Nx + 1)[1:end-1]

# Time steps to count branches.
ts = LinRange(0, xs[end], 200)

integrable_pot(x, y) = v0 * (cos(2pi * x / lattice_a) + cos(2pi * y / lattice_a))

# Make potentials
potential = LatticePotential(lattice_a * rotation_matrix(0),
    dot_radius, v0, offset=[0, 0])
rand_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
rand_potential = PeriodicGridPotential(xs, ys, rand_arr)
lattice_mat = lattice_a * rotation_matrix(-pi/10)
rot_lattice = LatticePotential(lattice_mat, dot_radius, v0)
repeated = RepeatedPotential(
    lattice_points_in_a_box(lattice_mat, -1, sim_width+1, -1, 2),
    FermiDotPotential(dot_radius, v0),
    0.2
)

## Visualize simulations
quasi2d_visualize_rays(path_prefix * "int_sim.png", xs, ys, integrable_pot)
quasi2d_visualize_rays(path_prefix * "grid_sim.png", xs, ys, potential)
quasi2d_visualize_rays(path_prefix * "int_rot_sim.png", xs, LinRange(0, 1, 1024),
    RotatedPotential(pi / 10, integrable_pot),
    triple_y = true
)
quasi2d_visualize_rays(path_prefix * "grid_rot_sim.png", xs, LinRange(0, 1, 1024),
    rot_lattice,
    triple_y = true
)


quasi2d_visualize_rays(path_prefix * "rep_sim.png", xs, LinRange(0, 1, 1024),
    repeated,
    triple_y = true
)

quasi2d_visualize_rays(path_prefix * "rand_sim.png", xs,
    LinRange(0, 1, 1024), rand_potential, triple_y=true)