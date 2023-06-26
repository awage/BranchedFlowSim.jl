using BranchedFlowSim
using CairoMakie
using Interpolations
using ProgressMeter
using Statistics
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Makie
using ColorTypes
using JLD2

GC.gc()
# Store results in a separate directory depending on the number of rays
num_rays = 200000
path_prefix = "outputs/quasi2d/$num_rays/"
mkpath(path_prefix)
latest_path = "outputs/quasi2d/latest"
rm(latest_path, force=true)
symlink("$num_rays/", latest_path)

sim_height = 1
sim_width = 6
dt = 0.01
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (E=1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.2

num_sims = 100

# Time steps to count branches.
ts = LinRange(0, sim_width, round(Int, 50 * sim_width))

# Periodic potential
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

num_angles = 50
angles = LinRange(0, π / 2, 51)[1:end-1]

function random_potential()
    return correlated_random_potential(sim_width, 3, correlation_scale, v0)
end

function make_integrable_potential(degree)
    return fermi_dot_lattice_cos_series(degree, lattice_a, dot_radius, v0, softness=softness)
end

function rotated_potentials(base)
    return [RotatedPotential(θ, base) for θ ∈ angles]
end

function random_dot_potential()
    y_extra = 4
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

## Generate potentials

lattice_pots =
    [LatticePotential(lattice_a * rotation_matrix(θ),
        dot_radius, dot_v0; softness=softness)
     for θ ∈ angles]
         
rand_pots = [random_potential() for _ ∈ 1:num_sims]
rand_dot_pots = [random_dot_potential() for _ ∈ 1:num_sims]
    
degrees = [1, 2, 3, 4, 5, 6]
int_pots = [
    rotated_potentials(make_integrable_potential(degree))
    for degree ∈ degrees
]

complex_int_degree = 8
complex_int = begin
    zpot8 = make_integrable_potential(complex_int_degree)
    w = Matrix(zpot8.w)
    w[2:end, 2:end] .= 0
    CosSeriesPotential{SMatrix{9,9,Float64,81}}(w, zpot8.k)
end
cint_pots = rotated_potentials(complex_int)

# Compute average over all angles

function potential_label(degree)
    if degree == 1
        return L"$c_1(\cos(2\pi x/a)+\cos(2\pi y/a))$, $a=0.2$"
    end
    return L"$\sum_{n+m\le%$(degree)}c_{nm}\cos(nkx)\cos(mky)$, $k=2\pi/0.2$"
end


function get_random_nb()::Vector{Float64}
    # Run simulations to count branches
    num_branches = zeros(length(ts), num_sims)
    p = Progress(num_sims, "random")
    Threads.@threads for i ∈ 1:num_sims
        potential = random_potential()
        num_branches[:, i] = quasi2d_num_branches(num_rays, dt, ts, potential)
        next!(p)
    end
    return vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)
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

# TODO: put data in a separate directory for easier copying
# data_prefix = path_prefix*"nb_data/"

angles_vec = Vector(angles)

quasi2d_compute_and_save_num_branches(
    path_prefix*"nb_rand.h5", num_rays, dt, ts, rand_pots, 
    (type="rand", v0=v0, correlation_scale=correlation_scale))
quasi2d_compute_and_save_num_branches(
    path_prefix*"nb_lattice.h5", num_rays, dt, ts, lattice_pots,
    (type="fermi_lattice", v0=v0, lattice_a=lattice_a, softness=softness, angles=angles_vec)
    )
quasi2d_compute_and_save_num_branches(
    path_prefix*"nb_fermi_rand.h5", num_rays, dt, ts, rand_dot_pots,
    (type="fermi_rand", v0=v0, lattice_a=lattice_a, softness=softness)
    )
for (di, degree) ∈ enumerate(degrees)
    pots = int_pots[di]
    quasi2d_compute_and_save_num_branches(
        path_prefix*"nb_int_$(degree).h5", num_rays, dt, ts, pots,
        (type="cos_series", v0=v0, lattice_a=lattice_a, degree=degree, angles=angles_vec)
        )
end
quasi2d_compute_and_save_num_branches(
    path_prefix*"nb_cint.h5", num_rays, dt, ts, cint_pots,
    (type="cint", v0=v0, lattice_a=lattice_a, degree=complex_int_degree, angles=angles_vec)
    )

# @time "rand sim" nb_rand = get_random_nb()
# @time "lattice sim" nb_lattice = get_lattice_mean_nb()
# @time "all int sims" nb_int = [
#     @time "int deg=$(degrees[di])" get_nb_int(int_potentials[di])
#     for di ∈ 1:length(int_potentials)]
# @time "rand_dots sim " nb_rand_dots = get_nb_rand_dots()
# 
# @time "cint sim" nb_cint = get_nb_int(complex_int)

## save results for later plotting

## Run plotting code as well
include("quasi2d_num_branches_plot.jl")

## Visualize simulations
if false
    vis_rays = 2048
    θ = pi / 5
    lattice_mat = lattice_a * rotation_matrix(-θ)
    rot_lattice = LatticePotential(lattice_mat, dot_radius, v0)
    quasi2d_visualize_rays(path_prefix * "lattice_rot_sim.png", vis_rays, sim_width,
        rot_lattice,
        triple_y=true
    )
    quasi2d_visualize_rays(path_prefix * "lattice_sim.png", vis_rays, sim_width,
        LatticePotential(lattice_a * I, dot_radius, v0),
        triple_y=true
    )
    quasi2d_visualize_rays(path_prefix * "cint_rot_sim.png", vis_rays, sim_width,
        RotatedPotential(θ, complex_int),
        triple_y=true
    )
    for (di, degree) ∈ enumerate(degrees)
        quasi2d_visualize_rays(path_prefix * "int_$(degree)_rot_sim.png",
            vis_rays, sim_width,
            RotatedPotential(θ, int_potentials[di]),
            triple_y=true
        )
        quasi2d_visualize_rays(path_prefix * "int_$(degree)_sim.png",
            vis_rays, sim_width,
            int_potentials[di],
            triple_y=true
        )
    end
    rand_potential = random_potential()
    quasi2d_visualize_rays(path_prefix * "rand_sim.png", num_rays, sim_width, rand_potential,
        triple_y=true)
end
