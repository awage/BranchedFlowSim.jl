using BranchedFlowSim
using Statistics
using LinearAlgebra
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--num_rays", "-n"
        help = "number of rays"
        arg_type = Int
        default = 4000
    "--num_angles"
        help = "number of angles used for periodic potentials"
        arg_type = Int
        default = 50
    "--time", "-T"
        help = "end time"
        arg_type = Float64
        default = 6.0
    "--v0"
        help = "potential height. Default is 8% of energy."
        arg_type = Float64
        default = 0.04
    "--potentials"
        help = "comma-separated list of potentials to run"
        arg_type = String
        default = "rand,fermi_lattice,fermi_rand,cos_series,cint"
    "--cos_max_degree"
        help = "maximum degree for the cos_series potentials"
        arg_type = Int
        default = 6
    "--visualizations"
        help = "Generate visualization pngs"
        action = :store_true
end

parsed_args = parse_args(ARGS, s)

potential_types = split(parsed_args["potentials"], ',')

# Store results in a separate directory depending on the number of rays
num_rays = parsed_args["num_rays"]
path_prefix = "outputs/quasi2d/$num_rays/"
mkpath(path_prefix)
latest_path = "outputs/quasi2d/latest"
rm(latest_path, force=true)
symlink("$num_rays/", latest_path)

sim_height = 1
sim_width = parsed_args["time"]
dt = 0.01
correlation_scale = 0.1
v0::Float64 = parsed_args["v0"]
softness = 0.2

num_sims = 100

# Time steps to count branches.
ts = LinRange(0, sim_width, round(Int, 50 * sim_width))

# Periodic potential
lattice_a = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = v0

num_angles = parsed_args["num_angles"]
angles = LinRange(0, π / 2, num_angles + 1)[1:end-1]

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
    return random_fermi_potential(xmin, xmax, ymin, ymax, lattice_a, dot_radius, v0)
end

## Generate potentials

lattice_pots =
    [LatticePotential(lattice_a * rotation_matrix(θ),
        dot_radius, dot_v0; softness=softness)
     for θ ∈ angles]


degrees = 1:parsed_args["cos_max_degree"]
int_pots = [
    rotated_potentials(make_integrable_potential(degree))
    for degree ∈ degrees
]

complex_int_degree = 8
complex_int = complex_separable_potential(complex_int_degree, lattice_a, dot_radius, v0; softness=softness)
cint_pots = rotated_potentials(complex_int)

## Actually evaluate

# TODO: put data in a separate directory for easier copying
# data_prefix = path_prefix*"nb_data/"

angles_vec = Vector(angles)

if "rand" ∈ potential_types
    rand_pots = [random_potential() for _ ∈ 1:num_sims]
    quasi2d_compute_and_save_num_branches(
        path_prefix * "nb_rand.h5", num_rays, dt, ts, rand_pots,
        (type="rand", v0=v0, correlation_scale=correlation_scale))
end
if "fermi_lattice" ∈ potential_types
    quasi2d_compute_and_save_num_branches(
        path_prefix * "nb_lattice.h5", num_rays, dt, ts, lattice_pots,
        (type="fermi_lattice", v0=v0, lattice_a=lattice_a, softness=softness, angles=angles_vec)
    )
end
if "fermi_rand" ∈ potential_types
    rand_dot_pots = [random_dot_potential() for _ ∈ 1:num_sims]
    quasi2d_compute_and_save_num_branches(
        path_prefix * "nb_fermi_rand.h5", num_rays, dt, ts, rand_dot_pots,
        (type="fermi_rand", v0=v0, lattice_a=lattice_a, softness=softness)
    )
end
if "cos_series" ∈ potential_types
    for (di, degree) ∈ enumerate(degrees)
        pots = int_pots[di]
        quasi2d_compute_and_save_num_branches(
            path_prefix * "nb_int_$(degree).h5", num_rays, dt, ts, pots,
            (type="cos_series", v0=v0, lattice_a=lattice_a, degree=degree, angles=angles_vec)
        )
    end
end
if "cint" ∈ potential_types
    quasi2d_compute_and_save_num_branches(
        path_prefix * "nb_cint.h5", num_rays, dt, ts, cint_pots,
        (type="cint", v0=v0, lattice_a=lattice_a, degree=complex_int_degree, angles=angles_vec)
    )
end

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
# include("quasi2d_num_branches_plot.jl")

## Visualize simulations
if parsed_args["visualizations"]
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
    quasi2d_visualize_rays(path_prefix * "cint_rot_sim.png", vis_rays, sim_width,
        complex_int,
        triple_y=true
    )
    for (di, degree) ∈ enumerate(degrees)
        quasi2d_visualize_rays(path_prefix * "int_$(degree)_rot_sim.png",
            vis_rays, sim_width,
            RotatedPotential(θ, int_pots[di][1]),
            triple_y=true
        )
        quasi2d_visualize_rays(path_prefix * "int_$(degree)_sim.png",
            vis_rays, sim_width,
            int_pots[di][1],
            triple_y=true
        )
    end
    rand_potential = random_potential()
    quasi2d_visualize_rays(path_prefix * "rand_sim.png", num_rays, sim_width, rand_potential,
        triple_y=true)

    rand_dots = random_dot_potential()
    quasi2d_visualize_rays(path_prefix * "fermi_rand_sim.png", num_rays, sim_width, rand_dots,
        triple_y=true)
end
