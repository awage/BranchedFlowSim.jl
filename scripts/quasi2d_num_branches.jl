using BranchedFlowSim
using Statistics
using LinearAlgebra
using ArgParse
using Printf

s = ArgParseSettings()
@add_arg_table s begin
    "--num_rays", "-n"
        help = "number of rays"
        arg_type = Int
        default = 4000
    "--time", "-T"
        help = "end time"
        arg_type = Float64
        default = 6.0
    "--visualizations"
        help = "Generate visualization pngs"
        action = :store_true
end

add_potential_args(s)

parsed_args = parse_args(ARGS, s)

# Store results in a separate directory depending on the number of rays
num_rays = parsed_args["num_rays"]
sim_height = 1
sim_width = parsed_args["time"]
v0::Float64 = parsed_args["v0"]

# Generate path based on key parameters
dir = @sprintf "%i_%.1f_%.2f" num_rays sim_width v0
path_prefix = "outputs/quasi2d/$(dir)/"
mkpath(path_prefix)
latest_path = "outputs/quasi2d/latest"
rm(latest_path, force=true)
symlink("$dir/", latest_path)

dt = 0.01
# Time steps to count branches.
ts = LinRange(0, sim_width, round(Int, 50 * sim_width))

y_extra = 1
@time "Created potentials" potentials = get_potentials_from_parsed_args(parsed_args, 0,
    sim_width, -y_extra, sim_height+y_extra)
    
for pot ∈ potentials
    quasi2d_compute_and_save_num_branches(
        path_prefix * "nb_$(pot.path_part).h5", num_rays, dt, ts, pot.instances,
        pot.params)
    println("$(pot.path_part) done")
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
