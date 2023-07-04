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
    "--angle_degrees"
    help = "Angle"
    arg_type = Float64
    default = 180 / 5
end

add_potential_args(s,
    defaults=Dict(
        "num_sims" => 1,
    )
)

parsed_args = parse_args(ARGS, s)

num_rays = parsed_args["num_rays"]
sim_height = 1
sim_width = parsed_args["time"]
v0::Float64 = parsed_args["v0"]

# Generate path based on key parameters
path_prefix = "outputs/quasi2d/vis/"
mkpath(path_prefix)

y_extra = 2
@time "Created potentials" potentials = get_potentials_from_parsed_args(
    parsed_args, sim_width, sim_height + y_extra,
)

θ = deg2rad(parsed_args["angle_degrees"])

for pot ∈ potentials
    p = pot.instances[1]

    path = path_prefix * pot.name * ".png"
    quasi2d_visualize_rays(path, num_rays, sim_width,
        p,
        triple_y=true
    )
    println("Wrote $path")
    # Also visualize a rotated potential if the potential is periodic
    if "angles" ∈ keys(pot.params)
        rotpot = RotatedPotential(θ, p)
        path = "$(path_prefix)$(pot.name)_rot_$(@sprintf "%.2f" rad2deg(θ)).png"
        quasi2d_visualize_rays(path, num_rays, sim_width,
            rotpot,
            triple_y=true
        )
        println("Wrote $path")
    end
end