using DifferentialEquations
using BranchedFlowSim
using ArgParse
using StaticArrays
using LinearAlgebra


s = ArgParseSettings()
@add_arg_table s begin
    "--grid_size"
    help = "width and height of the grid used for the Poincaré section plot"
    arg_type = Int
    default = 1200
    "--min_time"
    help = "end time"
    arg_type = Float64
    default = 500.0
    "--dt"
    help = "time step of integration"
    arg_type = Float64
    default = 0.002
end

add_potential_args(s,
    default_potentials="fermi_lattice",
    defaults=Dict(
        "num_angles" => 1,
        "num_sims" => 1
    )
)

parsed_args = parse_args(ARGS, s)

potentials = get_potentials_from_parsed_args(parsed_args, 1, 1)
pot = potentials[1].instances[1]

Q = parsed_args["grid_size"]
T = parsed_args["min_time"]
dt = parsed_args["dt"]

section = fill(false, Q, Q)
r0 = SA[0.0, 0.19]
p0 = SA[1.0, 0.0]

prob = SecondOrderODEProblem{false}(
    BranchedFlowSim.ray_f, p0, r0, (0,T), pot)
sol = solve(prob, Yoshida6(), dt=dt)
