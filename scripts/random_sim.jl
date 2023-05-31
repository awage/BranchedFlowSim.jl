using BranchedFlowSim
using CairoMakie
using Makie
using LinearAlgebra

# Simulation parameters
L = 20
num_grid_points = 1024
# Potential parameters
correlation_scale = 0.5
V0 = 50
# Initial wavefunction parameters
packet_pos = [0, -5]
packet_momentum = [0, 20]
E_kin = norm(packet_momentum).^2/2
packet_width = 0.05
# Output parameters
eigenfunction_E = [150, 200, 250]
path_prefix = "outputs/random"
colorscheme = BranchedFlowSim.RealPartColor(BranchedFlowSim.sigmoid_berlin)

dt = 1 / (2*E_kin)
T = 2 * L / norm(packet_momentum)
num_steps = round(Int, T/dt)

mkpath(dirname(path_prefix))

xgrid = LinRange(-L / 2, L / 2, num_grid_points)
potential = V0 * gaussian_correlated_random(xgrid, xgrid, correlation_scale) +
            absorbing_potential(xgrid, 50, 1)
Ψ = gaussian_packet(xgrid, packet_pos, packet_momentum, packet_width)

evolution = time_evolution(xgrid, xgrid, potential, Ψ, dt, num_steps)

Ψs = nothing # Just to free memory
@time "video done" make_animation(path_prefix * ".mp4", evolution, potential)

# Also compute an eigenfunction for some energy.

ΨE = collect_eigenfunctions(evolution, eigenfunction_E)
save(path_prefix * "_eigenfunc.png",
    wavefunction_to_image(
        ΨE[:, :, 2],
        colorscheme=colorscheme
    ))