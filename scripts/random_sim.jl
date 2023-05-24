using BranchedFlowSim
using CairoMakie
using Makie

# Simulation parameters
L = 20
num_grid_points = 1024
num_steps = 400
T = 1
dt = T / num_steps
# Potential parameters
correlation_scale = 0.5
V0 = 10
# Initial wavefunction parameters
packet_pos = [0, -5]
packet_momentum = [0, 20]
packet_width = 0.05
# Output parameters
eigenfunction_E = 250
path_prefix = "outputs/random"
colorscheme = BranchedFlowSim.RealPartColor(BranchedFlowSim.sigmoid_berlin)

mkpath(dirname(path_prefix))

xgrid = LinRange(-L / 2, L / 2, num_grid_points)
potential = V0 * gaussian_correlated_random(xgrid, xgrid, correlation_scale) +
            absorbing_potential(xgrid, 50, 1)
Ψ = gaussian_packet(xgrid, packet_pos, packet_momentum, packet_width)

Ψs = nothing # Just to free memory
@time "time evolution" Ψs, ts = time_evolution(xgrid, potential, Ψ, T, dt)
print("Start <Ψ|Ψ>=$(total_prob(xgrid, Ψs[:,:,1]))\n")
print("End <Ψ|Ψ>=$(total_prob(xgrid, Ψs[:,:,end]))\n")
@time "video done" save_animation(
    path_prefix * ".mp4", xgrid, potential, Ψs, duration=20, constant_colormap=false)

# Also compute an eigenfunction for some energy.
ΨE = compute_eigenfunctions(xgrid, Ψs, ts, [eigenfunction_E])
save(path_prefix * "_eigenfunc.png",
    wavefunction_to_image(
        xgrid, ΨE[:, :, 1],
        colorscheme=colorscheme
    ))