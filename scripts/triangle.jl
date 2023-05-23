using BranchedFlowSim
using CairoMakie
using Makie

# Simulation parameters
L = 20
num_grid_points = 800
num_steps = 200
T = 1
dt = T / num_steps
# Potential parameters
lattice_constant = 1.5
dot_potential = 50
dot_radius = 0.2 * lattice_constant
# Initial wavefunction parameters
packet_pos = [0.13, 0.21] * lattice_constant
packet_momentum = [0, 0]
packet_width = 0.1
# Output parameters
eigenfunction_E = 100
path_prefix = "outputs/triangle"
colorscheme = BranchedFlowSim.RealPartColor(BranchedFlowSim.sigmoid_berlin)

mkpath(dirname(path_prefix))

xgrid = LinRange(-L / 2, L / 2, num_grid_points)
potential = triangle_potential(xgrid, lattice_constant, dot_potential, dot_radius) +
            absorbing_potential(xgrid, 100, 2)
Ψ = gaussian_packet(xgrid, packet_pos, packet_momentum, packet_width)

Ψs = nothing # Just to free memory
@time "time evolution" Ψs, ts = time_evolution(xgrid, potential, Ψ, T, dt)
print("Start <Ψ|Ψ>=$(total_prob(xgrid, Ψs[:,:,1]))\n")
print("End <Ψ|Ψ>=$(total_prob(xgrid, Ψs[:,:,end]))\n")

# Plot energy spectrum (not yet sure what this means)
fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1])
pE, Es = compute_energy_spectrum(xgrid, Ψs, ts)
lines!(ax, Es, abs.(pE), label=raw"$|\mathcal{P}(E)|$")
# axislegend(ax)
save(path_prefix * "_spectrum.pdf", fig)

@time "video done" save_animation(
    path_prefix * ".mp4", xgrid, potential, Ψs, duration=20, constant_colormap=false)

# Also compute an eigenfunction for some energy.
ΨE = compute_eigenfunctions(xgrid, Ψs, ts, [eigenfunction_E])
save(path_prefix * "_eigenfunc.png",
    wavefunction_to_image(
        xgrid, ΨE[:, :, 1],
        potential,
        max_modulus=0.5 * maximum(abs.(ΨE))))