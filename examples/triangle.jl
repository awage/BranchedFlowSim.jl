using BranchedFlowSim
using FileIO
using ImageIO

# Parameters
L = 20
N = 1000

num_t = 100
T = 1
dt = T / (num_t+1)

dot_height = 40
lattice_constant = 1.5
packet_pos = [0.5, 0.7]
packet_momentum = [0,0]
packet_width = 0.05
path_prefix = "triangle" 

xgrid = LinRange(-L / 2, L / 2, N)
potential = triangle_potential(xgrid, lattice_constant, dot_height) +
            absorbing_potential(xgrid, 100, 2)
Ψ = gaussian_packet(xgrid, packet_pos, packet_momentum, packet_width)
@time "time evolution" Ψs, ts = time_evolution(xgrid, potential, Ψ, T, dt)
@time "making a video" save_animation(path_prefix*".mp4", xgrid, potential, Ψs, ts)
save(path_prefix*".png", wavefunction_to_image(xgrid, Ψs[:, :, length(ts)÷2], potential))
