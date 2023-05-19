using BranchedFlowSim

# Parameters
L = 20
N = 1000

num_t = 100
T = 0.5
dt = T / (num_t+1)

dot_potential = 400
lattice_constant = 1.5
packet_pos = [0.0, 0.0]
packet_momentum = [0,0]
packet_width = 0.05
path_prefix = "triangle" 

xgrid = LinRange(-L / 2, L / 2, N)
potential = triangle_potential(xgrid, lattice_constant, dot_potential) +
            absorbing_potential(xgrid, 100, 2)
Ψ = gaussian_packet(xgrid, packet_pos, packet_momentum, packet_width)
@time "time evolution" Ψs, ts = time_evolution(xgrid, potential, Ψ, T, dt)
@time "making a video" save_animation(path_prefix*".mp4", xgrid, potential, Ψs, ts)