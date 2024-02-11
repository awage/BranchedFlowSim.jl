using DrWatson
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using CodecZlib
using StatsBase
using LsqFit
using ProgressMeter

include("utils_decay_comp.jl")

# Comon parameters
num_rays = 40000; v0 = 0.04
dt = 0.01; T = 20; xres = 20
yres = 1024; threshold = 1.5; 
num_angles = 10

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
a = lattice_a;
V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
s = savename("decay_fermi_pol", @dict(v0))
data_fermi = get_data_decay(V, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)  


# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
degrees = [1, 6]
data_cos = Vector{typeof(data_fermi)}()
for degree ∈ degrees
    cos_pot(θ) = RotatedPotential(θ,              
        fermi_dot_lattice_cos_series(degree,  
        lattice_a, dot_radius, v0; softness))
    s = savename("decay_cos", @dict(degree, v0))
    data = get_data_decay(cos_pot, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
    push!(data_cos, data)
end

# Correlated random pot 
correlation_scale = 0.1;
sim_width = T; sim_height = 10. 
Vr(x) = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, round(Int, x*100))
s = savename("decay_rand", @dict(v0))
data_rand = get_data_decay(Vr, correlation_scale, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)


include("plot_branch_decay.jl")
