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
num_rays = 20000; v0 = 0.04
dt = 0.01; T = 100; xres = 10
yres = 1000; threshold = 1.5; 
num_angles = 50

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
a = lattice_a;
V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
s = savename("decay_fermi", @dict(v0))
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


include("plot_branch_decay.jl")
