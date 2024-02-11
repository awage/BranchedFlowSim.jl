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
num_rays = 15000; 
dt = 0.01; T = 20;
yres = 1024; threshold = 1.5; 
num_angles = 10
v0 = 0.04
# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
V = RotatedPotential(rand()*2*pi, CosMixedPotential(0.4, lattice_a, v0))
# V = correlated_random_potential(20, 20, 0.1, v0, 2)
# V = LatticePotential(lattice_a*rotation_matrix(0.), dot_radius, v0; softness=softness)
# V = RotatedPotential(0.,              
#     fermi_dot_lattice_cos_series(1,  
#     lattice_a, dot_radius, v0; softness))
x0 = 0.; y0 = 0.0
rg = range(0.01, T, step = dt) 
θg = range(-π, π, length =yres)

# area, max_I = quasi2d_get_stats(num_rays, dt, xg, yg, V; b = 0.003, threshold = threshold, periodic_bnd = false, x0, y0)

# @time ima, area, max_I = quasi2d_smoothed_intensity(num_rays, dt, xg, yg, V; b = 0.003,  x0 , y0, threshold = 1.5)
# heatmap(yg,xg,ima)

# ima, area=  quasi2d_histogram_intensity(num_rays, xg, yg, V; x0=rand()*4, y0=rand()*4, normalized = true);
# heatmap(yg,xg,log.(1e-6 .+ ima))

area_v = zeros(length(rg), num_angles)
max_Iv = zeros(length(rg), num_angles)
for n in 1:num_angles
    x0 = rand()*5; y0 = rand()*5
    # ima, area, max_I = quasi2d_smoothed_intensity(num_rays, dt, rg, θg, V; b = 0.003,  x0 , y0, threshold = 1.5)
    area, max_I = quasi2d_get_stats(num_rays, dt, rg, θg, V; b = 0.003,  x0 , y0, threshold = 1.5)
    # ima, area=  quasi2d_histogram_intensity(num_rays, xg, yg, V; x0, y0, normalized = true)
    area_v[:,n] = area
    max_Iv[:,n] = max_I
end


