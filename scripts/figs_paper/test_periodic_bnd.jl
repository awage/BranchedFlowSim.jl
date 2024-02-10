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
num_rays = 5000; 
dt = 0.001; T = 5;

yres = 1024; threshold = 1.5; 
num_angles = 20
v0 = 0.04
# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
V = correlated_random_potential(20, 20, 0.1, v0, 1)
# V = LatticePotential(lattice_a*rotation_matrix(0.), dot_radius, v0; softness=softness)
# V = RotatedPotential(0.,              
    # fermi_dot_lattice_cos_series(1,  
    # lattice_a, dot_radius, v0; softness))
# V = RotatedPotential(rand()*2*pi, CosMixedPotential(0.4, lattice_a, v0))
x0 = -1.; y0 = 0.0
xg = range(1e-9, T, step = dt) 
yg = range(-π, π, length =yres)

# area, max_I = quasi2d_get_stats(num_rays, dt, xg, yg, V; b = 0.003, threshold = threshold, periodic_bnd = false, x0, y0)
# ima, rmax = quasi2d_smoothed_intensity(num_rays, dt, xg, yg, V; b = 0.003,  periodic_bnd = false, x0 , y0)
ima, area=  quasi2d_histogram_intensity(num_rays, xg, yg, V; x0=rand()*4, y0=rand()*4);
heatmap(yg,xg,log.(ima))

# area_v = zeros(round(Int, xres*T), num_angles)
# for n in 1:num_angles
#     x0 = rand()*5; y0 = rand()*5
# xg = range(dt, T, length = round(Int,xres*T))
#     ima, area=  quasi2d_histogram_intensity(num_rays, xg, yg, V; x0, y0)
#     area_v[:,n] = area
# end


# function test()
# x = 0; x0 = 0; y0 = 0;
# ray_y = Vector{Float64}(range(0, 2π, length =  yres))
# ray_py = zeros(yres)
# while x <= 1

#     # kick and drift polar
#     for k in eachindex(ray_y)
#         y = ray_y[k]
#         F = force_x(cos_pot(0.), x*cos(y) + x0, x*sin(y) + y0)*(-x*sin(y))
#             + force_y(cos_pot(0.), x*cos(y) + x0, y*sin(y) + y0)*x*cos(y)
#         ray_py[k] += dt * F
#         # drift
#         ray_y[k] += dt*ray_py[k]
#     end
#     x += dt
#     @show ray_y
# end
# end

# test()
