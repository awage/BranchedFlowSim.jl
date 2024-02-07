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
num_rays = 50000; 
dt = 0.01; T = 100; xres = 20
yres = 1000; threshold = 1.5; 
num_angles = 100

v0_range = range(0.01, 0.11, step = 0.01)
f1 = zeros(length(v0_range))
f2 = zeros(length(v0_range))
c1 = zeros(length(v0_range),6)
c2 = zeros(length(v0_range),6)

for (k,v0) in enumerate(v0_range)
    # Fermi lattice
    lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; 
    a = lattice_a;
    V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
    s = savename("decay_fermi", @dict(v0))
    data = get_data_decay(V, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)  
    @unpack xg, mx_arr = data
    p, m, x = get_fit(xg, vec(mean(mx_arr; dims =2)))
    f1[k] = p[1]; f2[k] = p[3] 
    
    # Cosine sum 
    max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; 
    degrees = [1, 6]
    for degree ∈ degrees
        cos_pot(θ) = RotatedPotential(θ,              
            fermi_dot_lattice_cos_series(degree,  
            lattice_a, dot_radius, v0; softness))
        s = savename("decay_cos", @dict(degree, v0))
        data = get_data_decay(cos_pot, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
        @unpack xg, mx_arr = data
        p, m, x = get_fit(xg, vec(mean(mx_arr; dims =2)))
        c1[k,degree] = p[1]
        c2[k,degree] = p[3]
    end
end


fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = "Exp fit", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, f1, color = :blue, label = "Fermi a1")
lines!(ax1, v0_range, c1[:,1], color = :black, label = "Cos n=1 a1")
s = "comparison_fit_coeff_a1.png"
axislegend(ax1);
save(plotsdir(s),fig)


fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = "Exp fit", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, c2[:,1], linestyle = :dash, color = :black, label = " Cos n=1 a2")
lines!(ax1, v0_range, f2, color = :blue, linestyle = :dash, label = "Fermi a2")
s = "comparison_fit_coeff_a2.png"
axislegend(ax1);
save(plotsdir(s),fig)
