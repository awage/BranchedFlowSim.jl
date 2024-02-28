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

function print_f(x,y, xp, yp,s) 
    fig = Figure(size=(900, 600))
    ax1= Axis(fig[1, 1], xlabel = L"xg", ylabel = L"Imax", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30,  titlesize = 30, yscale = Makie.pseudolog10)
    lines!(ax1, x, y, color = :blue, linestyle=:dash)
    lines!(ax1, xp, yp, color = :orange)
    save(plotsdir(string(s,".png")),fig)
end

# Comon parameters
num_rays = 1500000; 
dt = 0.01; T = 80; xres = 20
yres = 1024; threshold = 2; 
num_angles = 50

v0_range = range(0.04, 0.4, step = 0.04)

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
    f_p[k,:] = p
    
    # Cosine sum 
    max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; degrees = [1,6]; 
    for degree ∈ degrees
        cos_pot(θ) = RotatedPotential(θ,              
            fermi_dot_lattice_cos_series(degree,  
            lattice_a, dot_radius, v0; softness))
        s = savename("decay_cos", @dict(degree, v0))
        data = get_data_decay(cos_pot, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
        @unpack xg, mx_arr = data
        p, m, x = get_fit(xg, vec(mean(mx_arr; dims =2)))
        c_p[k,degree,:] = p
    end
 
    # Correlated random pot 
    correlation_scale = 0.1; 
    sim_width = 20; sim_height = 10.;  
    Vr(x) = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, round(Int, x*100))
    s = savename("decay_rand", @dict(v0))
    data = get_data_decay(Vr, 1., num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
    @unpack xg, mx_arr = data
    p, m, x = get_fit(xg, vec(mean(mx_arr; dims =2)))
    r_p[k,:] = p
end

include("plot_v0_vs_expfit_Imax.jl")
include("plot_v0_vs_expfit_area.jl")
include("plot_v0_vs_expfit_pks.jl")

