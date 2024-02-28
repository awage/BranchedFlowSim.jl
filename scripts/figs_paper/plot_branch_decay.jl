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
num_rays = 600000; 
dt = 0.01;  xres = 20
yres = 1024; threshold = 3.; 
num_angles = 100

v0 = 0.08
    # Fermi lattice
    lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; T = 30
    a = lattice_a;
    V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
    s = savename("decay_fermi", @dict(v0))
    data_fermi = get_data_decay(V, 1., num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)  
    
    # Cosine sum 
    max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2;  T = 30
    data_cos = Vector{typeof(data_fermi)}()
    for degree ∈ [1,6]
        cos_pot(θ) = RotatedPotential(θ,              
            fermi_dot_lattice_cos_series(degree,  
            lattice_a, dot_radius, v0; softness))
        s = savename("decay_cos", @dict(degree, v0))
        data = get_data_decay(cos_pot, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
        push!(data_cos, data)
    end
 
    # Correlated random pot 
    correlation_scale = 0.1; T = 15.;
    sim_width = 15.; sim_height = 10.;  
    Vr(x) = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, round(Int, x*100))
    s = savename("decay_rand", @dict(v0))
    data_rand = get_data_decay(Vr, 1., num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)


# quick plot:
fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"t", ylabel = L"f_{area}", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

@unpack xg, nb_arr, mx_arr = data_fermi
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, xg, m, color = :blue, label = "Fermi")
p, model, xdata, σ = get_fit(xg,m) 
lines!(ax1, xdata, model(xdata,p), color = :blue, linestyle = :dash)

@unpack xg, nb_arr, mx_arr = data_rand
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, xg, m, color = :red, label = "Rand")
p, model, xdata, σ = get_fit(xg,m) 
lines!(ax1, xdata, model(xdata,p), color = :red, linestyle = :dash)

@unpack xg, nb_arr, mx_arr = data_cos[1]
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, xg, m, color = :black, label = "Integrable Cos n=1")
p, model, xdata, σ = get_fit(xg,m) 
lines!(ax1, xdata, model(xdata,p), linestyle = :dash, color = :black )

# @unpack xg, nb_arr = data_cos[2]
# m = vec(mean(nb_arr;dims = 2))
# lines!(ax1, xg, m, color = :green, label = L"Cos Pot deg = 6")
# p, model, xdata = get_fit(xg,m) 
# lines!(ax1, xdata, model(xdata,p), color = :green, label = L" Fit cos n = 6")

xlims!(ax1, 0, 15)
s = "quick_comparison_decay_area.png"
axislegend(ax1);
save(plotsdir(s),fig)



# quick plot:
# fig = Figure(size=(800, 600))
# ax1= Axis(fig[1, 1], xlabel = L"time", ylabel = L"I_{max}", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

# @unpack xg, nb_arr, mx_arr = data_fermi
# mx = vec(mean(mx_arr;dims = 2))
# lines!(ax1, xg, mx, color = :blue, label = "Fermi Lattice")
# p, model, xdata = get_fit(xg,mx) 
# lines!(ax1, xdata, model(xdata,p), color = :blue, linestyle = :dash, label = "Fermi Lattice fit")
# @show p

# @unpack xg, nb_arr, mx_arr = data_rand
# mx = vec(mean(mx_arr;dims = 2))
# lines!(ax1, xg, mx, color = :red, label = "Rand")
# p, model, xdata = get_fit(xg,mx) 
# lines!(ax1, xdata, model(xdata,p), color = :red, linestyle = :dash, label = "Rand")
# @show p

# @unpack xg, nb_arr, mx_arr = data_cos[1]
# mx = vec(mean(mx_arr;dims = 2))
# lines!(ax1, xg, mx, color = :black, label = "Integrable Cos Pot")
# p, model, xdata = get_fit(xg,mx) 
# lines!(ax1, xdata, model(xdata,p), color = :black, linestyle = :dash, label = " Fit cos n = 1")
# @show p
# # @unpack xg, nb_arr = data_cos[2]
# # m = vec(mean(nb_arr;dims = 2))
# # lines!(ax1, xg, m, color = :green, label = L"Cos Pot deg = 6")
# # p, model, xdata = get_fit(xg,m) 
# # lines!(ax1, xdata, model(xdata,p), color = :green, label = L" Fit cos n = 6")
# xlims!(ax1, 0, 20)
# s = "quick_comparison_decay_Imax.png"
# axislegend(ax1);
# save(plotsdir(s),fig)

