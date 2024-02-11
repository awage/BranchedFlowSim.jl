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
dt = 0.01; T = 100; xres = 20
yres = 1024; threshold = 1.5;
num_angles = 100

v0 = 0.04
    # Fermi lattice
    lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2;
    a = lattice_a;
    V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
    s = savename("decay_fermi_polar", @dict(v0))
    data_fermi = get_data_decay(V, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)

    # Cosine sum
    max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; degrees = 1:6;
    data_cos = Vector{typeof(data_fermi)}()
    for degree ∈ [1,6]
        cos_pot(θ) = RotatedPotential(θ,
            fermi_dot_lattice_cos_series(degree,
            lattice_a, dot_radius, v0; softness))
        s = savename("decay_cos_polar", @dict(degree, v0))
        data = get_data_decay(cos_pot, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
        push!(data_cos, data)
    end

    # Correlated random pot
    correlation_scale = 0.1; 
    sim_width = T; sim_height = 10.;
    Vr(x) = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, round(Int, x*100))
    s = savename("decay_rand_polar", @dict(v0))
    data_rand = get_data_decay(Vr, correlation_scale, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)


# quick plot:
fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"time", ylabel = "Area meas", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

@unpack rg, nb_arr, mx_arr = data_fermi
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, rg, m, color = :blue, label = "Fermi Lattice")
p, model, xdata = get_fit(rg,m)
lines!(ax1, xdata, model(xdata,p), color = :blue, linestyle = :dash, label = "Fermi Lattice")

@unpack rg, nb_arr, mx_arr = data_rand
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, rg, m, color = :red, label = "Rand")
p, model, xdata = get_fit(rg,m)
lines!(ax1, xdata, model(xdata,p), color = :red, linestyle = :dash, label = "Rand")

@unpack rg, nb_arr, mx_arr = data_cos[1]
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, rg, m, color = :black, label = "Integrable Cos Pot")
p, model, xdata = get_fit(rg,m)
lines!(ax1, xdata, model(xdata,p), linestyle = :dash, color = :black, label = " Fit cos n = 1")

# @unpack rg, nb_arr = data_cos[2]
# m = vec(mean(nb_arr;dims = 2))
# lines!(ax1, rg, m, color = :green, label = L"Cos Pot deg = 6")
# p, model, xdata = get_fit(rg,m)
# lines!(ax1, xdata, model(xdata,p), color = :green, label = L" Fit cos n = 6")

# xlims!(ax1, 0, 20)
s = "quick_comparison_decay_area_polar.png"
axislegend(ax1);
save(plotsdir(s),fig)



# quick plot:
fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"time", ylabel = L"I_{max}", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

@unpack rg, nb_arr, mx_arr = data_fermi
mx = vec(mean(mx_arr;dims = 2))
lines!(ax1, rg, mx, color = :blue, label = "Fermi Lattice")
p, model, xdata = get_fit(rg,mx)
lines!(ax1, xdata, model(xdata,p), color = :blue, linestyle = :dash, label = "Fermi Lattice fit")
@show p

@unpack rg, nb_arr, mx_arr = data_rand
mx = vec(mean(mx_arr;dims = 2))
lines!(ax1, rg, mx, color = :red, label = "Rand")
p, model, xdata = get_fit(rg,mx)
lines!(ax1, xdata, model(xdata,p), color = :red, linestyle = :dash, label = "Rand")
@show p

@unpack rg, nb_arr, mx_arr = data_cos[1]
mx = vec(mean(mx_arr;dims = 2))
lines!(ax1, rg, mx, color = :black, label = "Integrable Cos Pot")
p, model, xdata = get_fit(rg,mx)
lines!(ax1, xdata, model(xdata,p), color = :black, linestyle = :dash, label = " Fit cos n = 1")
@show p
# @unpack rg, nb_arr = data_cos[2]
# m = vec(mean(nb_arr;dims = 2))
# lines!(ax1, rg, m, color = :green, label = L"Cos Pot deg = 6")
# p, model, xdata = get_fit(rg,m)
# lines!(ax1, xdata, model(xdata,p), color = :green, label = L" Fit cos n = 6")
# xlims!(ax1, 0, 20)
s = "quick_comparison_decay_Imax_polar.png"
axislegend(ax1);
save(plotsdir(s),fig)
