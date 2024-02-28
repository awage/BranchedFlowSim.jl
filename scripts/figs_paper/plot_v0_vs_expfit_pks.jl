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
# num_rays = 200000; 
# dt = 0.01; 
# threshold = 2.5; 
# xres = 20
# yres = 1024; 
# num_angles = 100
# num_rays = 1200000; threshold = 3.; 
num_rays = 2000000; threshold = 3.; # Funciona bien
# num_rays = 600000; threshold = 3.; # Funciona bien
# num_rays = 1500000; T = 80; threshold = 2; 
# T = 100; threshold = 2.; 
# T = 100; threshold = 1.5; 
dt = 0.01;  xres = 20
yres = 1024; num_angles = 100
v0_range = range(0.04, 0.4, step = 0.04)
f_p = zeros(length(v0_range),3)
c_p = zeros(length(v0_range),6,3)
r_p = zeros(length(v0_range),3)

for (k,v0) in enumerate(v0_range)
    # Fermi lattice
    lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; T = 30
    a = lattice_a;
    V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
    s = savename("decay_fermi", @dict(v0))
    data = get_data_decay(V, 1., num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)  
    @unpack xg, pks_arr = data
    p, m, x = get_fit(xg, vec(mean(pks_arr; dims =2)); Ti = 3)
    f_p[k,:] = p
    print_f(x, m.(x,Ref(p)), xg,  vec(mean(pks_arr; dims =2)), string(s,"pks"))
    
    # Cosine sum 
    max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; degrees = [1,6]; T = 30
    for degree ∈ degrees
        cos_pot(θ) = RotatedPotential(θ,              
            fermi_dot_lattice_cos_series(degree,  
            lattice_a, dot_radius, v0; softness))
        s = savename("decay_cos", @dict(degree, v0))
        data = get_data_decay(cos_pot, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
        @unpack xg, pks_arr = data
        p, m, x = get_fit(xg, vec(mean(pks_arr; dims =2)); Ti = 3)
        c_p[k,degree,:] = p
        # print_f(x, m.(x,Ref(p)), xg,  vec(mean(pks_arr; dims =2)), string(s,"pks"))
    end
 
    # Correlated random pot 
    correlation_scale = 0.1; T = 15.
    sim_width = 15; sim_height = 10.;  
    Vr(x) = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, round(Int, x*100))
    s = savename("decay_rand", @dict(v0))
    data = get_data_decay(Vr, 1., num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
    @unpack xg, pks_arr = data
    p, m, x = get_fit(xg, vec(mean(pks_arr; dims =2)); Ti = 3)
    r_p[k,:] = p
    print_f(x, m.(x,Ref(p)), xg,  vec(mean(pks_arr; dims =2)), string(s,"pks"))
end


fig = Figure(size=(600, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = L"C", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30,  titlesize = 30, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, f_p[:,1], color = :blue, linestyle = :dash, label = L"Fermi")
lines!(ax1, v0_range, 1 .+ r_p[:,1], color = :orange, label = L"Rand")
lines!(ax1, v0_range, c_p[:,1,1], color = :black, linestyle = :dash, label = L"V_{cos} ~  n = 1")
# lines!(ax1, v0_range, c_p[:,2,1], color = :red, label = "Cos n=2 a1")
# lines!(ax1, v0_range, c_p[:,3,1], color = :green, label = "Cos n=3 a1")
# lines!(ax1, v0_range, c_p[:,4,1], color = :pink, label = "Cos n=4 a1")
# lines!(ax1, v0_range, c_p[:,5,1], color = :purple, label = "Cos n=5 a1")
lines!(ax1, v0_range, c_p[:,6,1], color = :cyan, label = L"V_{cos} ~ n = 6")
s = "comparison_fit_coeff_C_pks.png"
axislegend(ax1);
save(plotsdir(s),fig)


fig = Figure(size=(1200, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = L"\Omega", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30,  titlesize = 30, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, -f_p[:,3], color = :blue,  label = L"Fermi")
lines!(ax1, v0_range, -r_p[:,3], color = :orange, label = L"Rand")
lines!(ax1, v0_range, -c_p[:,1,3], color = :black,  label = L"V_{Cos} ~  n = 1")
# lines!(ax1, v0_range, c_p[:,2,3], color = :red, label = "Cos n=2 a2")
# lines!(ax1, v0_range, c_p[:,3,3], color = :green, label = "Cos n=3 a2")
# lines!(ax1, v0_range, c_p[:,4,3], color = :pink, label = "Cos n=4 a2")
# lines!(ax1, v0_range, c_p[:,5,3], color = :purple, label = "Cos n=5 a1")
lines!(ax1, v0_range, -c_p[:,6,3], color = :cyan,  label = L"V_{cos} ~ n=6")

using JLD2
# @load "stretch_factor_Nt=200_num_angles=50.jld2"
@load "stretch_factor_Nt=400_num_angles=50.jld2"
# @load "stretch_factor_Nt=600_num_angles=50.jld2"
# @load "stretch_factor_Nt=800_num_angles=50.jld2"
# @load "stretch_factor_Nt=1000_num_angles=50.jld2"
# @load "stretch_factor_Nt=2000_num_angles=50.jld2"
# gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1)

# c_r = mr.^2 ./ sr ./(0.1*v0_range.^(-2/3))
c_r = get_coeff_pks.(v0_range, :rand); 
lines!(ax1, v0_range, c_r, color = :orange, linestyle = :dash,  label = "Rand, measured with stretching")

# @load "coeff_stretch_factor_fermi.jld2"
c_f = mf.^2 ./ sf ./(0.2*v0_range.^(-2/3))
lines!(ax1, v0_range, c_f, color = :blue, linestyle = :dash,  label = "Fermi, measured with stretching")

# @load "coeff_stretch_factor_cos1.jld2"
c_c1 = mc1.^2 ./ sc1 ./(0.2*v0_range.^(-2/3))
lines!(ax1, v0_range, c_c1, color = :black, linestyle = :dash,  label = "cos n=1, measured with stretching")

# @load "coeff_stretch_factor_cos6.jld2"
c_c6 = mc6.^2 ./ sc6 ./(0.2*v0_range.^(-2/3))
lines!(ax1, v0_range, c_c6, color = :cyan, linestyle = :dash,  label = "cos n=6, measured with stretching")
s = "comparison_fit_coeff_omega_pks.png"
axislegend(ax1);
save(plotsdir(s),fig)
