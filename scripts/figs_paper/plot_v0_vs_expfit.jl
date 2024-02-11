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
dt = 0.01; T = 100; xres = 20
yres = 1024; threshold = 1.5;
num_angles = 200

v0_range = range(0.04, 0.4, step = 0.04)
f_p = zeros(length(v0_range),3)
c_p = zeros(length(v0_range),6,3)
r_p = zeros(length(v0_range),3)

for (k,v0) in enumerate(v0_range)
    # Fermi lattice
    lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2;
    a = lattice_a;
    V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
    s = savename("decay_fermi_polar", @dict(v0))
    data = get_data_decay(V, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
    @unpack rg, mx_arr = data
    p, m, x = get_fit(rg, vec(mean(mx_arr; dims =2)))
    f_p[k,:] = p

    # Cosine sum
    max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; degrees = [1,6];
    for degree ∈ degrees
        cos_pot(θ) = RotatedPotential(θ,
            fermi_dot_lattice_cos_series(degree,
            lattice_a, dot_radius, v0; softness))
        s = savename("decay_cos_polar", @dict(degree, v0))
        data = get_data_decay(cos_pot, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
        @unpack rg, mx_arr = data
        p, m, x = get_fit(rg, vec(mean(mx_arr; dims =2)))
        c_p[k,degree,:] = p
    end

    # Correlated random pot
    correlation_scale = 0.1;
    sim_width = 20; sim_height = 10.;
    Vr(x) = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, round(Int, x*100))
    s = savename("decay_rand_polar", @dict(v0))
    data = get_data_decay(Vr, 1., num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
    @unpack rg, mx_arr = data
    p, m, x = get_fit(rg, vec(mean(mx_arr; dims =2)))
    r_p[k,:] = p
end


fig = Figure(size=(600, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = L"C", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30,  titlesize = 30, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, f_p[:,1], color = :blue, linestyle = :dash, label = L"Fermi")
lines!(ax1, v0_range, r_p[:,1], color = :orange, label = L"Rand")
lines!(ax1, v0_range, c_p[:,1,1], color = :black, linestyle = :dash, label = L"V_{cos} ~  n = 1")
# lines!(ax1, v0_range, c_p[:,2,1], color = :red, label = "Cos n=2 a1")
# lines!(ax1, v0_range, c_p[:,3,1], color = :green, label = "Cos n=3 a1")
# lines!(ax1, v0_range, c_p[:,4,1], color = :pink, label = "Cos n=4 a1")
# lines!(ax1, v0_range, c_p[:,5,1], color = :purple, label = "Cos n=5 a1")
lines!(ax1, v0_range, c_p[:,6,1], color = :cyan, label = L"V_{cos} ~ n = 6")
s = "comparison_fit_coeff_C_polar.png"
axislegend(ax1);
save(plotsdir(s),fig)


fig = Figure(size=(600, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = L"\Omega", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30,  titlesize = 30, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, -f_p[:,3], color = :blue, linestyle=:dash, label = L"Fermi")
lines!(ax1, v0_range, -r_p[:,3], color = :orange, label = L"Rand")
lines!(ax1, v0_range, -c_p[:,1,3], color = :black, linestyle = :dash, label = L"V_{Cos} ~  n = 1")
# lines!(ax1, v0_range, c_p[:,2,3], color = :red, label = "Cos n=2 a2")
# lines!(ax1, v0_range, c_p[:,3,3], color = :green, label = "Cos n=3 a2")
# lines!(ax1, v0_range, c_p[:,4,3], color = :pink, label = "Cos n=4 a2")
# lines!(ax1, v0_range, c_p[:,5,3], color = :purple, label = "Cos n=5 a1")
lines!(ax1, v0_range, -c_p[:,6,3], color = :cyan,  label = L"V_{cos} ~ n=6")

s = "comparison_fit_coeff_omega_polar.png"
axislegend(ax1);
save(plotsdir(s),fig)
