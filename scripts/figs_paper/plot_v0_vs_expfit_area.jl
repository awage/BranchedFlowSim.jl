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
    fig = Figure(size=(899, 600))
    ax0= Axis(fig[1, 1], xlabel = L"xg", ylabel = L"Area", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30,  titlesize = 30, yscale = Makie.pseudolog10)
    lines!(ax0, x, y, color = :blue, linestyle=:dash)
    lines!(ax0, xp, yp, color = :orange)
    save(plotsdir(string(s,".png")),fig)
end

# Comon parameters

num_rays = 200000; threshold = 4.; 
# num_rays = 2000000; threshold = 3.; # Funciona bien
# num_rays = 4000000; threshold = 3.; # Funciona bien
# num_rays = 1200000; threshold = 3.; 
dt = 0.01; 
# T = 80; threshold = 2; 
# T = 100; threshold = 2.; 
# T = 100; threshold = 1.5; 
xres = 20
yres = 1024; 
num_angles = 50
v0_range = range(0.04, 0.4, step = 0.04)
f_p = zeros(length(v0_range),3)
f_err = zeros(length(v0_range))
c_p = zeros(length(v0_range),6,3)
r_p = zeros(length(v0_range),3)
r_err = zeros(length(v0_range))

for (k,v0) in enumerate(v0_range)
    # Fermi lattice
    lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; T = 30; 
    a = lattice_a;
    V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
    s = savename("decay_fermi", @dict(v0))
    data = get_data_decay(V, 40*a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
    @unpack xg, nb_arr = data
    p, m, x, σ = get_fit(xg, vec(mean(nb_arr; dims =2)))
    f_p[k,:] = p
    f_err[k] = σ[3]
    print_f(x, m.(x,Ref(p)), xg,  vec(mean(nb_arr; dims =2)), string(s,"area"))

    # Cosine sum
    max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
    softness = 0.2; degrees = [1,6]; T = 30
    for degree ∈ degrees
        cos_pot(θ) = RotatedPotential(θ,
            fermi_dot_lattice_cos_series(degree,
            lattice_a, dot_radius, v0; softness))
        s = savename("decay_cos", @dict(degree, v0))
        data = get_data_decay(cos_pot, 40*lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
        @unpack xg, nb_arr = data
        # c = get_coeff_area(v0, :cos1); Tf = round(5/c)
        p, m, x, σ = get_fit(xg, vec(mean(nb_arr; dims =2)))
        c_p[k,degree,:] = p
        print_f(x, m.(x,Ref(p)), xg,  vec(mean(nb_arr; dims =2)), string(s,"area"))
    end

    # Correlated random pot
    correlation_scale = 0.1;  T =15.
    sim_width = 15; sim_height = 10.;
    Vr(x) = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, round(Int, x*100))
    s = savename("decay_rand", @dict(v0))
    data = get_data_decay(Vr, 40*correlation_scale, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
    # data = get_data_decay(Vr, 1., 50, 500000, 15., 3., dt, xres, yres; prefix = s)
    @unpack xg, nb_arr = data
    p, m, x, σ = get_fit(xg, vec(mean(nb_arr; dims =2)))
    r_p[k,:] = p
    c = get_coeff_area(v0, :rand); 
    @show c + p[3] 
    r_err[k] = σ[3]
    print_f(x, m.(x,Ref(p)), xg,  vec(mean(nb_arr; dims =2)), string(s,"area"))
end


fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = L"C", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, r_p[:,1], color = :orange, label = L"Rand")
lines!(ax1, v0_range, f_p[:,1], color = :blue, linestyle = :dash, label = L"Fermi")
lines!(ax1, v0_range, c_p[:,1,1], color = :black, linestyle = :dash, label = L"V_{cos} ~  n = 1")
lines!(ax1, v0_range, c_p[:,6,1], color = :cyan, label = L"V_{cos} ~ n = 6")
s = "comparison_fit_coeff_C_area.png"
xlims!(ax1, 0.04, 0.4)
axislegend(ax1);
save(plotsdir(s),fig)


fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = L"\Omega", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, -f_p[:,3], color = :blue,  label = L"Fermi")
lines!(ax1, v0_range, -r_p[:,3],  color = :orange, label = L"Rand")
lines!(ax1, v0_range, -c_p[:,1,3], color = :black,  label = L"V_{Cos} ~  n = 1")
lines!(ax1, v0_range, -c_p[:,6,3], color = :cyan,  label = L"V_{cos} ~ n=6")

using JLD2
# @load "stretch_factor_Nt=200_num_angles=50.jld2"
# @load "stretch_factor_Nt=400_num_angles=50.jld2"
@load "stretch_factor_Nt=600_num_angles=50.jld2"
# @load "stretch_factor_Nt=600_num_angles=50_num_rays=5000.jld2"
# @load "stretch_factor_Nt=600_num_angles=50_num_rays=10000.jld2"
# @load "stretch_factor_Nt=800_num_angles=50.jld2"
# @load "stretch_factor_Nt=1000_num_angles=50.jld2"
# @load "stretch_factor_Nt=2000_num_angles=50.jld2"
gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1)

c_r = 2*gamma.(mr,sr)./(0.1*v0_range.^(-2/3))
lines!(ax1, v0_range, c_r, color = :orange, linestyle = :dash) #,  label = "Rand, measured with stretching")

c_f = 2*gamma.(mf,sf)./(0.2*v0_range.^(-2/3))
lines!(ax1, v0_range, c_f, color = :blue, linestyle = :dash) #,  label = "Fermi, measured with stretching")

c_c1 = 2*gamma.(mc1,sc1)./(0.2*v0_range.^(-2/3))
lines!(ax1, v0_range, c_c1, color = :black, linestyle = :dash) #,  label = "cos n=1, measured with stretching")

c_c6 = 2*gamma.(mc6,sc6)./(0.2*v0_range.^(-2/3))
lines!(ax1, v0_range, c_c6, color = :cyan, linestyle = :dash) #,  label = "cos n=6, measured with stretching")
s = "comparison_fit_coeff_omega_area.png"
xlims!(ax1, 0.04, 0.4)
axislegend(ax1; position = :lt);
save(plotsdir(s),fig)
