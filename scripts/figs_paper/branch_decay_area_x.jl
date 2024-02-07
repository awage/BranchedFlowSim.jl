using DrWatson
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using CodecZlib
using StatsBase
using LsqFit
using ProgressMeter


# Display and compute histograms :
function compute_area(V, a, dt, T; xres = 10, yres = 1000, num_rays = 20000, threshold = 1.5, x0 = 0, smoothing_b = 0.003)
    yg = range(0, a, length = yres)
    xg = range(0+x0, T+x0, length = xres*T)
    area, max_I, rmax = quasi2d_get_stats(num_rays, dt, xg, yg, V; b = smoothing_b, threshold = threshold, periodic_bnd = false)
    return xg, yg, area, max_I
end


function _get_area_avg(d)
    @unpack V, xres, yres, num_avg, num_rays, a, dt, T, threshold = d # unpack parameters
    xg = 0; yg = 0;
    xi = range(0, a, length = num_avg)
    p = Progress(num_avg, "Potential calc")
    nb_arr = zeros(T*xres, num_avg)
    mx_arr = zeros(T*xres, num_avg)
    Threads.@threads for i ∈ eachindex(xi)
        xg, yg, area, max_I = compute_area(V, a, dt, T; xres, yres,  num_rays, threshold , x0 = xi[i])
        nb_arr[:, i] = area
        mx_arr[:, i] = max_I
        next!(p)
    end
    return @strdict(xg, yg, nb_arr, mx_arr)
end

function compute_area_decay(V, a, num_avg, num_rays, T, threshold, dt, xres, yres; prefix = "decay") 
    d = @dict(V, a, num_avg, num_rays, T, threshold, dt, xres, yres)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        _get_area_avg, # function
        prefix = prefix, # prefix for savename
        force = true, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end 


# Comon parameters
num_rays = 4000; v0 = 0.04
dt = 0.001; T = 10; 
yres = 1000; threshold = 1.5; 
num_avg = 100; xres = 10;


# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; I = rotation_matrix(0)
V = LatticePotential(lattice_a*I, dot_radius, v0; softness=softness)
s = savename("decay_fermi_xavg", @dict(v0))
data_fermi = compute_area_decay(V, lattice_a, num_avg, num_rays, T, threshold, dt, xres, yres; prefix = s)  


# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
degrees = [1]; degree = 1
data_cos = Vector{typeof(data_fermi)}()
# for degree ∈ degrees
    # cos_pot = RotatedPotential(0, 
    #     fermi_dot_lattice_cos_series(degree,  
    #     lattice_a, dot_radius, v0; softness))
    Vcos = CosMixedPotential(0., lattice_a, v0)
    s = savename("decay_cos_xavg", @dict(degree, v0))
    data = compute_area_decay(Vcos, lattice_a, num_avg, num_rays, T, threshold, dt, xres, yres; prefix = s)
    push!(data_cos, data)
# end


# quick plot:
fig = Figure(resolution=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"x,t", ylabel = L"Area meas", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

@unpack xg, nb_arr, mx_arr = data_fermi
m = vec(mean(nb_arr;dims = 2))
mI = vec(mean(mx_arr;dims = 2))
lines!(ax1, xg, m, color = :blue, label = L"Fermi Lattice")

@unpack xg, nb_arr, mx_arr = data_cos[1]
m = vec(mean(nb_arr;dims = 2))
mI = vec(mean(mx_arr;dims = 2))
lines!(ax1, xg, m, color = :black, label = L"Integrable Cos Pot")

s = "quick_comparison_decay_avg_x.png"
axislegend(ax1);
save(plotsdir(s),fig)


