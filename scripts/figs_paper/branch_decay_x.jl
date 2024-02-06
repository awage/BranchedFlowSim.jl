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
function compute_area(V, a, dt, T; res = 1000, num_rays = 20000, threshold = 1.5, x0 = 0)
    yg = range(0, 5*a, length = res)
    dy = step(yg)
    xg, area, max_I, rmax = quasi2d_get_stats(num_rays, dt, T, yg, V; b = 4*dy, threshold = threshold, x0 = x0, periodic_bnd = false)
    return xg, yg, area, max_I
end


function _get_area_avg(d)
    @unpack V, res, num_avg, num_rays, a, dt, T, threshold = d # unpack parameters
    xg = range(0,T, step = dt); yg = 0;
    xi = range(0, a, length = num_avg)
    p = Progress(num_avg, "Potential calc")
    nb_arr = zeros(length(xg), num_avg)
    mx_arr = zeros(length(xg), num_avg)
    Threads.@threads for i ∈ eachindex(xi)
        xg, yg, area, max_I = compute_area(V, a, dt, T; res = res,  num_rays = num_rays, threshold = threshold, x0 = xi[i])
        nb_arr[:, i] = area
        mx_arr[:, i] = max_I
        next!(p)
    end
    return @strdict(xg, yg, nb_arr, mx_arr)
end

function compute_area_decay(V, a, num_avg, num_rays, T, threshold, dt, res; prefix = "decay") 
    d = @dict(V, a, num_avg, num_rays, T, threshold, dt, res)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        _get_area_avg, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end 

# Comon parameters
num_rays = 400000; v0 = 0.04
dt = 0.01; T = 40; 
res = 1000; threshold = 1.5; 
num_avg = 50

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; I = rotation_matrix(0)
V = LatticePotential(lattice_a*I, dot_radius, v0; softness=softness)
s = savename("decay_fermi_xavg", @dict(v0))
data_fermi = compute_area_decay(V, lattice_a, num_avg, num_rays, T, threshold, dt, res; prefix = s)  


# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
degrees = [1:6]
data_cos = Vector{typeof(data_fermi)}()
for degree ∈ degrees
    cos_pot = RotatedPotential(0, 
        fermi_dot_lattice_cos_series(degree,  
        lattice_a, dot_radius, v0; softness))
    s = savename("decay_cos_xavg", @dict(degree, v0))
    data = compute_area_decay(cos_pot, lattice_a, num_avg, num_rays, T, threshold, dt, res; prefix = s)
    push!(data_cos, data)
end
#
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

# @unpack xg, nb_arr, mx_arr = data_fermi
# m = vec(mean(nb_arr;dims = 2))
# mI = vec(mean(mx_arr;dims = 2))
# lines!(ax1, xg, mI, color = :green, label = L"Cos Pot deg = 6")

s = "quick_comparison_decay_avg_x.png"
axislegend(ax1);
save(plotsdir(s),fig)


