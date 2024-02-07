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
function compute_area(V, a, dt, T; yres = 1000, xres = 10, num_rays = 20000, threshold = 1.5, x0 = 0, smoothing_b = 0.003)
    yg = range(0, a, length = yres)
    xg = range(0+x0, T+x0, length = xres*T)
    area, max_I, rmax = quasi2d_get_stats(num_rays, dt, xg, yg, V; b = smoothing_b, threshold = threshold, periodic_bnd = false)
    return xg, yg, area, max_I
end

function _get_area_avg(d)
    @unpack V, xres, yres, num_angles, num_rays, a, dt, T, threshold = d # unpack parameters
    xg = 0; yg = 0;
    angles = range(0, π/2, length = num_angles + 1)
    angles = angles[1:end-1]
    p = Progress(num_angles, "Potential calc")
    nb_arr = zeros(T*xres, num_angles)
    mx_arr = zeros(T*xres, num_angles)
    Threads.@threads for i ∈ 1:num_angles
        potential = V(angles[i])
        xg, yg, area, max_I = compute_area(potential, a, dt, T; xres = xres, yres = yres,  num_rays = num_rays, threshold = threshold)
        nb_arr[:, i] = area
        mx_arr[:, i] = max_I
        next!(p)
    end
    return @strdict(xg, yg, nb_arr, mx_arr)
end


function compute_area_decay(V, a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = "decay") 
    d = @dict(V, a, num_angles, num_rays, T, threshold, dt, xres, yres)  
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

function get_fit(xg, yg)
    model(x, p) = p[1] .+ p[2] * x.^p[3]
    mx, ind = findmax(yg)
    xdata = xg[ind:end]
    ydata = yg[ind:end]
    p0 = [0.15, 0.2, -1]
    fit = curve_fit(model, xdata, ydata, p0)
    return fit.param, model, xdata
end

# Comon parameters
num_rays = 20000; v0 = 0.04
dt = 0.01; T = 100; xres = 10
yres = 1000; threshold = 1.5; 
num_angles = 50

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
a = lattice_a;
V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
s = savename("decay_fermi", @dict(v0))
data_fermi = compute_area_decay(V, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)  


# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
degrees = [1, 6]
data_cos = Vector{typeof(data_fermi)}()
for degree ∈ degrees
    cos_pot(θ) = RotatedPotential(θ,              
        fermi_dot_lattice_cos_series(degree,  
        lattice_a, dot_radius, v0; softness))
    s = savename("decay_cos", @dict(degree, v0))
    data = compute_area_decay(cos_pot, lattice_a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = s)
    push!(data_cos, data)
end


include("plot_branch_decay.jl")
