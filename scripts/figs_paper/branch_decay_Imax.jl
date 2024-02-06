using DrWatson 
@quickactivate
using BranchedFlowSim
using CairoMakie
using ProgressMeter
using LaTeXStrings
using LinearAlgebra
using StatsBase
using LsqFit

function _get_max_I(d)
    @unpack V, xres, yres, num_angles, num_rays, a, dt, T, smoothing_b = d # unpack parameters
    xg = range(0,T, length = T*xres); yg = 0;
    yg = sample_midpoints(0, 5*lattice_a, yres)
    dy = step(yg)
    angles = range(0, π/2, length = num_angles + 1)
    angles = angles[1:end-1]
    p = Progress(num_angles, "Potential calc")
    nb_arr = zeros(length(xg), num_angles)
    Threads.@threads for i ∈ 1:num_angles
        potential = V(angles[i])
        int, (rmin, rmax) = quasi2d_intensity(num_rays, dt, xg, yg, potential, b = smoothing_b)
        nb_arr[:, i] = maximum_intensity(int)
        next!(p)
    end
    return @strdict(xg, yg, nb_arr)
end


function compute_decay(V, a, num_angles, num_rays, T, smoothing_b, dt, xres, yres; prefix = "decay") 
    d = @dict(V, a, num_angles, num_rays, T, smoothing_b, xres, yres, dt, xres, yres)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        _get_max_I, # function
        prefix = prefix, # prefix for savename
        force = true, # true for forcing sims
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


function sample_midpoints(a::Real, b::Real, n::Integer)::LinRange{Float64,Int64}
    @assert n > 0
    dx::Float64 = (b - a) / n
    first = a + dx / 2
    last = b - dx / 2
    return LinRange(first, last, n)
end

function maximum_intensity(int)
    return vec(maximum(int, dims=1))
end

# Comon parameters
num_rays = 40000; v0 = 0.04
dt = 0.01; T = 100; 
num_angles = 10; smoothing_b = 0.003 
xres = 10; yres = 1024

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
int_v = []
V_fermi(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness)
data_fermi = compute_decay(V_fermi, lattice_a, num_angles, num_rays, T, smoothing_b, dt, xres, yres; prefix = "decay_fermi") 


# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; degrees  = [1 6]
data_cos = Vector{typeof(data_fermi)}()
for degree ∈ degrees
    s = savename("decay_cos", @dict(degree))
    cos_pot(θ) = RotatedPotential(θ,              
        fermi_dot_lattice_cos_series(degree,  
        lattice_a, dot_radius, v0; softness))
    data = compute_decay(cos_pot, lattice_a, num_angles, num_rays, T, smoothing_b, dt, xres, yres; prefix = s) 
    push!(data_cos, data)
end


# quick plot
fig = Figure(resolution=(1600, 1600))
ax1= Axis(fig[1, 1], xlabel = L"x,t", ylabel = L"I_{max}", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

@unpack xg, nb_arr = data_fermi
m = vec(mean(nb_arr;dims = 2))
# m = vec(maximum(nb_arr;dims = 1))
lines!(ax1, xg, m, color = :blue, label = L"Fermi Lattice")
p, model, xdata = get_fit(xg,m) 
lines!(ax1, xdata, model(xdata,p), color = :blue, label = L"Fermi Lattice")


@unpack xg, nb_arr = data_cos[1]
m = vec(mean(nb_arr;dims = 2))
# m = vec(maximum(nb_arr;dims = 1))
lines!(ax1, xg, m, color = :black, label = L"Integrable Cos Pot")
p, model, xdata = get_fit(xg,m) 
lines!(ax1, xdata, model(xdata,p), color = :black, label = L" Fit cos n = 1")

@unpack xg, nb_arr = data_cos[2]
m = vec(mean(nb_arr;dims = 2))
# m = vec(maximum(nb_arr;dims = 1))
lines!(ax1, xg, m, color = :green, label = L"Cos Pot deg = 6")
p, model, xdata = get_fit(xg,m) 
lines!(ax1, xdata, model(xdata,p), color = :green, label = L" Fit cos n = 6")
s = "quick_comparison_decay_Imax.png"
axislegend(ax1);
save(plotsdir(s),fig)

