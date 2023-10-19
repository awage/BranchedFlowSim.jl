using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Peaks



# Compute average over all angles
function get_branch_nb(r; num_rays = 10000, dt = 1/512, ts = LinRange(0,6,200), sim_height = 1)
        potential = CosMixedPotential(r, 2π)
        nb = quasi2d_num_branches(num_rays, dt, ts, potential; rays_span = (-pi, pi)) / sim_height
    return nb
end

function plt_branch_number(r)
    sim_width = 200
    ts = LinRange(0,sim_width,200)
    @time " sim" nb_branch = get_branch_nb(r; num_rays = 20000,  ts = ts )
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
        title=LaTeXString("Number of branches"), limits=((0, sim_width), (0, nothing)))
    lines!(ax, ts, nb_branch, label=L"Cos mixed r")
    axislegend(ax, position=:lt)
    display(fig)
end

# Display and compute histograms :
function compute_histogram(r,a; res = 1000, num_rays = 20000, θ = 0.)
    xg = range(0, 10*a, length = res) 
    yg = range(-a/2, a/2, length = res)  
    if θ != 0. 
        pot2 = RotatedPotential(θ, CosMixedPotential(r,a)) 
    else 
        pot2 = CosMixedPotential(r, a)
    end
    I = quasi2d_histogram_intensity(num_rays, xg, yg, pot2, (-a/2,a/2))
    I = quasi2d_smoothed_intensity(num_rays, xg, yg, pot2, (-a/2,a/2))
    return xg, yg, I 
end

function count_area(hst, threshold) 
    brchs = zeros(size(hst)[1])
    for (k,h) in enumerate(eachcol(hst))
        t = findall(h .> threshold) 
        brchs[k] = length(t)/length(h)  
    end
    return brchs
end

function count_peaks(hst, threshold)
    pks = zeros(size(hst)[1])
    for (k,h) in enumerate(eachcol(hst))
        t = findmaxima(h) 
        ind = findall(t[2] .> threshold)
        pks[k] = length(ind)  
    end
    return pks
end

function count_heights(hst, threshold)
    pks = zeros(size(hst)[1])
    for (k,h) in enumerate(eachcol(hst))
        t = findmaxima(h) 
        ind = findall(t[2] .> threshold)
        pks[k] = sum(t[2][ind])  
    end
    return pks
end

function average_theta(r, num_rays, res)
     θ = 0.
    xg, yg, I = compute_histogram(r,2; res = res,  num_rays = num_rays, θ = θ)
    for θ in 0:0.001:0.01
        xg, yg, It = compute_histogram(r,2; res = res,  num_rays = num_rays, θ = θ)
        I = I + It
    end
    return xg, yg, I
end


function _get_histograms(d)
    @unpack r,res,num_rays, a = d # unpack parameters
    xg, yg, I = compute_histogram(r, a; res = res,  num_rays = num_rays, θ = 0.)
    xg, yg, Is = compute_histogram(r, a; res = res,  num_rays = num_rays, θ = 0.)
    return @strdict(xg, yg, I)
end

function get_stats(I, threshold)
    b = count_area(I, threshold)
    c = count_peaks(I, threshold)
    d = count_heights(I, threshold)
    return b,c,d
end


res = 1000; num_rays = 400000; r = 0.7; a = 2; 
d = @dict(res,num_rays, r, a) # parametros

data, file = produce_or_load(
    datadir("./storage"), # path
    d, # container for parameter
    _get_histograms, # function
    prefix = "periodic_bf", # prefix for savename
    force = false, # true for forcing sims
    wsave_kwargs = (;compress = true)
)

@unpack I,xg,yg = data
b,c,d = get_stats(I, 2.)
fig = Figure(resolution=(800, 600))
ax1= Axis(fig[1, 1]) 
ax2= Axis(fig[1, 1]) 
ax3= Axis(fig[1, 1]) 
lines!(ax1, xg, b, color = :blue)
lines!(ax2, xg, c, color = :black)
lines!(ax3, xg, d, color = :red)
display(fig)

