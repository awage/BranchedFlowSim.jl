using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Peaks
using DrWatson 



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
function compute_histogram(r, a, v0, dt, T; res = 1000, num_rays = 20000, θ = 0.)
    xg = range(0, T, step = dt) 
    yg = range(-a/2, a/2, length = res)  
    dy = yg[2] - yg[1]
    if θ != 0. 
        pot2 = RotatedPotential(θ, CosMixedPotential(r,a, v0)) 
    else 
        pot2 = CosMixedPotential(r, a, v0)
    end
    I = quasi2d_histogram_intensity(num_rays, xg, yg, pot2)
    Is, rmax = quasi2d_smoothed_intensity(num_rays, dt, xg, yg, pot2; b = 4*dy, d = dt*6)
    return xg, yg, I, Is
end

function count_area(hst, threshold) 
    brchs = zeros(size(hst)[2])
    for (k,h) in enumerate(eachcol(hst))
        t = findall(h .> threshold) 
        brchs[k] = length(t)/length(h)  
    end
    return brchs
end

function count_peaks(hst, threshold)
    pks = zeros(size(hst)[2])
    for (k,h) in enumerate(eachcol(hst))
        t = findmaxima(h) 
        ind = findall(t[2] .> threshold)
        pks[k] = length(ind)  
        # pks[k] = length(t[2])  
    end
    return pks
end

function count_heights(hst, threshold)
    pks = zeros(size(hst)[2])
    for (k,h) in enumerate(eachcol(hst))
        t = findmaxima(h) 
        ind = findall(t[2] .> threshold)
        pks[k] = sum(t[2][ind])  
    end
    return pks
end

function average_theta(r, a, v0, dt, T; res = 1000,  num_rays = 1000)
     θr = range(0, 0.01, length = 5)
    xg, yg, I, Is = compute_histogram(r, a, v0, dt, T; res = res,  num_rays = num_rays, θ = θr[1])
    for θ in θr[2:end]
        xg, yg, It, Its = compute_histogram(r, a, v0, dt, T; res = res,  num_rays = num_rays, θ = θ)
        I = I + It
        Is = Is + Its
    end
    I = I/length(θr)
    Is = Is/length(θr)
    return xg, yg, I, Is
end


function _get_histograms(d)
    @unpack r, res, num_rays, a, v0, dt, T = d # unpack parameters
    xg, yg, Ih, Is = compute_histogram(r, a, v0, dt, T; res = res,  num_rays = num_rays, θ = 0.)
    
    # Average over angles
    # xg, yg, I, Is = average_theta(r, a, v0; res = res,  num_rays = num_rays)
    return @strdict(xg, yg, Ih, Is)
end

function get_stats(I, threshold)
    a = count_area(I, threshold)
    p = count_peaks(I, threshold)
    h = count_heights(I, threshold)
    return a,p,h
end


res = 1000; num_rays = 100000; r = 0.5; a = 1; v0 = 1.; dt = 0.01; T = 10
d = @dict(res,num_rays, r, a, v0, dt, T) # parametros

data, file = produce_or_load(
    datadir("./storage"), # path
    d, # container for parameter
    _get_histograms, # function
    prefix = "periodic_bf", # prefix for savename
    force = false, # true for forcing sims
    wsave_kwargs = (;compress = true)
)

@unpack Ih, Is, xg, yg = data
dy = yg[2] - yg[1]
dt = xg[2] - xg[1]
for threshold in 1:0.1:2
    background = (num_rays/(res*length(xg)))
    bckgnd_density = background/(dt*dy*num_rays)
    a,p,h = get_stats(Is', threshold*bckgnd_density)
    fig = Figure(resolution=(800, 600))
    ax1= Axis(fig[1, 1]) 
    ax2= Axis(fig[1, 1]) 
    ax3= Axis(fig[1, 1]) 
    lines!(ax1, xg, a, color = :blue, label = "area")
    lines!(ax2, xg, p, color = :black, label = "# peaks")
    lines!(ax3, xg, h, color = :red, label = "Σ heights")
    axislegend(ax1)
    # display(fig)
    save(string("../outputs/plot_smoothed_hist_periodic", threshold, ".png"),fig)

    background = (num_rays/res)
    bckgnd_density = background/(dy*num_rays)
    a,p,h = get_stats(Ih, threshold*bckgnd_density)
    fig = Figure(resolution=(800, 600))
    ax1= Axis(fig[1, 1]) 
    ax2= Axis(fig[1, 1]) 
    ax3= Axis(fig[1, 1]) 
    lines!(ax1, xg, a, color = :blue, label = "area")
    lines!(ax2, xg, p, color = :black, label = "# peaks")
    lines!(ax3, xg, h, color = :red, label = "Σ heights")
    axislegend(ax1); 
    save(string("../outputs/plot_hist_periodic", threshold, ".png"),fig)

end



# # Ajustes las curvas. 
# using LsqFit 
# model(x, p) = p[1] * exp.(-p[2] * x)
# threshold = 1.5
# ydata = count_heights(Is, threshold*bckgnd_density)
# xdata = xg 
# p0 = [0.5, 0.5]
# fit = curve_fit(model, xdata, ydata, p0)
# param = fit.param


