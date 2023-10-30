using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Peaks
using DrWatson 
using LsqFit


# Display and compute histograms :
function compute_histogram(r, a, v0; res = 1000, num_rays = 20000, θ = pi/4)
    xg = range(0, 10*a, length = res) 
    yg = range(-a/2, a/2, length = res)  
    dt = xg[2] - xg[1]; dy = yg[2] - yg[1]
    if θ != 0. 
        pot2 = RotatedPotential(θ, CosMixedPotential(r,a, v0)) 
    else 
        pot2 = CosMixedPotential(r, a, v0)
    end
    I = quasi2d_histogram_intensity(num_rays, xg, yg, pot2)
    Is, rmax = quasi2d_smoothed_intensity(num_rays, dt, xg, yg, pot2; b = 4*dy)
    return xg, yg, I, Is
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

function average_theta(r, a, v0; res = 1000,  num_rays = 1000)
     θr = range(0, 0.01, length = 5)
    xg, yg, I, Is = compute_histogram(r, a, v0; res = res,  num_rays = num_rays, θ = θr[1])
    for θ in θr[2:end]
        xg, yg, It, Its = compute_histogram(r, a, v0; res = res,  num_rays = num_rays, θ = θ)
        I = I + It
        Is = Is + Its
    end
    I = I/length(θr)
    Is = Is/length(θr)
    return xg, yg, I, Is
end


function _get_histograms(d)
    @unpack r, res, num_rays, a, v0 = d # unpack parameters
    xg, yg, I, Is = compute_histogram(r, a, v0; res = res,  num_rays = num_rays, θ = 0.)
    
    # Average over angles
    # xg, yg, I, Is = average_theta(r, a, v0; res = res,  num_rays = num_rays)
    return @strdict(xg, yg, I, Is)
end

function get_stats(I, threshold)
    a = count_area(I, threshold)
    p = count_peaks(I, threshold)
    h = count_heights(I, threshold)
    return a,p,h
end


res = 1000; num_rays = 500000; r = 1.; a = 1; v0 = 1.
rrange = range(0,1,length=20)

p = zeros(length(rrange))

for (k,r) in enumerate(rrange)
    d = @dict(res,num_rays, r, a, v0) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_histograms, # function
        prefix = "periodic_bf", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    @unpack I,Is,xg,yg = data
    dy = yg[2] - yg[1]
    background = (num_rays/res)
    bckgnd_density = background/(dy*num_rays)
    model(x, p) = p[1] * exp.(-p[2] * x)
    threshold = 1.5
    ydata = count_area(Is, threshold*bckgnd_density)
    p0 = [0.5, 0.5]
    ind = findall(xg .> 2) # skip transient 
    fit = curve_fit(model, xg[ind], ydata[ind], p0)
    param = fit.param
    p[k] = param[2]
end

plot(rrange,p)


# Ajustes las curvas. 



