using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Peaks
using DrWatson 

using ImageFiltering
using LsqFit 



# Display and compute histograms :
function compute_area(r, a, v0, dt, T; res = 1000, num_rays = 20000, θ = 0., threshold = 1.5)
    yg = range(-a/2, a/2, length = res)  
    dy = yg[2] - yg[1]
    if θ != 0. 
        pot2 = RotatedPotential(θ, CosMixedPotential(r,a, v0)) 
    else 
        pot2 = CosMixedPotential(r, a, v0)
    end
    xg, area, max_I, rmax = quasi2d_get_stats(num_rays, dt, T, yg, pot2; b = 4*dy, threshold = threshold)
    return xg, yg, area, max_I
end


function _get_area(d)
    @unpack r, res, num_rays, a, v0,dt, T, threshold = d # unpack parameters
    xg, yg, area, max_I = compute_area(r, a, v0, dt, T; res = res,  num_rays = num_rays, θ = 0., threshold = threshold)
    return @strdict(xg, yg, area, max_I)
end

rrange = 0:0.05:1 
ps = []
for r in rrange
    res = 1000; num_rays = 100000;  a = 1; v0 = 1.; threshold = 1.5; 
    T = 20.; dt = 0.01
    d = @dict(res,num_rays, r, a, v0, threshold, T, dt) # parametros

    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_area, # function
        prefix = "periodic_bf_area_", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )

    @unpack area,xg,yg, max_I = data
    # smoothing over a cell
    l = round(Int16, a/dt)
    aa = imfilter(area, ones(l)./l)
    # Ajustes las curvas. 
    model(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
    # model(x, p) =  p[1] * exp.(-p[2] * x)
    threshold = 1.5
    ind = findall(xg .> a) 
    xdata = xg[ind]
    ydata = aa[ind]
    p0 = [0.5, 0.5, 0.5]
    fit = curve_fit(model, xdata, ydata, p0)
    param = fit.param
    push!(ps, param)  
end

# v = [ u[2] for u in ps]
# plot(rrange, v) 
# Figura Ejemplos r = 0.05 y r = 0.5 : 

function print_fig_(r, T)
    res = 1000; num_rays = 100000;  a = 1; v0 = 1.; threshold = 1.5; 
    dt = 0.01
    d = @dict(res,num_rays, r, a, v0, threshold, T, dt) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_area, # function
        prefix = "periodic_bf_area_", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )

    @unpack area,xg,yg, max_I = data
    l = round(Int16, a/dt)
    aa = imfilter(area, ones(l)./l)
    # Ajustes las curvas. 
    model(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
    # model(x, p) =  p[1] * exp.(-p[2] * x)
    threshold = 1.5
    ind = findall(xg .> a) 
    xdata = xg[ind]
    ydata = aa[ind]
    p0 = [0.5, 0.5, 0.5]
    fit = curve_fit(model, xdata, ydata, p0)
    param = fit.param
    
    fig = Figure(resolution=(800, 600))
    ax1= Axis(fig[1, 1], title = string("r = ", r, "; f(x) = ", trunc(param[1]; digits = 2), " exp(-", trunc(param[2]; digits = 2), "x)") , xlabel = "x", ylabel = "Lf_{area}") 
    lines!(ax1, xg, area, color = :blue, label = "area")
    lines!(ax1, xg, aa, color = :black, label = "smoothed datas")
    lines!(ax1, xdata, model(xdata, param), color = :red, label = "exp fit")
    axislegend(ax1); 
    save(string("../outputs/plot_fit_periodic_r=", r, ".png"),fig)

end


print_fig_(0.0, 20)
print_fig_(0.1, 20)
print_fig_(0.3, 20)
print_fig_(0.5, 20)
print_fig_(0.7, 20)
print_fig_(0.9, 20)
print_fig_(1., 20)
