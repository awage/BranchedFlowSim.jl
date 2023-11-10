using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Peaks
using DrWatson 
using ImageFiltering
using StatsBase
using LsqFit 



# Display and compute histograms :
function compute_area(r, a, v0, dt, T; res = 1000, num_rays = 20000, θ = 0., threshold = 1.5, x0 = 0)
    yg = range(-a/2, a/2, length = res)  
    dy = yg[2] - yg[1]
    if θ != 0. 
        pot2 = RotatedPotential(θ, CosMixedPotential(r,a, v0)) 
    else 
        pot2 = CosMixedPotential(r, a, v0)
    end
    xg, area, max_I, rmax = quasi2d_get_stats(num_rays, dt, T, yg, pot2; b = 4*dy, threshold = threshold, x0 = x0)
    return xg, yg, area, max_I
end


function _get_area_avg(d)
    @unpack r, res, N, num_rays, a, v0,dt, T, threshold = d # unpack parameters
    maxx_I = []
    area_v = []
    xg = 0; yg = 0;
    for x0 in range(0,a/2, length = N) 
        @show x0
        xg, yg, area, max_I = compute_area(r, a, v0, dt, T; res = res,  num_rays = num_rays, θ = 0., threshold = threshold, x0 = x0)
        push!(maxx_I, max_I)
        push!(area_v, area)
    end
    return @strdict(xg, yg, area_v, maxx_I)
end

function get_fit_p(r; T = 20, res = 1000, num_rays = 100000,  a = 1, v0 = 1., threshold = 1.5, N = 30,  dt = 0.01)
    d = @dict(N,res,num_rays, r, a, v0, threshold, T, dt) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_area_avg, # function
        prefix = "periodic_bf_area_avg", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )

    @unpack area_v,xg,yg, maxx_I = data
    # smoothing over a cell
    area = mean(area_v)
    # Ajustes las curvas. 
    model(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
    threshold = 1.5
    mx, ind = findmax(area)
    # ind, vals= findmaxima(area)
    # xdata = xg[ind]
    # ydata = area[ind]
    # ind = findall(xg .> a) 
    xdata = xg[ind:end]
    ydata = area[ind:end]
    p0 = [0.15, 0.2, 0.2]
    fit = curve_fit(model, xdata, ydata, p0)
    return fit.param
end


function print_fig_(r; T = 20, res = 1000, num_rays = 100000,  a = 1, v0 = 1., threshold = 1.5, N = 30,  dt = 0.01)
    d = @dict(N,res,num_rays, r, a, v0, threshold, T, dt) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_area_avg, # function
        prefix = "periodic_bf_area_avg", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )

    @unpack area_v,xg,yg, maxx_I = data
    area = mean(area_v)
    l = round(Int16, a/dt)
    aa = imfilter(area, ones(l)./l)
    # Ajustes las curvas. 
    model(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
    threshold = 1.5
    mx, ind = findmax(area)
    xdata = xg[ind:end]
    ydata = area[ind:end]
    p0 = [0.1, 0.2, 0.2]
    fit = curve_fit(model, xdata, ydata, p0)
    param = fit.param
    
    fig = Figure(resolution=(800, 600))
    ax1= Axis(fig[1, 1], title = string("r = ", r, "; f(x) = ", trunc(param[1]; digits = 2), " exp(-", trunc(param[2]; digits = 2), "x)") , xlabel = "x", ylabel = "Lf_{area}") 
    lines!(ax1, xg, area, color = :blue, label = "area")
    # lines!(ax1, xg, aa, color = :black, label = "smoothed datas")
    lines!(ax1, xdata, model(xdata, param), color = :red, label = "exp fit")
    axislegend(ax1); 
    save(string("../outputs/plot_fit_periodic_r=", r, ".png"),fig)

end

T = 20; res = 1000; num_rays = 100000;  a = 1; v0 = 1.;hreshold = 1.5; N = 30;  dt = 0.01;
rrange = range(0,1,length = 50)
ps = []
for r in rrange
    print_fig_(r; T, res, num_rays, a, v0, threshold, N, dt)
    p = get_fit_p(r; T, res, num_rays, a, v0, threshold, N, dt) 
    push!(ps, p)
end


d = @dict(N,res,num_rays, r, a, v0, threshold, T, dt) # parametros
s = savename("floor_branches",d, ".png")
fig = Figure(resolution=(800, 600))
v = [ a[1] for a in ps]
ax1= Axis(fig[1, 1], title = "Superwires volume as a function of r", xlabel = "r", ylabel = "lower threshold") 
lines!(ax1, rrange, v, color = :blue)
save(string("../outputs/",s),fig)
