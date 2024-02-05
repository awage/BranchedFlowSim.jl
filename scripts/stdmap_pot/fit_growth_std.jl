using DrWatson 
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using StatsBase
using LsqFit



function compute_init_branches(d; x0 = 0)
    @unpack num_rays, V, T, dt, xmin, xmax = d
    xg = range(x0, T+x0, step = dt)
    num_br = quasi2d_num_branches(num_rays, dt, xg, V; rays_span =(xmin,xmax))
    return @strdict(num_br)
end

# function _get_branches_avg(d)
#     @unpack r,  N, num_rays, a, v0,dt, T = d # unpack parameters
#     nbr_v = []
#     for x0 in range(0,a/2, length = N)
#         @show x0
#         nbr = compute_init_branches(r, a, v0, dt, T;  num_rays = num_rays, θ = 0., x0 = x0)
#         push!(nbr_v, nbr)
#     end
#     xg = range(0, T, step = dt)
#     return @strdict(xg, nbr_v )
# end

function get_datas(V, K; T = 10, num_rays = 100000, dt = 0.01, xmin = 0, xmax = V.a)
    d = @dict(num_rays, K, V, T, dt, xmax, xmin) # parametros
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_init_branches, # function
        prefix = "stdmap_brchs", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    
    return data
end

function plot_fit(d,K,xdata, ydata, model, param)
    fig = Figure(resolution=(800, 600))
    s = savename("Std Map",d)
    ax1= Axis(fig[1, 1], title = s)
    lines!(ax1, xdata, ydata, color = :blue, label = "#br")
    lines!(ax1, xdata, model(xdata, param), color = :red, label = "exp fit")
    axislegend(ax1);
    save(plotsdir(string("exp_fit_std_map_K=", K,".png")),fig)
end


ntraj = 3000000;  a = 1; K = 0.1; dt = 1; T = 50; 
Kv = range(0.1, 4, step = 0.1)
α = zeros(length(Kv))
for (n,K) in enumerate(Kv)
    V = StdMapPotential(a, K)
    d = @dict(ntraj, a, K, T) # parametros
    data = get_datas(V, K; T = T, num_rays = ntraj, dt = 1)
    @unpack num_br = data

    model(x, p) =  p[1] * exp.(p[2] * x)
    ind = 3:12 
    xdata = ind
    ydata = num_br[ind]
    p0 = [0.5, 0.5]
    fit = curve_fit(model, xdata, ydata, p0)
    param = fit.param
    α[n] = param[2]
end


    fig = Figure(resolution=(800, 800))
    s = savename("Std Map",d)
    ax1= Axis(fig[1, 1], title = s, xlabel = "K")
    lines!(ax1, Kv, m, color = :blue, label = "Average transverse Stretching Factor")
    lines!(ax1, Kv, α, color = :red, label = "Exponential growth model fit")
    axislegend(ax1);
    save(plotsdir(string("compare_stretching and_exp_fit_std_map.png")),fig)
