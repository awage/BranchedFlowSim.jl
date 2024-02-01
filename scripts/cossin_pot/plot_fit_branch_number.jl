using DrWatson
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using CodecZlib
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
    xg, area, max_I, rmax = quasi2d_get_stats(num_rays, dt, T, yg, pot2; b = 4*dy, threshold = threshold, x0 = x0, periodic_bnd = true)
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

function get_fit_p(r, model, p0; T = 20, res = 1000, num_rays = 100000,  a = 1, v0 = 1., threshold = 1.5, N = 30,  dt = 0.01)
    d = @dict(N,res,num_rays, r, a, v0, threshold, T, dt) # parametros
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        _get_area_avg, # function
        prefix = "periodic_bf_area_avg", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )

    @unpack area_v,xg,yg, maxx_I = data
    area = mean(area_v)
    mx, ind = findmax(area)
    xdata = xg[ind:end]
    ydata = area[ind:end]
    fit = curve_fit(model, xdata, ydata, p0)
    return fit.param, xg, area, xdata, ydata
end


function print_fig_lf_vs_x(r; T = 20, res = 1000, num_rays = 100000,  a = 1, v0 = 1., threshold = 1.5, N = 30,  dt = 0.01)
    d = @dict(N,res,num_rays, r, a, v0, threshold, T, dt) # parametros
    model1(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
    p0 = [0.15, 0.2, 0.2]
    param,xg,area,xdata1,ydata = get_fit_p(r, model1, p0; T, res, num_rays, a, v0, threshold, N, dt)
    ydata1 = model1(xdata1, param)
    s1 = string("f(x) = ", trunc(param[1]; digits = 2), "+", trunc(param[2]; digits = 2),  "exp(", trunc(param[3]; digits = 2), "x)")

    model2(x, p) = p[1] .+ p[2] * x.^p[3]
    p0 = [0.15, 0.2, -0.5]
    param,xg,area,xdata2,ydata = get_fit_p(r, model2, p0; T, res, num_rays, a, v0, threshold, N, dt)
    ydata2 = model2(xdata2, param)
    s2 =  string("f(x) = ", trunc(param[1]; digits = 2), "+", trunc(param[2]; digits = 2),  "x^", trunc(param[3]; digits = 2))

    fig = Figure(size=(800, 600))
    ax1= Axis(fig[1, 1], 
    title = string("r = ", r, "; ", s1, ";\n ", s2), xlabel = L"x", ylabel = L"f_{area}", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 30)
    lines!(ax1, xg, area, color = :blue, label = "area")
    lines!(ax1, xdata1, ydata1 , color = :red, label = "exp fit")
    lines!(ax1, xdata2, ydata2, color = :green, label = "algebraic fit")
    axislegend(ax1);
    save(plotsdir(string("plot_fit_periodic_r=", r, ".png")),fig)
end

function print_fig_a0_vs_r(rrange; T = 20, res = 1000, num_rays = 100000,  a = 1, v0 = 1., threshold = 1.5, N = 30,  dt = 0.01)
    ps = []
    model(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
    p0 = [0.15, 0.2, 0.2]
    for r in rrange
        p,_,_,_,_ = get_fit_p(r, model, p0; T, res, num_rays, a, v0, threshold, N, dt)
        push!(ps, p)
    end
    d = @dict(N,res,num_rays, a, v0, threshold, T, dt) # parametros
    s = savename("floor_branches",d, "png")
    fig = Figure(size=(800, 600))
    v = [ a[1] for a in ps]
    ax1= Axis(fig[1, 1],  xlabel = L"r", ylabel = L"a_1", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40)
    lines!(ax1, rrange, v, color = :blue)
    save(plotsdir(s),fig)
end

T = 20; res = 1000; num_rays = 100000;  a = 1; v0 = 1.; threshold = 1.5; N = 30;  dt = 0.01;
rrange = range(0,0.5,length = 50)

print_fig_lf_vs_x(0.; T, res, num_rays, a, v0, threshold, N, dt)
print_fig_lf_vs_x(0.12; T, res, num_rays, a, v0, threshold, N, dt)
print_fig_lf_vs_x(0.25; T, res, num_rays, a, v0, threshold, N, dt)
print_fig_lf_vs_x(0.5; T, res, num_rays, a, v0, threshold, N, dt)
print_fig_a0_vs_r(rrange; T, res, num_rays, a, v0, threshold, N, dt)


