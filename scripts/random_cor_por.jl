using DrWatson 
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Peaks


function _get_area(d)
    @unpack v0, res, num_rays, a, dt, T, threshold = d # unpack parameters
    yg = range(0, 1, length = res)  
    dy = yg[2]-yg[1]
    pot2 = correlated_random_potential(60*a,10*a, a, v0, rand(UInt))
    xg, area, max_I, rmax = quasi2d_get_stats(num_rays, dt, T, yg, pot2; b = 4*dy, threshold = threshold)
    return @strdict(xg, yg, area, rmax, max_I)
end

function _average(d, N = 10)
    data = _get_area(d)
    @unpack area, xg, yg, rmax, max_I= data
    avg_area = deepcopy(area)
    for k = 1:N-1
        data = _get_area(d)
        @unpack area = data
        avg_area = avg_area + area
    end
    area = avg_area/N
    return @strdict(xg, yg, area, rmax, max_I)
end



res = 1000; num_rays = 100000;  a = 0.1; v0 = 0.1; threshold = 2.; 
T = 10.; dt = 0.01
d = @dict(res,num_rays, a, v0, threshold, T, dt) # parametros

data, file = produce_or_load(
    datadir("./storage"), # path
    d, # container for parameter
    _average, # function
    prefix = "random_average_area", # prefix for savename
    force = false, # true for forcing sims
    wsave_kwargs = (;compress = true)
)


function print_fig_(xg, area, a, threshold)

    # Ajustes las curvas. 
    # model(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
    model(x, p) =  p[1] * exp.(-p[2] * x)
    ind = findall(xg .> a*10) 
    xdata = xg[ind]
    ydata = area[ind]
    p0 = [0.5, 0.5]
    fit = curve_fit(model, xdata, ydata, p0)
    param = fit.param
    
    fig = Figure(resolution=(800, 600))
    ax1= Axis(fig[1, 1], title = string("r = ", r, "; f(x) =",   trunc(param[1]; digits = 2),  "exp(-", trunc(param[2]; digits = 2), "x)") , xlabel = "x", ylabel = "Lf_{area}") 
    # ax1= Axis(fig[1, 1], title = string("r = ", r, "; f(x) = ", trunc(param[1]; digits = 2), "+ ", trunc(param[2]; digits = 2),  "exp(-", trunc(param[3]; digits = 2), "x)") , xlabel = "x", ylabel = "Lf_{area}") 
    lines!(ax1, xg, area, color = :blue, label = "area")
    # lines!(ax1, xg, aa, color = :black, label = "smoothed datas")
    lines!(ax1, xdata, model(xdata, param), color = :red, label = "exp fit")
    axislegend(ax1); 
    save(string("../outputs/plot_fit_random_pot=", r, "_thr=", threshold, ".png"),fig)

end


@unpack area,xg,yg, max_I= data
plot(xg, area)
print_fig_(xg,area,a, threshold)
