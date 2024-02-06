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
function compute_area(V, a, dt, T; res = 1000, num_rays = 20000, threshold = 1.5, x0 = 0)
    yg = range(-a/2, a/2, length = res)
    dy = step(yg)
    xg, area, max_I, rmax = quasi2d_get_stats(num_rays, dt, T, yg, V; b = 4*dy, threshold = threshold, x0 = x0, periodic_bnd = true)
    return xg, yg, area, max_I
end


function _get_area_avg(d)
    @unpack V, res, num_avg, num_rays, a, dt, T, threshold = d # unpack parameters
    xg = range(0,T, step = dt); yg = 0;
    xi = range(0, lattice_a, length = num_avg)
    p = Progress(num_avg, "Potential calc")
    nb_arr = zeros(length(xg), num_avg)
    Threads.@threads for i ∈ eachindex(xi)
        xg, yg, area, max_I = compute_area(V, a, dt, T; res = res,  num_rays = num_rays, threshold = threshold, x0 = xi[i])
        nb_arr[:, i] = area
        next!(p)
    end
    return @strdict(xg, yg, nb_arr)
end

# function get_fit_p(r; T = 20, res = 1000, num_rays = 100000,  a = 1, v0 = 1., threshold = 1.5, N = 30,  dt = 0.01)
#     d = @dict(N,res,num_rays, r, a, v0, threshold, T, dt) # parametros
#     data, file = produce_or_load(
#         datadir("storage"), # path
#         d, # container for parameter
#         _get_area_avg, # function
#         prefix = "periodic_bf_area_avg", # prefix for savename
#         force = false, # true for forcing sims
#         wsave_kwargs = (;compress = true)
#     )

#     @unpack area_v,xg,yg, maxx_I = data
#     area = mean(area_v)
#     # model(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
#     model(x, p) = p[1] .+ p[2] * x.^p[3]
#     threshold = 1.5
#     mx, ind = findmax(area)
#     xdata = xg[ind:end]
#     ydata = area[ind:end]
#     p0 = [0.15, 0.2, -1]
#     fit = curve_fit(model, xdata, ydata, p0)
#     return fit.param, xg, area, xdata, ydata
# end


# function print_fig_lf_vs_x(r; T = 20, res = 1000, num_rays = 100000,  a = 1, v0 = 1., threshold = 1.5, N = 30,  dt = 0.01)
#     d = @dict(N,res,num_rays, r, a, v0, threshold, T, dt) # parametros
#     # model(x, p) = p[1] .+ p[2] * exp.(-p[3] * x)
#     model(x, p) = p[1] .+ p[2] * x.^p[3]
#     param,xg,area,xdata,ydata = get_fit_p(r; T, res, num_rays, a, v0, threshold, N, dt)

#     fig = Figure(resolution=(800, 600))
#     ax1= Axis(fig[1, 1], title = string("r = ", r, "; f(x) = ", trunc(param[1]; digits = 2), "+", trunc(param[2]; digits = 2),  "x^", trunc(param[3]; digits = 2)) , xlabel = L"x", ylabel = L"f_{area}", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 30)
#     lines!(ax1, xg, area, color = :blue, label = "area")
#     lines!(ax1, xdata, model(xdata, param), color = :red, label = "exp fit")
#     axislegend(ax1);
#     save(plotsdir(string("plot_fit_periodic_r=", r, ".png")),fig)
# end

# T = 20; res = 1000; num_rays = 100000;  a = 1; v0 = 1.; threshold = 1.5; N = 30;  dt = 0.01;
# # rrange = range(0,0.5,length = 50)

# print_fig_lf_vs_x(0.; T, res, num_rays, a, v0, threshold, N, dt)
# print_fig_lf_vs_x(0.12; T, res, num_rays, a, v0, threshold, N, dt)
# print_fig_lf_vs_x(0.25; T, res, num_rays, a, v0, threshold, N, dt)
# print_fig_lf_vs_x(0.5; T, res, num_rays, a, v0, threshold, N, dt)
# # print_fig_a0_vs_r(rrange; T, res, num_rays, a, v0, threshold, N, dt)

function compute_area_decay(V, a, num_avg, num_rays, T, threshold, dt, res; prefix = "decay") 
    d = @dict(V, a, num_avg, num_rays, T, threshold, dt, res)  
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

# Comon parameters
num_rays = 1000; sim_height = 1.
sim_width = 10.; v0 = 0.04
dt = 0.01; T = 40; 
res = 1000; threshold = 1.5; 
num_avg = 10

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; I = rotation_matrix(0)
V = LatticePotential(lattice_a*I, dot_radius, v0; softness=softness)
data_fermi = compute_area_decay(V, lattice_a, num_avg, num_rays, T, threshold, dt, res; prefix = "decay_fermi_xavg")  


# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
degrees = collect(1:max_degree)
data_cos = Vector{typeof(data_fermi)}()
for degree ∈ degrees
    cos_pot = fermi_dot_lattice_cos_series(degree,  
        lattice_a, dot_radius, v0; softness)
    s = savename("decay_cos_xavg", @dict(degree))
    data = compute_area_decay(cos_pot, lattice_a, num_avg, num_rays, T, threshold, dt, res; prefix = s)
    # data = get_datas(cos_pot, num_rays, num_angles, ts, dt, s)
    push!(data_cos, data)
end
#
# quick plot:
fig = Figure(resolution=(1600, 1600))
ax1= Axis(fig[1, 1], xlabel = L"x,t", ylabel = L"Area meas", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

@unpack xg, nb_arr = data_fermi
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, xg, m, color = :blue, label = L"Fermi Lattice")
# @unpack mean_nbr = data_rand
# lines!(ax1, ts, mean_nbr, color = :red, label = L"Rand Correlated")
@unpack xg, nb_arr = data_cos[1]
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, xg, m, color = :black, label = L"Integrable Cos Pot")
@unpack xg, nb_arr = data_cos[6]
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, xg, m, color = :green, label = L"Cos Pot deg = 6")
s = "quick_comparison_decay_avg_x.png"
axislegend(ax1);
save(plotsdir(s),fig)


