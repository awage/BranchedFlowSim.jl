using DrWatson 
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using StatsBase



function compute_init_branches(r, a, v0, dt, T;  num_rays = 20000, θ = 0., x0 = 0.)
    xg = range(x0, T+x0, step = dt)
    if θ != 0. 
        pot2 = RotatedPotential(θ, CosMixedPotential(r,a, v0)) 
    else 
        pot2 = CosMixedPotential(r, a, v0)
    end
    num_br = quasi2d_num_branches(num_rays, dt, xg, pot2; rays_span =(-a/2,a/2))
    return num_br
end

function _get_branches_avg(d)
    @unpack r,  N, num_rays, a, v0,dt, T = d # unpack parameters
    nbr_v = []
    for x0 in range(0,a/2, length = N)
        @show x0
        nbr = compute_init_branches(r, a, v0, dt, T;  num_rays = num_rays, θ = 0., x0 = x0)
        push!(nbr_v, nbr)
    end
    xg = range(0, T, step = dt)
    return @strdict(xg, nbr_v )
end

function get_datas(r; T = 10, num_rays = 100000,  a = 1, v0 = 1., dt = 0.01, N = 20)
    d = @dict(N,num_rays, r, a, v0, T, dt) # parametros
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        _get_branches_avg, # function
        prefix = "periodic_bf_brchs_avg", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    @unpack xg, nbr_v = data
    return xg, mean(nbr_v), d
end

# Print init growth with the Metzger method for three 
# values of r. 
xg, nbr, d = get_datas(0.01)
s = savename("init_branches",d, "png")
fig = Figure(resolution=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"x", ylabel = L"N_{branches}", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30)
lines!(ax1, xg, nbr, color = :blue, label = L"r=0.01")
xg, nbr, d = get_datas(0.3)
lines!(ax1, xg, nbr, color = :red, label = L"r=0.3")
xg, nbr, d = get_datas(0.5)
lines!(ax1, xg, nbr, color = :black, label = L"r=0.5")
axislegend(ax1);
save(plotsdir(s),fig)


