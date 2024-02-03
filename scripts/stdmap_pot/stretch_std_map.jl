using DrWatson
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using CodecZlib
using ChaosTools
using StatsBase
include(srcdir("utils.jl"))

function quasi2d_map!(du,u,p,t)
    y,py = u; potential, dt = p
    # kick
    du[2] = py + dt * force_y(potential, t, y)
    # drift
    du[1] = y + dt * du[2]
    return nothing
end


function _get_stretch(d) 
    @unpack  a, dt, T, ntraj, pot = d
    df = DeterministicIteratedMap(quasi2d_map!, [0., 0.4], [pot, dt])
    yrange = range(0, a, ntraj)
    py = 0.
    points = [ [y, py] for y in yrange] 
    λ = transverse_growth_rates(df, points; Δt = T)
    return @strdict(λ,  d)
end


function get_dat(ntraj, K, pot, dt, T)
    d = @dict(pot, ntraj, a, K,  T, dt) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_stretch, # function
        prefix = "stdmap_stretch", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end

ntraj = 1500;  a = 1; K = 0.5; dt = 1; T = 5;
Kv = range(0.1,4,step = 0.1)
m = zeros(length(Kv)); s = zeros(length(Kv))  
for (n,K) in enumerate(Kv)
    pot = StdMapPotential(a, K)
    dat = get_dat(ntraj, K, pot, dt, T)
    @unpack λ = dat
    m[n] = mean(vec(mean(λ, dims = 2)))
    s[n] = mean(vec(std(λ, dims = 2)))
end


# hist(m)

# d = @dict(ntraj, a, T, dt) # parametros
# s = savename("lyap_index_std",d, "png")
# fig = Figure(resolution=(800, 600))
# ax1= Axis(fig[1, 1],  xlabel = L"K", ylabel = "lyap index", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40) 
# lines!(ax1, Krange, ll, color = :blue)
# save(plotsdir(s),fig)

