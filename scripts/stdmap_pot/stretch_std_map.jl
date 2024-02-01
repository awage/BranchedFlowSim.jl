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
    @unpack  a, v0, dt, T, ntraj, pot = d
    df = DeterministicIteratedMap(quasi2d_map!, [0., 0.4], [pot, dt])
    # region = HRectangle([0, 0],[1, 1])
    # sampler, = statespace_sampler(region, 1235)
    # λ = [lyapunov(df, T; u0 = sampler(), Ttr = 0 ) for _ in 1:ntraj]
    yrange = range(0, a, ntraj)
    py = 0.
    points = [ [y, py] for y in yrange] 
    λ = transverse_growth_rates(df, points; Δt = T)
    return @strdict(λ,  d)
end


function get_dat(ntraj, v0, pot, dt, T)
    d = @dict(pot, ntraj, a, v0,  T, dt) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_stretch, # function
        prefix = "stdmap_stretch", # prefix for savename
        force = true, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end

# Compute max lyap exp for a range of parameters
ntraj = 1500;  a = 1; v0 = 3.5; dt = 1; T = 10;
pot = StdMapPotential(a, v0)
dat = get_dat(ntraj, v0, pot, dt, T)
@unpack λ = dat

m = vec(mean(λ, dims = 2))
s = vec(std(λ, dims = 2))

hist(m)

# d = @dict(ntraj, a, T, dt) # parametros
# s = savename("lyap_index_std",d, "png")
# fig = Figure(resolution=(800, 600))
# ax1= Axis(fig[1, 1],  xlabel = L"K", ylabel = "lyap index", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40) 
# lines!(ax1, Krange, ll, color = :blue)
# save(plotsdir(s),fig)

