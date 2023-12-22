using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using DrWatson
using CodecZlib
using ChaosTools
using Interpolations

function quasi2d_map!(du,u,p,t)
    y,py = u; potential, dt = p
    # kick
    du[2] = py + dt * force_y(potential, t, y)
    # drift
    du[1] = y + dt * du[2]
    return nothing
end


function _get_lyap_1D(d) 
    @unpack  a, v0, dt, T, ntraj = d
    pot = StdMapPotential(a, v0)
    df = DeterministicIteratedMap(quasi2d_map!, [0., 0.4], [pot, dt])
    yrange = range(-a/2, a/2, ntraj)
    py = 0.
    λ = [lyapunov(df, T; u0 = [y, py]) for y in yrange]
    return @strdict(λ, yrange, d)
end


function get_lyap_dat(ntraj = 500,  a = 1, v0 = 1., dt = 0.01, T = 10000)
    d = @dict(ntraj, a, v0,  T, dt) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_lyap_1D, # function
        prefix = "periodic_bf_lyap_1D", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end

# Compute max lyap exp for a range of parameters
ntraj = 500;  a = 1; v0 = 1.; dt = 1; T = 10000; Krange = range(0., 5, length = 50); threshold = 0.001
ll = Float64[]
for v0 in Krange
    dat = get_lyap_dat(ntraj, a, v0, dt, T)
    @unpack λ = dat
    ind = findall(λ .> threshold)
    l_index = length(ind)/length(λ) 
    push!(ll, l_index)
end

d = @dict(res, a, T, dt) # parametros
s = savename("lyap_index_std",d, "png")
fig = Figure(resolution=(800, 600))
ax1= Axis(fig[1, 1],  xlabel = L"K", ylabel = "lyap index", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40) 
lines!(ax1, Krange, ll, color = :blue)
save(s,fig)


