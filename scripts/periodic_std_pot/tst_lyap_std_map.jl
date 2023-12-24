using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using DrWatson
using CodecZlib
using ChaosTools
using Interpolations

function std_map!(du,u,p,t)
    y,py = u; a,v0 = p
    # kick
    du[2] = py + v0/(2π)*sin(y*2π)
    # drift
    du[1] = y + du[2]
    # while du[2] > a; du[2] - a; end
    # while du[2] < 0; du[2] + a; end
    # while du[1] > a; du[1] - a; end
    # while du[1] < 0; du[1] + a; end

    return nothing
end


function _get_lyap_1D(d) 
    @unpack  a, v0, dt, T, ntraj = d
    df = DeterministicIteratedMap(std_map!, [0., 0.4], [a, v0])
    region = HRectangle([0, 0],[1, 1])
    sampler, = statespace_sampler(region, 1234)
    λ = [lyapunov(df, T; u0 = sampler(), Ttr = 0 ) for _ in 1:ntraj]
    # yrange = range(-a/2, a/2, ntraj)
    # py = 0.
    # λ = [lyapunov(df, T; u0 = [y, py]) for y in yrange]
    return @strdict(λ,  d)
end


function get_lyap_dat(ntraj = 500,  a = 1, v0 = 1., dt = 0.01, T = 100000)
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
ntraj = 40000;  a = 1; v0 = 1.; dt = 1; T = 10000; Krange = range(0., 11, step = 0.01); threshold = 0.0001
ll = Float64[]
for v0 in Krange
    dat = get_lyap_dat(ntraj, a, v0, dt, T)
    @unpack λ = dat
    ind = findall(λ .< threshold)
    l_index = length(ind)/length(λ) 
    push!(ll, l_index)
end

d = @dict(ntraj, a, T, dt) # parametros
s = savename("lyap_index_std",d, "png")
fig = Figure(resolution=(800, 600))
ax1= Axis(fig[1, 1],  xlabel = L"K", ylabel = "lyap index", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40) 
lines!(ax1, Krange, ll, color = :blue)
save(s,fig)

