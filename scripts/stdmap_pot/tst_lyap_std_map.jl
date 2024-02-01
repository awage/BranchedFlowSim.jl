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

function _get_orbit(d) 
    @unpack dt, pot, T, a, ntraj = d
    df = DeterministicIteratedMap(quasi2d_map!, [0.4, 0.2], [pot, dt])
    yrange = range(0, a, ntraj)
    py = 0.
    tr_v = [trajectory(df, T,[y, py])[1] for y in yrange]
    return @strdict(tr_v, yrange, d)
end
 
# function std_map!(du,u,p,t)
#     y,py = u; a,v0 = p
#     # kick
#     du[2] = py + v0/(2π)*sin(y*2π)
#     # drift
#     du[1] = y + du[2]
#     # while du[2] > a; du[2] - a; end
#     # while du[2] < 0; du[2] + a; end
#     # while du[1] > a; du[1] - a; end
#     # while du[1] < 0; du[1] + a; end

#     return nothing
# end


function _get_lyap_1D(d) 
    @unpack  a, v0, dt, T, ntraj, pot = d
    df = DeterministicIteratedMap(quasi2d_map!, [0., 0.4], [pot, dt])
    region = HRectangle([0, 0],[1, 1])
    sampler, = statespace_sampler(region, 1235)
    λ = [lyapunov(df, T; u0 = sampler(), Ttr = 0 ) for _ in 1:ntraj]
    # yrange = range(-a/2, a/2, ntraj)
    # py = 0.
    # λ = [lyapunov(df, T; u0 = [y, py]) for y in yrange]
    return @strdict(λ,  d)
end


function get_lyap_dat(ntraj = 500, v0 = 1., pot, dt = 1, T = 100000)
    d = @dict(pot, ntraj, a, v0,  T, dt) # parametros
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
ntraj = 15000;  a = 1; v0 = 1.; dt = 1; T = 1000; Krange = range(0., 5.3, step = 0.1); threshold = 0.00001
pot = StdMapPotential(a, v0)
ll = Float64[]
for v0 in Krange
    dat = get_lyap_dat(ntraj, v0, pot, dt, T)
    @unpack λ = dat
    ind = findall(abs.(λ) .< threshold)
    # if !isempty(ind);  @show λ[ind]; end
    l_index = length(ind)/length(λ) 
    push!(ll, l_index)
end

_log(x) = x > 0 ? log(x) : -10

d = @dict(ntraj, a, T, dt) # parametros
s = savename("lyap_index_std",d, "png")
fig = Figure(resolution=(800, 600))
ax1= Axis(fig[1, 1],  xlabel = L"K", ylabel = "lyap index", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40) 
lines!(ax1, Krange, ll, color = :blue)
# save(s,fig)

