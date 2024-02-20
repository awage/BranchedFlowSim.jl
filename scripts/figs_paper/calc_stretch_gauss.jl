using DrWatson 
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StatsBase
# using StaticArrays
# using ChaosTools

include(srcdir("utils.jl"))

function quasi2d_map!(du,u,p,t)
    y,py = u; potential, dt = p
    # kick
    du[2] = py + dt * force_y(potential, dt*t, y)
    # drift
    du[1] = y + dt * du[2]
    return nothing
end

function _get_stretch(d) 
    @unpack V, dt, a, Nt, res = d
    df = DeterministicIteratedMap(quasi2d_map!, [0., 0.4], [V, dt])
    yrange = range(0, a, res)
    py = 0.
    points = [ [y, py] for y in yrange] 
    λ = transverse_growth_rates(df, points; Δt = Nt)
    λ .= λ./dt # normalize by time step
    return @strdict(λ)
end


function get_stretch_index(V, threshold; res = 500, a = 1, v0 = 1., dt = 0.01, Nt = 10000, prefix = "lyap", ξ = 0.1)
    d = @dict(res,  a, v0,  Nt, dt, V, ξ) # parametros
    # data, file = produce_or_load(
    #     datadir("./storage"), # path
    #     d, # container for parameter
    #     _get_stretch, # function
    #     prefix = prefix, # prefix for savename
    #     force = false, # true for forcing sims
    #     wsave_kwargs = (;compress = true)
    # )
    data = _get_stretch(d)
    @unpack λ = data
    ml = vec(mean(λ; dims = 2))
    @show mean(ml)
    @show var(λ)
    # sl = vec(var(λ; dims = 2))
    return mean(ml)*v0^(-2/3)*ξ , var(ml)*Nt*dt*2*v0^(-2/3)*ξ
end

# Comon parameters
num_rays = 200; 
dt = 0.1; Nt = 400
threshold = 0.0

# v0_range = range(0.01, 0.4, step = 0.1)
# # v0_range = [0.01, 0.4]
# l_fermi = zeros(length(v0_range),2)
# l_cos = zeros(length(v0_range),6,2)
# l_rand = zeros(length(v0_range),2)

# for (k,v0) in enumerate(v0_range)
    # Correlated random pot 
    v0 = 0.057
    correlation_scale = 1.;
    sim_width = 10; sim_height = 10.
    Vr = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, 10)
    s = savename("stretch_rand", @dict(v0))
    @show α,β = get_stretch_index(Vr, threshold; res = num_rays, a = 1, v0, dt, Nt, prefix = s, ξ = correlation_scale)
# end
