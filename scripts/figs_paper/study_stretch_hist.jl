using DrWatson 
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StatsBase
using CodecZlib

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
    yrange = range(0, a, length = res)
    py = 0.; S = 10
    points = [ [y, py] for y in yrange] 
    λi = [lyapunov(df, Nt*400; u0 = points[k]) for k in 1:res]
    λ = transverse_stretching(df, points; Δt = Nt)
    return @strdict(λ,λi)
end



res = 1000; a = 0.1; v0 = 0.04; dt = 0.01; 
Nt = 500
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; II = rotation_matrix(0.)
V = LatticePotential(lattice_a*II, dot_radius, v0; softness=softness)

d = @dict(res,  a, v0,  Nt, dt, V) # parametros
data = _get_stretch(d)
@unpack λ, λi = data
ml = vec(mean(λ; dims = 2))
# DEFINTION OF KAPLAN 
# of α and β adding the correlation length in the mix
l1 = ml[abs.(λi) .> 1e-4]
ξ = 0.2
T = Nt*dt # We have to mulitply the variance by T 
α = mean(l1)*(v0^(-2/3))*ξ 
β = var(l1)*2*(v0^(-2/3))*ξ*T
gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1)
c_f = 2*gamma.(α,β)./(0.2*v0^(-2/3))

