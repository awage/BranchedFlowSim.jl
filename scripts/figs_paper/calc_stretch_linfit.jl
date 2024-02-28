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
    yrange = range(0, a, res)
    py = 0.; S = 10
    points = [ [y, py] for y in yrange] 
    λ = transverse_stretching(df, points; Δt = Nt, S)
    return @strdict(λ)
end


function get_stretch_dat(V; res = 500, a = 1, v0 = 1., dt = 0.01, Nt = 10000, prefix = "lyap", ξ)
    d = @dict(res,  a, v0,  Nt, dt, V) # parametros
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        _get_stretch, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    @unpack λ = data
    ml = vec(mean(λ; dims = 2))
    @show mean(ml[ml .> 0])
    @show mean(ml)
    # @show std(ml)*2
    # sl = vec(std(λ; dims = 2))
    # DEFINTION OF KAPLAN 
    # of α and β adding the correlation length in the mix
    # T = Nt*dt # We have to mulitply the variance by T 
    return mean(ml)*(v0^(-2/3))*ξ , var(ml)*(v0^(-2/3))*ξ
end



function compute_stretch_rand(v0, num_rays, dt, Nt; sd = 100)
l_rand = zeros(length(Nt), 2)
γ = zeros(length(v0_range))
c = zeros(length(v0_range))
    # for n in 1:num_angles
    for (k,t) in enumerate(Nt)
        # Correlated random pot 
        correlation_scale = 0.1;
        sim_width = t*dt ; sim_height = 10.
        Vr = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, sd)
        s = savename("stretch_rand", @dict(sd))
        α,β  = get_stretch_index(Vr; res = num_rays, a = 1, v0, dt, Nt = t, prefix = s, ξ = correlation_scale)
        l_rand[k,:] = [α, β]
    end
    # end
    # m = mean(l_rand,dims =2)
    model(x, p) = p[1] .+ p[2] * x
    lb = [0., 0.]
    ub = [10., 20.]   
    p0 = [1.,1.]
    f = curve_fit(model, Nt.*dt, l_rand[:,1],  p0; lower = lb, upper = ub)
    α = f.param[2]
    f = curve_fit(model, (Nt.*dt).^2, l_rand[:,2],  p0; lower = lb, upper = ub)
    β = f.param[2]
    return α,β
end




# num_rays = 400; 
# dt = 0.01; 
# v0 = 0.04
# Nt = 200:50:600


# @show α,β = compute_stretch_rand(v0, num_rays, dt, Nt)
   

