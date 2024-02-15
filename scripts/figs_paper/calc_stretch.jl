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
    py = 0.; S = 100
    points = [ [y, py] for y in yrange] 
    λ = transverse_growth_rates(df, points; Δt = Nt)
    # λ = zeros(res, S)
    # Threads.@threads for k in 1:length(yrange)
    #     λ[k,:] = transverse_growth_rates(df, [yrange[k] py]; Δt = Nt, S)
    # end
    λ .= λ./dt # normalize by time step
    return @strdict(λ,  d)
end


function get_stretch_index(V; res = 500, a = 1, v0 = 1., dt = 0.01, Nt = 10000, prefix = "lyap")
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
    @show mean(ml)
    @show std(ml)*2
    sl = vec(std(λ; dims = 2))
    return ml, std(ml)*2
end



function compute_stretch_rand(v0_range, num_rays, dt)
gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1 )
l_rand = zeros(length(v0_range),2)
γ = zeros(length(v0_range))
c = zeros(length(v0_range))
for (k,v0) in enumerate(v0_range)
    # Correlated random pot 
    correlation_scale = 0.1;
    t0 = v0^(-2/3)
    Nt = round(Int, t0*100)
    sim_width = 20; sim_height = 20.
    Vr = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, 100)
    s = savename("stretch_rand", @dict(v0))
    α,β  = get_stretch_index(Vr; res = num_rays, a = 2, v0, dt, Nt, prefix = s)
    l_rand[k,:] = [mean(α), mean(β)]*(v0^(-2/3)) 
    @show a,b = [mean(α), mean(β)]*(v0^(-2/3))
    @show γ[k] = gamma(a,b)
    @show c[k] = gamma(a,b)/(v0^(-2/3))
end
    return l_rand
end


function compute_stretch_fermi(v0_range, num_rays, num_angles, dt)
    gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1 )
    angles = range(0, π/4, length = num_angles + 1)
    angles = angles[1:end-1]
    l_fer = zeros(length(v0_range),num_angles,2)
    γ = zeros(length(v0_range), num_angles)
    c = zeros(length(v0_range), num_angles)

    for (j, θ) in enumerate(angles)
        for (k,v0) in enumerate(v0_range)
            # Correlated random pot 
            correlation_scale = 0.1;
            # t0 = v0^(-2/3)
            Nt = round(Int, 40/dt)
            lattice_a = 0.2; dot_radius = 0.2*0.25
            softness = 0.2; II = rotation_matrix(θ)
            V = LatticePotential(lattice_a*II, dot_radius, v0; softness=softness)
            s = savename("stretch_fermi", @dict(v0, θ))
            α,β  = get_stretch_index(V; res = num_rays, a = 2, v0, dt, Nt, prefix = s)
            l_fer[k,j,:] = [mean(α), mean(β)]*(v0^(-2/3)) 
            @show a,b = [mean(α), mean(β)]*(v0^(-2/3))
            @show γ[k,j] = gamma(a,b)
            @show c[k,j] = gamma(a,b)/(v0^(-2/3))
        end
    end
    return l_fer
end

function compute_stretch_cos(v0_range, d, num_rays, num_angles, dt)
    gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1 )
    angles = range(0, π/4, length = num_angles + 1)
    angles = angles[1:end-1]
    l_fer = zeros(length(v0_range),num_angles,2)
    γ = zeros(length(v0_range), num_angles)
    c = zeros(length(v0_range), num_angles)

    for (j, θ) in enumerate(angles)
        for (k,v0) in enumerate(v0_range)
            # Correlated random pot 
            correlation_scale = 0.1;
            # t0 = v0^(-2/3)
            Nt = round(Int, 40/dt)
            lattice_a = 0.2; dot_radius = 0.2*0.25
            softness = 0.2; II = rotation_matrix(θ)
            V = RotatedPotential(θ,              
                fermi_dot_lattice_cos_series(d,  
                lattice_a, dot_radius, v0; softness))
            s = savename("stretch_cos_n", @dict(d, θ))
            α,β  = get_stretch_index(V; res = num_rays, a = 2, v0, dt, Nt, prefix = s)
            l_fer[k,j,:] = [mean(α), mean(β)]*(v0^(-2/3)) 
            @show a,b = [mean(α), mean(β)]*(v0^(-2/3))
            @show γ[k,j] = gamma(a,b)
            @show c[k,j] = gamma(a,b)/(v0^(-2/3))
        end
    end
    return c
end



num_rays = 300; 
num_angles = 20
dt = 0.01; 
v0_range = range(0.02, 0.4, step = 0.01); 
c_r = compute_stretch_rand(v0_range, num_rays, dt)
c_f = compute_stretch_fermi(v0_range, num_rays, num_angles, dt)
c_c1 = compute_stretch_cos(v0_range, 1, num_rays, num_angles, dt)
c_c6 = compute_stretch_cos(v0_range, 6, num_rays, num_angles, dt)

