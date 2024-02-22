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
    py = 0.; S = 100
    points = [ [y, py] for y in yrange] 
    λ = transverse_growth_rates(df, points; Δt = Nt)
    λ .= λ./dt # normalize by time step
    return @strdict(λ)
end


function get_stretch_index(V; res = 500, a = 1, v0 = 1., dt = 0.01, Nt = 10000, prefix = "lyap", ξ)
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
    T = Nt*dt # We have to mulitply the variance by T 
    return mean(ml)*(v0^(-2/3))*ξ , var(ml)*2*(v0^(-2/3))*ξ*T
end



function compute_stretch_rand(v0_range, num_rays, num_angles, dt; Nt =40)
gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1 )
l_rand = zeros(length(v0_range),num_angles,2)
γ = zeros(length(v0_range))
c = zeros(length(v0_range))

for n in 1:num_angles
   Threads.@threads for k in 1:length(v0_range)
        # Correlated random pot 
        correlation_scale = 0.1;
        sim_width = Nt*dt ; sim_height = 10.
        Vr = correlated_random_potential(sim_width, sim_height, correlation_scale, v0_range[k], n)
        s = savename("stretch_rand", @dict(n))
        α,β  = get_stretch_index(Vr; res = num_rays, a = 1, v0=v0_range[k], dt, Nt, prefix = s, ξ = correlation_scale)
        l_rand[k,n,:] = [α, β]
        @show a,b = [α, β]
        # @show γ[k] = gamma(a,b)
        # @show c[k] = gamma(a,b)/(v0^(-2/3))
    end
end
    m = mean(l_rand,dims =2)
    return vec(m[:,:,1]), vec(m[:,:,2])
end


function compute_stretch_fermi(v0_range, num_rays, num_angles, dt; Nt = 40)
    gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1 )
    angles = range(0, π/4, length = num_angles + 1)
    angles = angles[1:end-1]
    l_fer = zeros(length(v0_range),num_angles,2)
    γ = zeros(length(v0_range), num_angles)
    c = zeros(length(v0_range), num_angles)

    for (j, θ) in enumerate(angles)
        Threads.@threads for k in 1:length(v0_range)
            # Correlated random pot 
            lattice_a = 0.2; dot_radius = 0.2*0.25
            softness = 0.2; II = rotation_matrix(θ)
            V = LatticePotential(lattice_a*II, dot_radius, v0_range[k]; softness=softness)
            s = savename("stretch_fermi", @dict(θ))
            α,β  = get_stretch_index(V; res = num_rays, a = 1, v0 = v0_range[k], dt, Nt, prefix = s, ξ = lattice_a)
            l_fer[k,j,:] = [α, β]
            @show a,b = [α,β]
            # @show γ[k,j] = gamma(a,b)
            # @show c[k,j] = gamma(a,b)/(v0_range[k]^(-2/3))
        end
    end
    m = mean(l_fer,dims =2)
    return vec(m[:,:,1]), vec(m[:,:,2])
end

function compute_stretch_cos(v0_range, d, num_rays, num_angles, dt; Nt = 40)
    gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1 )
    angles = range(0, π/4, length = num_angles + 1)
    angles = angles[1:end-1]
    l_cos = zeros(length(v0_range),num_angles,2)
    γ = zeros(length(v0_range), num_angles)
    c = zeros(length(v0_range), num_angles)

    for (j, θ) in enumerate(angles)
        Threads.@threads for k in 1:length(v0_range)
            lattice_a = 0.2; dot_radius = 0.2*0.25
            softness = 0.2; II = rotation_matrix(θ)
            V = RotatedPotential(θ,              
                fermi_dot_lattice_cos_series(d,  
                lattice_a, dot_radius, v0_range[k]; softness))
            s = savename("stretch_cos_n", @dict(d, θ))
            α,β  = get_stretch_index(V; res = num_rays, a = 1, v0 = v0_range[k], dt, Nt, prefix = s, ξ = lattice_a)
            l_cos[k,j,:] = [α, β]
            println("θ=", θ, " v0=", v0_range[k])
            # @show a,b = [mean(α), mean(β)]*(v0_range[k]^(-2/3))
            # @show γ[k,j] = gamma(a,b)
            # @show c[k,j] = gamma(a,b)/(v0_range[k]^(-2/3))
        end
    end
    m = mean(l_cos,dims =2)
    return vec(m[:,:,1]), vec(m[:,:,2])
end



num_rays = 2000; 
num_angles = 50
Nt = 400
dt = 0.01; 
v0_range = range(0.02, 0.4, step = 0.04); 
mr,sr = compute_stretch_rand(v0_range, num_rays, num_angles, dt; Nt)
mf,sf = compute_stretch_fermi(v0_range, num_rays, num_angles, dt; Nt)
mc1,sc1 = compute_stretch_cos(v0_range, 1, num_rays, num_angles, dt; Nt)
mc6,sc6 = compute_stretch_cos(v0_range, 6, num_rays, num_angles, dt; Nt)

using JLD2
s = savename("stretch_factor", @dict(Nt, num_angles, num_rays),"jld2")
@save s mr sr mf sf mc1 sc1 mc6 sc6 v0_range
# gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1)
# c = gamma(mf,sf)./(v0_range.^(-2/3))

