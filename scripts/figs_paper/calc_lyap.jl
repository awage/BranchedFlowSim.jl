using DrWatson 
@quickactivate
using BranchedFlowSim
using ProgressMeter
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using ChaosTools
using StatsBase


function quasi2d_map!(du,u,p,t)
    y,py = u; potential, dt = p
    # kick
    du[2] = py + dt * force_y(potential, dt*t, y)
    # drift
    du[1] = y + dt * du[2]
    return nothing
end



function _get_lyap_1D(d) 
    @unpack V, dt, a, T, res = d
    df = DeterministicIteratedMap(quasi2d_map!, [0.4, 0.2], [V, dt])
    yrange = range(0, a, res)
    py = 0.
    λ = zeros(res)
    p = Progress(res, "Lyap calc") 
    Threads.@threads for k in eachindex(yrange)
        λ[k] = lyapunov(df, T; u0 = [yrange[k], py]) 
        next!(p)
    end
    return @strdict(λ, yrange, d)
end

function get_lyap_index(V, threshold; res = 500, a = 1, v0 = 1., dt = 0.01, T = 10000, prefix = "lyap")
    d = @dict(res,  a, v0,  T, dt, V) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_lyap_1D, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    @unpack λ = data
    @show mean(λ[λ .> 0])
    ind = findall(λ .> threshold)
    l_index = length(ind)/length(λ) 
    # return mean(λ[λ .> 0])/dt
    return l_index
end


# Comon parameters
num_rays = 5000; 
dt = 0.01; T = 2000; 
threshold = 0.001

num_angles = 50
angles = range(0, π/4, length = num_angles + 1)
angles = angles[1:end-1]

v0_range = range(0.04, 0.4, step = 0.02)
# v0_range = [0.04, 0.4] 
l_fermi = zeros(length(v0_range), num_angles)
l_cos = zeros(length(v0_range),num_angles,6)
l_rand = zeros(length(v0_range),num_angles)

for (j, θ) in enumerate(angles)
    for (k,v0) in enumerate(v0_range)
        # Fermi lattice
        lattice_a = 0.2; dot_radius = 0.2*0.25
        softness = 0.2; II = rotation_matrix(θ)
        V = LatticePotential(lattice_a*II, dot_radius, v0; softness=softness)
        s = savename("lyap_fermi", @dict(v0,θ))
        l = get_lyap_index(V, threshold; res = num_rays, a = lattice_a, v0, dt, T, prefix = s)
        l_fermi[k,j] = l   

        # Cosine sum 
        max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
        softness = 0.2; 
        degrees = [1, 6] #1:max_degree
        for degree ∈ degrees
            cos_pot = RotatedPotential(θ,              
                fermi_dot_lattice_cos_series(degree,  
                lattice_a, dot_radius, v0; softness))
            s = savename("lyap_cos", @dict(v0,θ,degree))
            l = get_lyap_index(cos_pot, threshold; res = num_rays, a = lattice_a, v0 , dt, T , prefix = s)
            l_cos[k,j,degree] = l
        end

        # Correlated random pot 
        correlation_scale = 0.1;
        sim_width = 20; sim_height = 20. 
        Vr = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, j)
        s = savename("lyap_rand", @dict(v0,j))
        l = get_lyap_index(Vr, threshold; res = num_rays, a = 1, v0, dt, T, prefix = s)
        l_rand[k,j] = l   
    end
end

mcos = mean(l_cos; dims = 2)
lc1 = vec(mcos[:,:,1])
lc6 = vec(mcos[:,:,6])
lf = vec(mean(l_fermi, dims =2))
lr = vec(mean(l_rand, dims =2))

fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"v_0", ylabel = "Lyap index", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30,  titlesize = 30, yscale = Makie.pseudolog10)
lines!(ax1, v0_range, lc1, linestyle = :dash, color = :black, label = L"V_{cos} ~ n = 1")
# lines!(ax1, v0_range, l_cos[:,2], color = :red, label = "Cos n=2")
# lines!(ax1, v0_range, l_cos[:,3], color = :green, label = "Cos n=3")
# lines!(ax1, v0_range, l_cos[:,4], color = :pink, label = "Cos n=4")
# lines!(ax1, v0_range, l_cos[:,5], color = :purple, label = "Cos n=5")
lines!(ax1, v0_range, lc6, color = :cyan, label = L"V_{cos} ~ n=6")
lines!(ax1, v0_range, lf, color = :blue, linestyle = :dash, label = L"Fermi")
lines!(ax1, v0_range, lr, color = :orange, label = L"Rand")

s = "comparison_lyap_index.png"
axislegend(ax1);
save(plotsdir(s),fig)
