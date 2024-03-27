using DrWatson 
@quickactivate
using BranchedFlowSim
using ProgressMeter
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StatsBase
using CodecZlib

include(srcdir("utils.jl"))

function get_branch_number(v0, V, y_init, xs, num_rays, j, K; prefix = "fermi_dec") 
    d = @dict(V, v0, y_init, xs, num_rays, j, K)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_branches, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    @unpack nb_br = data
    return nb_br
end

function compute_branches(d)
    @unpack V, y_init, xs, K  = d
    nb_br = manifold_track(xs, y_init, V; K)
    return @strdict(nb_br)
end

function count_branches(y_ray, dy; K = 1.)
    dy_ray = diff(y_ray)/dy
    ind = findall(abs.(dy_ray) .< K)
    str = find_consecutive_streams(ind) 
    br = count(s -> length(s) ≥ 3, str)
    return br
end

function manifold_track(xs, y_init, potential; K = 1.)
    dt = xs[2] - xs[1]
    num_rays = length(y_init)
    ray_y = deepcopy(collect(y_init))
    ray_py = zeros(num_rays)
    nb_br = zeros(Int64,length(xs))
    dyr = (ray_y[2]-ray_y[1])
    xi = 1; x = xs[1]
    while x <= xs[end] + dt
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        x += dt
        while xi <= length(xs) &&  xs[xi] <= x 
            nb_br[xi] = count_branches(ray_y, dyr; K)
            xi += 1
        end
    end
    return  nb_br
end

v0 = 0.1; dt = 0.01;  num_rays = 250000; K = 1.; n_avg = 40
xs = range(0,20, step = dt)
y_init = range(-30*0.2, 30*0.2, length = num_rays)

pargs = (yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40) 
fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1];  xlabel = L"x", ylabel = L"n_{br}", pargs...) 
# ax2= Axis(fig[1, 2];  xlabel = L"x", ylabel = L"n_{br}", pargs...) 

ξ = 0.1; nb_br_v = Vector{Vector{Float64}}()
Threads.@threads for j = 1:n_avg
    V = correlated_random_potential(20*ξ,20*ξ, ξ, v0, j)
    nb_br = get_branch_number(v0, V, y_init, xs, num_rays, j, K; prefix = "rand_dec") 
    push!(nb_br_v, nb_br)
end
lines!(ax1, xs, mean(nb_br_v, dims = 1)[1]; label = "Random correlated")

a = 0.2; dot_radius = 0.2*0.25; softness = 0.2; θ_range = range(0,π/4, length = n_avg) 
nb_br_v = Vector{Vector{Float64}}()
Threads.@threads for j = 1:n_avg
    V = LatticePotential(a*rotation_matrix(θ_range[j]), dot_radius, v0; softness=softness)
    nb_br = get_branch_number(v0, V, y_init, xs, num_rays, j, K; prefix = "fermi_dec") 
    push!(nb_br_v, nb_br)
end
lines!(ax1, xs, mean(nb_br_v, dims = 1)[1]; color = :red, label = "Fermi dot")

a = 0.2; dot_radius = 0.2*0.25; softness = 0.2; θ_range = range(0,π/4, length = n_avg) 
nb_br_v = Vector{Vector{Float64}}()
Threads.@threads for j = 1:n_avg
    V = RotatedPotential(θ_range[j],
        fermi_dot_lattice_cos_series(1,
        a, dot_radius, v0; softness))
    nb_br = get_branch_number(v0, V, y_init, xs, num_rays, j, K; prefix = "cos0_dec") 
    push!(nb_br_v, nb_br)
end
lines!(ax1, xs, mean(nb_br_v, dims = 1)[1]; color = :green, label = "Periodic integrable")
axislegend(ax1);
s = "branch_comp.png"
save(plotsdir(s),fig)
