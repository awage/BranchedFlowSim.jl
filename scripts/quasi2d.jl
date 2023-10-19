using BranchedFlowSim
using CairoMakie
using Interpolations
using Statistics
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Makie
using ColorTypes


function pixel_heatmap(path, data; kwargs...)
    scene = Scene(camera=campixel!, resolution=size(data))
    heatmap!(scene, data; kwargs...)
    save(path, scene)
end

path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)

sim_height = 1
sim_width = 6
num_rays = 1000
dt = 1 / 512
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.2
num_sims = 100


# Time steps to count branches.
ts = LinRange(0, sim_width, 200)

# Periodic potential
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

function random_potential()
    return correlated_random_potential(sim_width, sim_height, correlation_scale, v0)
end

function get_random_nb()::Vector{Float64}
    # Run simulations to count branches
    num_branches = zeros(length(ts), num_sims)
    Threads.@threads for i ∈ 1:num_sims
        potential = random_potential()
        num_branches[:, i] = quasi2d_num_branches(num_rays, dt, ts, potential)
    end
    return vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)
end

# Compute average over all angles
function get_lattice_mean_nb()::Vector{Float64}
    angles = LinRange(0, π / 2, 51)[1:end-1]
    grid_nb = zeros(length(ts), length(angles))
    Threads.@threads for di ∈ 1:length(angles)
        θ = angles[di]
        potential = LatticePotential(lattice_a * rotation_matrix(θ),
            dot_radius, dot_v0; offset=[lattice_a / 2, 0], softness=softness)
        grid_nb[:, di] = quasi2d_num_branches(num_rays, dt, ts, potential) / sim_height
    end
    return vec(sum(grid_nb, dims=2) / length(angles))
end


# Compute average over all angles
function get_branch_mixed(r, a; num_rays = 10000, dt = 1/512, ts = LinRange(0,6,200), sim_height = 1)
    potential = CosMixedPotential(r, a)
    nb = quasi2d_num_branches(num_rays, dt, ts, potential) / sim_height
    return nb
end

sim_width = 200
ts = LinRange(0,sim_width,200)

@time "mixed sim" nb_mixed = get_branch_mixed(0.4, 2π; num_rays = 1000,  ts = ts )
sim_width = 6
ts = LinRange(0,sim_width,200)
# @time "rand sim" nb_rand = get_random_nb()
# @time "lattice sim" nb_lattice = get_lattice_mean_nb()
#
## Make a comparison between rand, lattice, and int

fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
    title=LaTeXString("Number of branches") )

lines!(ax,  nb_rand, label=L"Metzger, $l_c=%$correlation_scale$")
lines!(ax,  nb_lattice, label=L"periodic lattice, $a=%$(lattice_a)$ (mean)")
lines!(ax,  nb_mixed, label=L"Cos mixed r")
axislegend(ax, position=:lt)
save(path_prefix * "branches.png", fig, px_per_unit=2)
display(fig)



