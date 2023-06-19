using BranchedFlowSim
using CairoMakie
using Interpolations
using Statistics
using FFTW
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Makie
using ColorTypes


path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)
sim_height = 1
sim_width = 6
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.2

# Time steps to count branches.
ts = LinRange(0, sim_width, 100)

# Periodic potential
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

# Compute average over all angles
function get_lattice_mean_nb(num_rays, dt)::Vector{Float64}
    num_angles = 50
    angles = LinRange(0, π / 2, num_angles + 1)[1:end-1]
    grid_nb = zeros(length(ts), length(angles))
    Threads.@threads for di ∈ 1:length(angles)
        θ = angles[di]
        potential = LatticePotential(lattice_a * rotation_matrix(θ),
            dot_radius, dot_v0; offset=[lattice_a / 2, 0], softness=softness)
        grid_nb[:, di] = quasi2d_num_branches(num_rays, dt, ts, potential)
    end
    return vec(mean(grid_nb, dims=2))
end

num_sims = 50
@time "generated random potentials" rand_potentials = [
    correlated_random_potential(sim_width, sim_height, correlation_scale, v0)
    for i ∈ 1:num_sims
]

function get_rand_nb(num_rays, dt)::Vector{Float64}
    num_sims = length(rand_potentials)
    correlation_scale = 0.1
    nb = zeros(length(ts), num_sims)
    Threads.@threads for di ∈ 1:num_sims
        nb[:, di] = quasi2d_num_branches(num_rays, dt, ts, rand_potentials[di])
    end
    return vec(mean(nb, dims=2))
end

## Actually evaluate

# num_rays = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
num_rays = [500, 1000, 2000, 4000, 8000, 16000, 32000]
dt = [0.01, 0.005]

params = [(n, d) for n ∈ num_rays, d ∈ dt]

@time "lattice sims" nb_lattice = [
    @time "lattice sim $n $d" get_lattice_mean_nb(n, d)
    for (n, d) ∈ params
]

# @time "random sims" nb_rand = [
#     @time "random sim $n $d" get_rand_nb(n, d)
#     for (n, d) ∈ params
# ]

## Plot
data_title_fname = [
    (nb_lattice,
        LaTeXString("Number of branches, periodic lattice, \$a=$lattice_a\$"),
        "params_lattice.png"),
    # (nb_rand, LaTeXString("Number of branches, correlated random, \$l_c=0.1\$"), "params_rand.png")
]
for (data, title, fname) ∈ data_title_fname
    fig = Figure(resolution=(1024, 768))
    ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
        title=title,
        limits=((0, sim_width), (0, nothing)),
        yticks=Vector(0:500:30000)
    )

    linestyles = [:solid, :dash, :dot]
    for r ∈ 1:size(params)[1]
        for c ∈ 1:size(params)[2]
            n, d = params[r, c]
            lines!(ax, ts, data[r, c], label=L"n=%$n,\, dt=%$d", linestyle=linestyles[c])
        end
    end
    axislegend(ax, position=:lt)
    save(path_prefix * fname, fig, px_per_unit=2)
    display(fig)
end