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

function compare_shapes(ts, left_nb, right_nb, llabel, rlabel; same_axis=false)
    fig = Figure(resolution=(800, 600))
    rcolor = RGB(0.8, 0, 0)

    # Scale such that midpoints are equal
    llimits = (0.0, maximum(left_nb))
    rlimits = llimits .* (right_nb[end÷2] / left_nb[end÷2])
    xlimits = (0.0, sim_width)

    axl = Axis(fig[1, 1],
        xlabel=L"t", ylabel=L"N_b", title=LaTeXString("Number of branches"), limits=(xlimits, llimits))
    if !same_axis
        axr = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
            yaxisposition=:right, ygridcolor=RGBAf(0.8, 0.0, 0.0, 0.12),
            yticklabelcolor=rcolor, limits=(xlimits, rlimits))
        hidespines!(axr)
        hidexdecorations!(axr)
    else
        axr = axl
    end

    lines!(axl, ts, left_nb, label=llabel, linestyle=:dash)
    lines!(axr, ts, right_nb, label=rlabel, linestyle=:dot,
        color=rcolor)
    axislegend(axl, position=:lt)
    if !same_axis
        axislegend(axr, position=:rb)
    end
    return fig
end

path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)
sim_height = 1
sim_width = 4
num_rays = 512
# num_rays = 512
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.2
dynamic_rays = false

num_sims = 100

Ny = num_rays
Nx = round(Int, num_rays * (sim_width / sim_height))
ys = LinRange(0, sim_height, Ny + 1)[1:end-1]
xs = LinRange(0, sim_width, Nx + 1)[1:end-1]

# Time steps to count branches.
ts = LinRange(0, xs[end], 200)


# Periodic potential
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

function random_potential()
    # Use a different discretization for the random potential, no need to
    # match the number of rays
    rand_N = 256
    rNy = rand_N
    rNx = round(Int, rand_N * (sim_width / sim_height))
    rys = LinRange(0, sim_height, rNy + 1)[1:end-1]
    rxs = LinRange(0, sim_width, rNx + 1)[1:end-1]
    pot_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
    return PeriodicGridPotential(xs, ys, pot_arr)
end

function get_random_nb()::Vector{Float64}
    # Run simulations to count branches
    num_branches = zeros(length(ts), num_sims)
    Threads.@threads for i ∈ 1:num_sims
        potential = random_potential()
        num_branches[:, i] = quasi2d_num_branches(xs, ys, ts, potential,
            dynamic_rays=dynamic_rays)
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
        grid_nb[:, di] = quasi2d_num_branches(xs, ys, ts, potential, dynamic_rays=dynamic_rays) / sim_height
    end
    return vec(sum(grid_nb, dims=2) / length(angles))
end

# Integrable potential
function integrable_pot(x::Real, y::Real)::Float64
    v0 * (cos(2pi * x / lattice_a) + cos(2pi * y / lattice_a))
end

function get_nb_int()::Vector{Float64}
    num_angles = 50
    angles = LinRange(0, π / 2, num_angles+1)[1:end-1]
    int_nb_arr = zeros(length(ts), length(angles))
    Threads.@threads for di ∈ 1:length(angles)
        θ = angles[di]
        potential = RotatedPotential(θ, integrable_pot)
        # tys = LinRange(0, 1, 2num_rays)
        int_nb_arr[:, di] = quasi2d_num_branches(xs, ys, ts, potential,
            dynamic_rays=false) / sim_height
    end
    return vec(sum(int_nb_arr, dims=2) / length(angles))
end

function random_dot_potential()
    y_extra = 2
    xmin = -1
    xmax = sim_width + 1
    ymin = -y_extra
    ymax = sim_height + y_extra
    box_dims = 
    num_dots = round(Int, (xmax-xmin)*(ymax-ymin)/lattice_a^2)
    locs = zeros(2, num_dots)
    for i ∈ 1:num_dots
        locs[:,i] = [xmin,ymin] + rand(2) .* [xmax-xmin, ymax-ymin]
    end
    return RepeatedPotential(
        locs,
        FermiDotPotential(dot_radius, v0),
        lattice_a
    )
end

function get_nb_rand_dots()::Vector{Float64}
    # Run simulations to count branches
    num_branches = zeros(length(ts), num_sims)
    Threads.@threads for i ∈ 1:num_sims
        potential = random_dot_potential()
        num_branches[:, i] = quasi2d_num_branches(xs, ys, ts, potential,
            dynamic_rays=dynamic_rays)
    end
    return vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)
end

## Actually evaluate

@time "rand sim" nb_rand = get_random_nb()
@time "lattice sim" nb_lattice = get_lattice_mean_nb()
@time "int sim" nb_int = get_nb_int()
@time "rand dot sim" nb_rand_dots = get_nb_rand_dots()

## Make a comparison between rand, lattice, and int

fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
    title=LaTeXString("Number of branches"), limits=((0, sim_width), (0, nothing)))

lines!(ax, ts, nb_rand, label=L"Metzger, $l_c=%$correlation_scale$")
lines!(ax, ts, nb_lattice, label=L"periodic lattice, $a=%$(lattice_a)$ (mean)")
lines!(ax, ts, nb_rand_dots, label=LaTeXString("random dots"))
lines!(ax, ts, nb_int,
    label=L"${v_0}(\cos(2\pi x / a)+\cos(2\pi y / a))$, $a=%$(lattice_a)$ (mean)")
axislegend(ax, position=:lt)
save(path_prefix * "branches.png", fig, px_per_unit=2)
display(fig)

## Visualize simulations
rand_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
rand_potential = PeriodicGridPotential(xs, ys, rand_arr)
quasi2d_visualize_rays(path_prefix * "rand_sim.png", xs, ys, rand_potential)
quasi2d_visualize_rays(path_prefix * "int_sim.png", xs, ys, integrable_pot)
potential = LatticePotential(lattice_a * rotation_matrix(0),
    dot_radius, v0, offset=[0, 0])
lattice_mat = lattice_a * rotation_matrix(-pi/10)
rot_lattice = LatticePotential(lattice_mat, dot_radius, v0)
quasi2d_visualize_rays(path_prefix * "lattice_sim.png", xs, ys,
    potential
)
quasi2d_visualize_rays(path_prefix * "int_rot_sim.png", xs, LinRange(0, 1, 1024),
    RotatedPotential(pi / 10, integrable_pot),
    triple_y = true
)
quasi2d_visualize_rays(path_prefix * "lattice_rot_sim.png", xs, LinRange(0, 1, 1024),
    rot_lattice,
    triple_y = true
)

rand_dot_potential = random_dot_potential()

quasi2d_visualize_rays(path_prefix * "rand_dots_sim.png", xs,
    ys, rand_dot_potential, triple_y=true)