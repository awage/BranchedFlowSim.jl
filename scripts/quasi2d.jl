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
v0 = ϵ_percent * 0.01 * 0.5
softness = 0.2
dynamic_rays = false

num_sims = 100

Ny = num_rays
Nx = round(Int, num_rays * (sim_width / sim_height))
ys = LinRange(0, sim_height, Ny + 1)[1:end-1]
xs = LinRange(0, sim_width, Nx + 1)[1:end-1]

# Time steps to count branches.
ts = LinRange(0, xs[end], 200)


function 
# Run simulations to count branches
num_branches = zeros(length(ts), num_sims)
Threads.@threads for i ∈ 1:num_sims
    pot_arr = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
    potential = PeriodicGridPotential(xs, ys, pot_arr)
    num_branches[:, i] = quasi2d_num_branches(xs, ys, ts, potential,
        dynamic_rays=dynamic_rays)
end
nb_rand = vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)

# Periodic potential
lattice_a = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

fig = Figure(resolution=(600, 600))
ax = Axis(fig[1, 1],
    xlabel=L"t", ylabel=L"N_b",
    title=L"Number of branches, $\epsilon=%$ϵ_percent%%$")

lines!(ax, ts, nb_rand, label=L"random $l_c=%$correlation_scale$ (mean) ", linestyle=:dash)
for deg ∈ [0, 15, 30, 45]
    # lattice_θ = π / 6
    lattice_θ = deg2rad(deg)
    # lattice_θ =0
    potential = LatticePotential(lattice_a * rotation_matrix(lattice_θ),
        dot_radius, dot_v0; offset=[lattice_a / 2, 0], softness=softness)

    grid_nb = quasi2d_num_branches(xs, ys, ts, potential, dynamic_rays=dynamic_rays) / sim_height

    lines!(ax, ts, grid_nb, label=L"grid ($a=%$(lattice_a)$) %$(deg)°")
end

# Compute average over all angles
angles = LinRange(0, π / 2, 51)[1:end-1]
grid_nb = zeros(length(ts), length(angles))
Threads.@threads for di ∈ 1:length(angles)
    θ = angles[di]
    potential = LatticePotential(lattice_a * rotation_matrix(θ),
        dot_radius, dot_v0; offset=[lattice_a / 2, 0], softness=softness)
    grid_nb[:, di] = quasi2d_num_branches(xs, ys, ts, potential, dynamic_rays=dynamic_rays) / sim_height
end
grid_nb_mean = vec(sum(grid_nb, dims=2) / length(angles))
lines!(ax, ts, grid_nb_mean, label=L"grid ($a=%$(lattice_a)$) (mean)", linestyle=:dot)

# Integrable potential

integrable_pot(x, y) = v0 * (cos(2pi * x / lattice_a) + cos(2pi * y / lattice_a))

angles = LinRange(0, π / 2, 51)[1:end-1]
int_nb = zeros(length(ts), length(angles))
Threads.@threads for di ∈ 1:length(angles)
    θ = angles[di]
    potential = RotatedPotential(θ, integrable_pot)
    tys = LinRange(0, 1, 2num_rays)
    int_nb[:, di] = quasi2d_num_branches(xs, tys, ts, potential,
     dynamic_rays=false) / sim_height
end
int_nb_mean = vec(sum(int_nb, dims=2) / length(angles))

lines!(ax, ts, int_nb_mean, label=L"integrable (mean)$.$", linestyle=:dashdot)

axislegend(position=:lt)
display(fig)
save(path_prefix * "branches.png", fig, px_per_unit=2)


## Make a comparison between rand and grid but scale it
fig = compare_shapes(ts, nb_rand, grid_nb_mean,
    L"Metzger, $l_c=%$correlation_scale$",
    L"periodic lattice, $a=%$(lattice_a)$ (mean)", same_axis=true)
display(fig)
save(path_prefix * "compare_grid_rand.png", fig, px_per_unit=2)

# 
fig = compare_shapes(ts, nb_rand, int_nb_mean,
    L"Metzger, $l_c=%$correlation_scale$",
    L"${v_0}(\cos(2\pi x / a)+\cos(2\pi y / a))$, $a=%$(lattice_a)$ (mean)",
    same_axis=true)
display(fig)
save(path_prefix * "compare_int_rand.png", fig, px_per_unit=2)

fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
    title=LaTeXString("Number of branches"), limits=((0, sim_width), (0, nothing)))

lines!(ax, ts, nb_rand, label=L"Metzger, $l_c=%$correlation_scale$")
lines!(ax, ts, grid_nb_mean, label=L"periodic lattice, $a=%$(lattice_a)$ (mean)")
lines!(ax, ts, int_nb_mean,
    label=L"${v_0}(\cos(2\pi x / a)+\cos(2\pi y / a))$, $a=%$(lattice_a)$ (mean)")
axislegend(ax, position=:lt)
save(path_prefix * "compare_three.png", fig, px_per_unit=2)
display(fig)