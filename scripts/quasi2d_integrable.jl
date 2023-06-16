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
sim_width = 4
num_rays = 4000
dt = 0.01
# num_rays = 512
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.10

num_sims = 100

# Time steps to count branches.
ts = LinRange(0, sim_width, 100)

# Periodic potential
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

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

function get_nb_int(V)::Vector{Float64}
    num_angles = 50
    angles = LinRange(0, π / 2, num_angles + 1)[1:end-1]
    int_nb_arr = zeros(length(ts), length(angles))
    Threads.@threads for di ∈ 1:length(angles)
        θ = angles[di]
        potential = RotatedPotential(θ, V)
        # tys = LinRange(0, 1, 2num_rays)
        int_nb_arr[:, di] = quasi2d_num_branches(num_rays, dt, ts, potential) / sim_height
    end
    return vec(sum(int_nb_arr, dims=2) / length(angles))
end

function potential_label(degree)
    lbl = L"\sum_{n+m\le%$(degree)}a_{nm}\cos(nkx)\cos(mky)"
end

function make_integrable_potential(degree)
    return fermi_dot_lattice_cos_series(degree, lattice_a, dot_radius, v0, softness=softness)
end

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

complex_int = begin
    zpot8 = make_integrable_potential(8)
    w = Matrix( zpot8.w)
    w[2:end,2:end] .= 0
    CosSeriesPotential{SMatrix{9,9,Float64, 81}}(w, zpot8.k)
end

## Actually evaluate

@time "rand sim" nb_rand = get_random_nb()
@time "lattice sim" nb_lattice = get_lattice_mean_nb()
max_degree = 6
int_potentials = [make_integrable_potential(degree) for degree ∈ 1:max_degree]
@time "all int sims" nb_int = [
    @time "int deg=$deg" get_nb_int(int_potentials[deg])
    for deg ∈ 1:length(int_potentials)]

@time "cint sim" nb_cint = get_nb_int(complex_int)

## Plots

fig = Figure(resolution=(1024, 768))
ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
    title=LaTeXString("Number of branches, $num_rays rays, dt=$dt"),
    limits=((0, sim_width), (0, nothing)),
    yticks=[1,10,100,1000],
    yscale=Makie.pseudolog10
)

lines!(ax, ts, nb_rand, label=L"Metzger, $l_c=%$correlation_scale$", linestyle=:dash)
lines!(ax, ts, nb_lattice, label=L"periodic lattice, $a=%$(lattice_a)$ (mean)")
lines!(ax, ts, nb_cint, label=L"XXX, $a=%$(lattice_a)$ (mean)", linestyle=:dot)
for degree ∈ 1:max_degree
    #lbl=L"degree %$(degree) $\cos$ approx $a=%$(lattice_a)$ (mean)")
    lbl = potential_label(degree)
    lines!(ax, ts, nb_int[degree], label=lbl)
end
axislegend(ax, position=:lt)
save(path_prefix * "int_branches.png", fig, px_per_unit=2)
display(fig)

## Visualize simulations
num_rays = 2048
lattice_mat = lattice_a * rotation_matrix(-pi / 10)
rot_lattice = LatticePotential(lattice_mat, dot_radius, v0)
quasi2d_visualize_rays(path_prefix * "lattice_rot_sim.png", num_rays, sim_width,
    rot_lattice
)
quasi2d_visualize_rays(path_prefix * "cint_rot_sim.png", num_rays, sim_width,
    RotatedPotential(pi/10, complex_int)
)
for degree ∈ 1:max_degree
    quasi2d_visualize_rays(path_prefix * "int_$(degree)_rot_sim.png", 
        num_rays, sim_width, 
        RotatedPotential(pi / 10, int_potentials[degree])
    )
end