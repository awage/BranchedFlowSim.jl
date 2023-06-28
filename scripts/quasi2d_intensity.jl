using BranchedFlowSim
using CairoMakie
using Interpolations
using Statistics
using LaTeXStrings
using ProgressMeter
using LinearAlgebra
using StaticArrays
using Makie
using ColorTypes
using JLD2

path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)
sim_height = 1
sim_width = 20
num_rays = 40000
dt = 0.01
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (E=1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.2

num_sims = 100

# y values for sampling the intensity
ys = LinRange(0, 1, 4000)
# smoothing parameter
smoothing_b = 0.03
# Time steps to count branches.
ts = LinRange(0, sim_width, round(Int, 10 * sim_width))

# Periodic potential
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

function make_integrable_potential(degree)
    return fermi_dot_lattice_cos_series(degree, lattice_a, dot_radius, v0, softness=softness)
end

function random_potential()
    return correlated_random_potential(sim_width, sim_height+2, correlation_scale, v0)
end

complex_int_degree = 8
complex_int =
    complex_separable_potential(complex_int_degree, lattice_a, dot_radius, v0; softness=softness)

function random_dot_potential()
    y_extra = 4
    xmin = -1
    xmax = sim_width + 1
    ymin = -y_extra
    ymax = sim_height + y_extra
    return random_fermi_potential(xmin, xmax, ymin, ymax, lattice_a, dot_radius, v0)
end

function maximum_intensity(int)
    return vec(maximum(int, dims=1))
end

function mean_max_intensity(pots, progress_text="")
    maxint_all = zeros(length(ts), length(pots))
    int_all = zeros(length(ys), length(ts), length(pots))
    p = Progress(length(pots); desc=progress_text, enabled=false)
    dy = ys[2] - ys[1]
    Threads.@threads for j ∈ 1:length(pots)
        int = quasi2d_intensity(num_rays, dt, ts, ys, pots[j], b=smoothing_b)
        int_all[:,:, j] = int
        maxint_all[:, j] = maximum_intensity(int)
        next!(p)
    end
    end_p = vec(dy * sum(int_all[:,end,:],dims=1))
    println("$progress_text total_p=$(mean(end_p)) ($(minimum(end_p)) - $(maximum(end_p)))")
    return vec(mean(maxint_all, dims=2)), maxint_all,int_all
end

@time "generate rpots" rpots = [random_potential() for _ ∈ 1:num_sims]
num_angles = 50
angles = LinRange(0, pi / 2, num_angles + 1)[1:end-1]
lattice_pots = [LatticePotential(lattice_a * rotation_matrix(θ), dot_radius, v0)
                for θ ∈ angles
]
int1_pot =make_integrable_potential(1)
int_pots = [ RotatedPotential(θ, int1_pot) for θ ∈ angles]
## Actually evaluate

GC.gc()
@time "random intensity" rand_maxint, rand_int_all,rand_int = mean_max_intensity(rpots, "correlated random")
@time "lattice intensity" lattice_maxint, lattice_int_all,lattice_int = mean_max_intensity(lattice_pots, "lattice")
@time "int intensity" int_maxint, int_int_all,int_int = mean_max_intensity(int_pots, "integrable")


## Plot results
fig = Figure()
ax = Axis(fig[1, 1],
    title="Maximum intensity",
    limits=(nothing, (0, 10)),
    yticks=0:10)
# heatmap(ts, ys, int_rand')
lines!(ax, ts, rand_maxint, label="correlated random")
lines!(ax, ts, lattice_maxint, label="periodic lattice")
lines!(ax, ts, int_maxint, label="integrable")
axislegend(ax)
save(path_prefix * "max_intensity.png", fig, px_per_unit=2)
save(path_prefix * "max_intensity.pdf", fig)
# lines(ys, int_rand[:, end÷2], axis=(title="Profile at w/2",))
# display(maximum_intensity(int_rand)[end])
display(fig)