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

potential = LatticePotential(lattice_a * rotation_matrix(pi / 10),
    dot_radius, dot_v0, softness=softness)

## Actually evaluate

num_rays = [4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
# num_rays = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
dt = [0.01]

params = [(n, d) for n ∈ num_rays, d ∈ dt]

@time "lattice sims" nb_lattice = [
    @time "lattice sim $n $d" quasi2d_num_branches(n, d, ts, potential)
    for (n, d) ∈ params
]

## Dynamic stuff
#=
start_rays = 1024
ray_y = Vector{Float64}(LinRange(0, 1, start_rays + 1)[1:end-1])
threshold = 0.005
@time "dynamic stuff" begin
    end_t = 1
    while end_t < sim_width
        global end_t
        _, final_ray_y, _ = quasi2d_num_branches(ray_y, dt[1], [end_t], potential, return_rays=true)
        new_ray_y = Vector{Float64}()
        push!(new_ray_y, ray_y[1])
        added = 0
        for j ∈ 2:length(ray_y)
            if abs(final_ray_y[j] - final_ray_y[j-1]) > threshold
                push!(new_ray_y, (ray_y[j-1] + ray_y[j]) / 2)
                added += 1
            end
            push!(new_ray_y, ray_y[j])
        end
        ray_y = new_ray_y
        println("end_t=$end_t, added=$added, num_rays=$(length(ray_y))")
        if added == 0
            end_t += 1
        end
    end
end
dyn_branches = quasi2d_num_branches(ray_y, dt[1], ts, potential)
=#

## 

@time "all rand rays" rand_rays_nb= [
    @time "rand rays $n $d" quasi2d_num_branches(sort(rand(n)), d, ts, potential)
    for (n, d) ∈ params
]

## Plot
data, title, fname = (nb_lattice,
    LaTeXString("Number of branches, periodic lattice, \$a=$lattice_a\$"),
    "dynamic_rays.png")
fig = Figure(resolution=(1024, 768))
ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
    title=title,
    limits=((0, sim_width), (0, nothing)), yticks=Vector(0:100:3000)
)

colors = Makie.wong_colors()
ci = 1
for r ∈ 1:size(params)[1]
    for c ∈ 1:size(params)[2]
        n, d = params[r, c]
        color = colors[ci]
        ci = 1 + ci%length(colors)
        lines!(ax, ts, data[r, c], color=color, label=L"n=%$n,\, dt=%$d")
        lines!(ax, ts, rand_rays_nb[r, c], color=color, label=L"n=%$n,\, dt=%$d (rand)", linestyle=:dash)
    end
end
# lines!(ax, ts, dyn_branches, label=L"dynamic, $n=%$(length(ray_y))$")

axislegend(ax, position=:lt)
save(path_prefix * fname, fig, px_per_unit=2)
display(fig)


