using BranchedFlowSim
using CairoMakie
using Printf
using Makie

rs = [
    0.5 0.5
    1.5 0.6
]'
ps = [
    0.5 0.5
    1.5 0.6
]'

intersections = Vector{Float64}[]
qs = Float64[]

lattice_intersections([0, 1], [1, 0], [0, 0], rs, ps) do r, p, q, w
    push!(intersections, r)
    push!(qs, q)
end

intersections = hcat(intersections...)

fig = Figure()
ax = Axis(fig[1,1])

lines!(ax, rs)

scatter!(ax, intersections)
text!(ax, intersections, text=["q=$q" for q ∈ qs])

display(fig)

lattice_a::Float64 = 0.2
dot_radius::Float64 = 0.25 * lattice_a
v0::Float64 = 0.1
softness = 0.1
dt::Float64 = 0.01

pot = LatticePotential(lattice_a * I, dot_radius, v0;
    softness=softness, offset=[lattice_a/2, lattice_a/2])
# pot = correlated_random_potential(4,4,0.1,v0)
# pot= fermi_dot_lattice_cos_series(1, lattice_a, dot_radius, -v0)

num_particles = 3
r0 = lattice_a * [0.1, 0.05] * ones(num_particles)'
angles = LinRange(0, 2pi, num_particles + 1)[1:end-1]
p0 = hcat(([cos(θ), sin(θ)] for θ ∈ angles)...)
    
normalize_momentum!(p0, r0, pot)

T = 5.0

fig = Figure()
ax = Axis(fig[1,1], limits=((-1,1),(-1,1)))

intersect = [0.2,0.2]
step = [0.2, 0.0]

for k ∈ -10:10
    lines!(ax, hcat(k * step-100*intersect, k * step + 100*intersect))
end

ray_trajectories(r0, p0, pot, T, dt) do ts, rs, ps
    lines!(ax, rs)
    lattice_intersections([0.2, 0.2], [0.2, 0], [0, 0], rs, ps) do r, p, q, w
        scatter!(ax, r)
        text!(ax, r, text=@sprintf("q=%.2f", q))
    end
end

display(fig)

