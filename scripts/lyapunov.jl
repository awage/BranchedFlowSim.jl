using BranchedFlowSim
using Makie
using Printf
using Random
using ArgParse
using StaticArrays
using LinearAlgebra
using CairoMakie
using HDF5

s = ArgParseSettings()
@add_arg_table s begin
    "--grid_size"
    help = "width and height of the grid used for the Poincaré section plot"
    arg_type = Int
    default = 600
    "--dt"
    help = "time step of integration"
    arg_type = Float64
    default = 0.005
    "--output_dir"
    help = "Directory for outputs"
    arg_type = String
    default = "outputs/classical/"
end

add_potential_args(s,
    default_potentials="fermi_lattice",
    defaults=Dict(
        "cos_degrees" => "2",
        "num_angles" => 1,
        "num_sims" => 1,
        "lattice_a" => 1.0,
        "fermi_dot_radius" => 0.25,
        "fermi_softness" => 0.20,
    )
)

function get_r0_p0(pot, y, py)
    r0 = [0, y]
    V = pot(r0[1], r0[2])
    if (py^2 / 2 + V >= 0.5)
        return nothing
    end
    px = sqrt(2 * (0.5 - V) - py^2)
    p0 = [px, py]
    return r0, p0
end

function nearest_index(range, x)
    dx = range[2] - range[1]
    return round(Int, 1 + (x - range[1]) / dx)
end

function lyapunov_exponent_and_hits(pot, ris, pis, r_idx, p_idx, intersect, step, dt)
    T = 4000
    mapper = PoincareMapper(pot, ris[r_idx], pis[p_idx], intersect, step, [0, 0], dt)
    le = local_lyapunov_exponent(pot,
        particle_position(mapper),
        particle_momentum(mapper),
        dt, T
    )
    N = length(ris)
    M = length(pis)
    # Time per simulation
    hits = SVector{2,Int64}[SA[r_idx, p_idx]]
    orig_r = ris[r_idx]
    orig_p = pis[p_idx]
    while true
        int = next_intersection!(mapper)
        if isnothing(int)
            # No intersection was found in reasonable time.
            println("Bad trajectory: r=$orig_r p=$orig_p")
            return 0.0, []
        end
    # 
        if int.t > T / 2
           break
        end
        r = int.r_int
        p = int.p_int
        if r > ris[end]
            r = 2*ris[end] - r
        end
        if p < pis[begin]
            p = pis[begin] - p
        end
        r_idx = nearest_index(ris, r)
        p_idx = nearest_index(pis, p)
        @assert 1 <= r_idx <= N
        @assert 1 <= p_idx <= M
        push!(hits, SA[r_idx, p_idx])
    end
    return le, hits
end

parsed_args = parse_args(ARGS, s)

potentials = get_potentials_from_parsed_args(parsed_args, 1, 1)
if length(potentials) != 1
    println("potentials=$potentials")
    error("Exactly one potential is required, got $(length(potentials))")
end
pot = potentials[1].instances[1]
pot_name = potentials[1].name

W = parsed_args["grid_size"]
dt = parsed_args["dt"]

lattice_a = 1
dot_radius = 0.25
v0 = 0.04

quarter_plot = true

intersect = [0, 1]
step = [1, 0]

ris = LinRange(0, 1, W)
pis = LinRange(-1, 1, W)
if quarter_plot
    ris = LinRange(0, 0.5, W)
    pis = LinRange(0, 1, W)
end

le_count = zeros(Int64, W, W)
# Sum of lyapunov 
le_sum = zeros(W, W)
le_max = zeros(W, W)

# Shuffle the order of computation so that parallel trajectories
# are less likely to intersect
indices = collect(vec(CartesianIndices((1:W, 1:W))))
shuffle!(indices)
lk = ReentrantLock()
Threads.@threads for idx_i ∈ 1:length(indices)
    idx = indices[idx_i]
    r_idx = idx.I[1]
    p_idx = idx.I[2]
    maxp = max_momentum(pot, ris[r_idx] * intersect)
    if abs(pis[p_idx]) >= maxp
        continue
    end
    do_compute = false
    lock(lk)
    do_compute = le_count[r_idx, p_idx] == 0
    unlock(lk)
    if do_compute
        le, hits = lyapunov_exponent_and_hits(
            pot, ris, pis, r_idx, p_idx, intersect, step, dt)
        println("r=$(ris[r_idx]) p=$(pis[p_idx]) le=$le")
        # Update stats
        lock(lk)
        try
            for (rj, pj) ∈ hits
                le_max[rj, pj] = max(le_max[rj, pj], le)
                le_sum[rj, pj] += le
                le_count[rj, pj] += 1
            end
        finally
            unlock(lk)
        end
    end
end

le_mean = le_sum ./ le_count

## Write results

pot_dir = "$(potentials[1].name)_$(potentials[1].params["v0"])"
if potentials[1].name == "fermi_lattice"
    pot_dir = @sprintf("%s_%.2f_%.2f",
        pot_dir,
        potentials[1].params["dot_radius"],
        potentials[1].params["softness"])
end
dir = parsed_args["output_dir"]* pot_dir

mkpath(dir)

path ="$dir/lyapunov.h5"
h5open(path, "w") do f
    create_dataset(f, "le_mean", le_mean)
    create_dataset(f, "le_max", le_max)
    create_dataset(f, "le_count", le_count)
    create_dataset(f, "ris", ris)
    create_dataset(f, "pis", pis)
end
println("Wrote $path")

## 
# for i ∈ 1:W
#     Threads.@threads for j ∈ 1:W
#         r0p0 = get_r0_p0(pot, ys[i], pys[j])
#         if isnothing(r0p0)k
#             continue
#         end
#         r0,p0 = r0p0
#         le = local_lyapunov_exponent(pot, r0, p0, dt, T)
#         lle[i, j] = le
#         @printf "r0= (%.3f, %.3f) p0= (%.3f, %.3f) LE=%.3f \n" r0[1] r0[2] p0[1] p0[2] le
#     end
#     fig = heatmap(ys[1:i], pys, lle[1:i,:])
#     display(fig)
# end
# 
# heatmap(ys, pys, lle)