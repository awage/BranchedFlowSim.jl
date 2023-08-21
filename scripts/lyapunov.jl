using BranchedFlowSim
using Makie
using Printf
using Random
using StaticArrays
using LinearAlgebra
using CairoMakie
using HDF5

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

#=
function lyapunov_exponent_and_hits(pot, ris, pis, r_idx, p_idx, intersect, step, dt)
    δ_norm = 1e-5
    δrp = δ_norm * normalize(rand(2) .- 0.5)

    mapper = PoincareMapper(pot, ris[r_idx], pis[p_idx], intersect, step, [0,0], dt)
    mapper_var = PoincareMapper(pot, ris[r_idx]+δrp[1], pis[p_idx]+δrp[2],
        intersect, step, [0,0], dt)
    N = length(ris)
    M = length(pis)
    # Time per simulation
    num_hits = 2000
    log_sum = 0.0
    last_time = 0.0
    hits = SVector{2, Int64}[SA[r_idx, p_idx]]
    orig_r = ris[r_idx]
    orig_p = pis[p_idx]
    for _ ∈ 1:num_hits
        int = next_intersection!(mapper)
        int_var = next_intersection!(mapper_var)
        if isnothing(int) || isnothing(int_var)
            # No intersection was found in reasonable time.
            println("Bad trajectory: r=$orig_r p=$orig_p")
            return 0.0, []
        end
        r_idx = nearest_index(ris, int.r_int)
        p_idx = nearest_index(pis, int.p_int)
        @assert 1 <= r_idx <= N
        @assert 1 <= p_idx <= M
        push!(hits, SA[r_idx, p_idx])
        δr = int_var.r_int - int.r_int
        δp = int_var.p_int - int.p_int
        dist = sqrt(δr^2 + δp^2)
        log_sum += log(dist / δ_norm)
        # Normalize back to δ_norm
        δr = δ_norm * δr / dist
        δp = δ_norm * δp / dist
        set_position_on_surface!(mapper_var, int.r_int + δr, int.p_int + δp)
        last_time = int.t
    end
    le = (1/num_hits) * log_sum
    return le, hits  
end
=#
function lyapunov_exponent_and_hits(pot, ris, pis, r_idx, p_idx, intersect, step, dt)
    δ_norm = 1e-5
    δrp = δ_norm * normalize(rand(2) .- 0.5)

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
        if int.t > T / 2
            break
        end
        r_idx = nearest_index(ris, int.r_int)
        p_idx = nearest_index(pis, int.p_int)
        @assert 1 <= r_idx <= N
        @assert 1 <= p_idx <= M
        push!(hits, SA[r_idx, p_idx])
    end
    return le, hits
end

lattice_a = 1
dot_radius = 0.25
v0 = 0.04

pot = LatticePotential(lattice_a * I, dot_radius, v0)

dt = 0.005
T = 500
W = 512
intersect = [0, 1]
step = [1, 0]

ris = LinRange(0, 1, W)
pis = LinRange(-1, 1, W)

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
        # Update stats
        lock(lk)
        try
            for (rj, pj) ∈ hits
                le_max[rj, pj] = max(le_max[rj, pj], le)
                le_sum[rj, pj] += le
                le_count[rj, pj] += 1
            end
            println("r=$(ris[r_idx]) p=$(pis[p_idx]) le=$le")
            le_mean = le_sum ./ le_count
            fig = heatmap(ris, pis, le_mean)
            display(fig)
        finally
            unlock(lk)
        end
    end
end

le_mean = le_sum ./ le_count

path ="outputs/classical/lyapunov.h5"
h5open(path) do f
    create_dataset(f, "le_mean", le_mean)
    create_dataset(f, "le_max", le_max)
    create_dataset(f, "le_count", le_count)
    create_dataset(f, "ris", ris)
    create_dataset(f, "ris", pis)
end

## 
# for i ∈ 1:W
#     Threads.@threads for j ∈ 1:W
#         r0p0 = get_r0_p0(pot, ys[i], pys[j])
#         if isnothing(r0p0)
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