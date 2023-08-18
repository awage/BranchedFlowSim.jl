using BranchedFlowSim
using Makie
using Printf
using StaticArrays
using LinearAlgebra
using CairoMakie

function get_r0_p0(pot, y, py)
    r0 = [0, y]
    V = pot(r0[1], r0[2])
    if (py^2/2 + V >= 0.5) 
        return nothing
    end
    px = sqrt(2*(0.5 - V) - py^2)
    p0 = [px, py]
    return r0, p0
end

function nearest_index(range, x)
    dx = range[2] - range[1]
    return round(Int, 1 + (x - range[1]) / dx)
end

function lyapunov_section(pot, ris, pis, r_idx, p_idx, intersect, step, dt)
    δ_norm = 1e-6
    δrp = δ_norm * normalize(rand(2) .- 0.5)

    mapper = PoincareMapper(pot, ris[r_idx], pis[p_idx], intersect, step, [0,0], dt)
    mapper_var = PoincareMapper(pot, ris[r_idx]+δrp[1], pis[p_idx]+δrp[2],
        intersect, step, [0,0], dt)
    N = length(ris)
    M = length(pis)
    # Time per simulation
    num_hits = 20000
    hit_count = zeros(Int64, N,M)
   #  update_order = SVector{2, Int64}[SA[r_idx, p_idx]]
    log_sum = 0.0
    
    last_time = 0.0
    hit_count[r_idx, p_idx] += 1
    for _ ∈ 1:num_hits
        int = next_intersection!(mapper)
        int_var = next_intersection!(mapper_var)
        r_idx = nearest_index(ris, int.r_int)
        p_idx = nearest_index(pis, int.p_int)
        @assert 1 <= r_idx <= N
        @assert 1 <= p_idx <= M
        hit_count[r_idx, p_idx] += 1

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
    le = (1/last_time) * log_sum
end

lattice_a = 1
dot_radius = 0.25
v0 = 0.04

pot = LatticePotential(lattice_a * I, dot_radius, v0)

dt = 0.005
T = 500
W = 512

ys = LinRange(0, 0.5, W)
pys = LinRange(0, 1, W)
lle = zeros(W,W)

## 
for i ∈ 1:W
    Threads.@threads for j ∈ 1:W
        r0p0 = get_r0_p0(pot, ys[i], pys[j])
        if isnothing(r0p0)
            continue
        end
        r0,p0 = r0p0
        le = local_lyapunov_exponent(pot, r0, p0, dt, T)
        lle[i, j] = le
        @printf "r0= (%.3f, %.3f) p0= (%.3f, %.3f) LE=%.3f \n" r0[1] r0[2] p0[1] p0[2] le
    end
    fig = heatmap(ys[1:i], pys, lle[1:i,:])
    display(fig)
end

heatmap(ys, pys, lle)