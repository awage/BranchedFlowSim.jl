using BranchedFlowSim
using Makie
using LinearAlgebra
using CairoMakie
using Colors
using FileIO
using ColorTypes
using ColorSchemes
using HDF5

function matrix_from_one_indices(size, one_indices)
    mat = falses(size...)
    for I ∈ eachcol(one_indices)
        mat[I...] = true
    end
    return mat
end

function normalize_momentum!(p, r, potential, E=0.5)
    @assert size(p) == size(r)
    @assert length(size(p)) <= 2

    if length(size(p)) == 2
        for i ∈ 1:size(p)[2]
            V = potential(r[1, i], r[2, i])
            pnorm = sqrt(2 * (E - V))
            @assert isreal(pnorm)
            p[:, i] .*= pnorm / norm(p[:, i])
        end
    elseif length(size(p)) == 1
        V = potential(r[1], r[2])
        pnorm = sqrt(2 * (E - V))
        @assert isreal(pnorm)
        p .*= pnorm / norm(p)
    end
end

function potential_heatmap(xs, ys, potential)
    V = grid_eval(xs, ys, potential)
    minV, maxV = extrema(V)
    pot_colormap = reverse(ColorSchemes.grays)
    return get(pot_colormap, 0.4 * (V .- minV) ./ (maxV - minV + 1e-9))
end

function color_trajectory!(img, traj, color)
    for (rk, pk) ∈ eachcol(traj)
        img[rk, pk] = color
    end
end

function potential_heatmap(xs, ys, potential)
    V = grid_eval(xs, ys, potential)
    minV, maxV = extrema(V)
    pot_colormap = reverse(ColorSchemes.grays)
    return get(pot_colormap, 0.4 * (V .- minV) ./ (maxV - minV + 1e-9))
end


data = load("outputs/classical/fermi_lattice_0.04_0.25_0.20_correlation/data.h5")

psos_rs = data["rs"]
psos_ps = data["ps"]
Nr = length(psos_rs)
Np = length(psos_ps)
threshold = 1.4
dt = data["dt"]

dim_max = data["dim_max"]
dim_count = data["dim_count"]

trajectory_dim = data["trajectory_dim"]
trajectory_start = data["trajectory_start"]
trajectory_plots = [
    data["trajectory_plot/$x"] for x ∈ 1:length(trajectory_dim)
]
trajectory_bitmaps = [
    matrix_from_one_indices(
        (length(psos_rs), length(psos_ps)), t)
    for t ∈ trajectory_plots
]
first_traj_idx = zeros(Int, Nr, Np)
for (i, plot) ∈ enumerate(trajectory_plots)
    if trajectory_dim[i] <= threshold
        for (rj, pj) ∈ eachcol(plot)
            if first_traj_idx[rj, pj] == 0
                first_traj_idx[rj, pj] = i
            end
        end
    end
end

step = [1.0, 0.0]
intersect = [0.0, 1.0]

## Find islands

island_color = zeros(Int, Nr, Np)

is_stable = dim_max .< threshold

next_color::Int64 = 1
color_sizes = Int64[]
println("starting loop")
for rj ∈ 1:Nr
    for pj ∈ 1:Np
        if island_color[rj, pj] != 0 || dim_max[rj, pj] > threshold || first_traj_idx[rj, pj] == 0
            continue
        end
        crj = rj
        cpj = pj
        island_bitmap = falses(Nr, Np)
        while true
            println("Next: $crj $cpj")
            plot = trajectory_plots[first_traj_idx[crj, cpj]]
            color_trajectory!(island_bitmap, plot, true)
            island_bitmap[crj, cpj] = true
            island_bitmap .&= is_stable

            down = circshift(island_bitmap, (1, 0))
            down[1, :] .= false
            up = circshift(island_bitmap, (-1, 0))
            up[end, :] .= false
            right = circshift(island_bitmap, (0, 1))
            right[:, 1] .= false
            left = circshift(island_bitmap, (0, -1))
            left[:, end] .= false
            next_candidate = (down .| up .| right .| left) .& (.~island_bitmap)
            # display(heatmap(psos_rs, psos_ps, next_candidate, axis=(title="next",)))

            found = false
            for idx ∈ findall(next_candidate)
                if dim_max[idx] <= threshold && first_traj_idx[idx] != 0
                    found = true
                    crj, cpj = Tuple(CartesianIndices(island_bitmap)[idx])
                    break
                end
            end
            if !found
                break
            end
        end
        global next_color
        for idx ∈ findall(island_bitmap)
            island_color[idx] = next_color
        end
        display(heatmap(psos_rs, psos_ps, island_bitmap, axis=(title="c=$next_color",)))
        next_color += 1
        push!(color_sizes, sum(island_bitmap))
    end
end

## Plotting

num_big = 22
colors = distinguishable_colors(num_big, [RGB(0, 0, 0), RGB(1, 1, 1)], dropseed=true)
# Only pick the largest islands
sp = reverse(sortperm(color_sizes))

fig = Figure(resolution=(1024, 512))
ax = Axis(fig[1, 2])
img = zeros(typeof(colors[1]), Nr, Np)
for i ∈ 1:num_big
    for idx ∈ eachindex(island_color)
        if island_color[idx] == sp[i]
            img[idx] = colors[i]
        end
    end
end
image!(ax, psos_rs, psos_ps, img)

pot = potential_from_h5_data(data)

num_rays = 400
W = 80
real_ax = Axis(fig[1, 1], limits=((0,W), (0, W)), aspect=DataAspect())

bg_xs = LinRange(0, W, 1024)

pot_hm = potential_heatmap(bg_xs, bg_xs, pot)
image!(real_ax, bg_xs, bg_xs, pot_hm)
display(fig)

r0 = [0.53, 0.22]
for θ ∈ LinRange(0, pi/2, num_rays + 1)[1:end-1]
    # Find color for this ray
    p = [sin(θ), cos(θ)]
    normalize_momentum!(p, r0, pot)
    ts, rs, ps = ray_trajectory(r0, p, pot, 2W, dt, 0.05)

    color = RGB(0,0,0)
    if abs(rs[1,end]) > 1
        @show θ,p
        m = PoincareMapper(pot, r0, p, intersect, step, [0, 0], dt)
        inter = next_intersection!(m)
        @show inter
        r_int = inter.r_int
        if r_int > 0.5
            r_int = 1.0 - r_int
        end
        p_int = abs(inter.p_int)
        rj = round(Int, 1 + (r_int - psos_rs[1]) / (psos_rs[2] - psos_rs[1]))
        pj = round(Int, 1 + (p_int - psos_ps[1]) / (psos_ps[2] - psos_ps[1]))
        color = img[rj, pj]

        if color != RGB(0,0,0)
            lines!(real_ax, rs, color=RGBA(color, 0.3))
        end
    end
end

display(fig)
save("outputs/classical/islands.png", fig, px_per_unit=2)
save("outputs/classical/islands.pdf", fig)