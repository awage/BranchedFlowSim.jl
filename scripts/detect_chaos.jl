using DifferentialEquations
using BranchedFlowSim
using ColorTypes
using LaTeXStrings
using FileIO
using Serialization
using ArgParse
using Printf
using CurveFit
using CairoMakie
using Makie
using StaticArrays
using LinearAlgebra
using HDF5
using ColorSchemes

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
    default_potentials="cos_series",
    defaults=Dict(
        "cos_degrees" => "3",
        "num_angles" => 1,
        "num_sims" => 1,
        "lattice_a" => 1.0,
        "fermi_dot_radius" => 0.25,
        "fermi_softness" => 0.20,
    )
)

function count_nonzero(img, scale)::Int
    w = size(img)[1] ÷ scale
    h = size(img)[2] ÷ scale
    c::Int = 0
    for x ∈ 0:w-1
        for y ∈ 0:h-1
            if !iszero(@view img[(1+scale*x):(scale*(x+1)),
                (1+scale*y):(scale*(y+1))])
                c += 1
            end
        end
    end
    return c
end

function one_indices(bm::AbstractMatrix{Bool})
    num_ones = sum(bm)
    indices = zeros(Int32, 2, num_ones)
    for (k, I) ∈ enumerate(findall(bm))
        indices[:, k] .= Tuple(I)
    end
    return indices
end

function matrix_from_one_indices(size, one_indices)
    mat = falses(size...)
    for I ∈ eachcol(one_indices)
        mat[I...] = true
    end
    return mat
end

function fractal_dimension(section, debug=false)::Float64
    max_scale = 2
    w = size(section)[1]

    scales = 1:max_scale
    grid_width = [w / s for s ∈ scales]
    num_cells = [count_nonzero(section, s) for s ∈ scales]
    a, d = power_fit(grid_width, num_cells)

#     a2, d2 = power_fit(grid_width[2:end], num_cells[2:end])
#     if d2 < d
#         d = d2
#         a = a2
#     end

    if debug
        @show grid_width, num_cells
        fig = Figure()
        ax = Axis(fig[1, 1])
        scatter!(ax, grid_width, num_cells)
        xs = LinRange(0, w, 100)
        lines!(ax, xs, a * xs .^ d)
        display(fig)
        @show a, d
    end
    return d
end

function nearest_index(range, x)
    dx = range[2] - range[1]
    return round(Int, 1 + (x - range[1]) / dx)
end

function section_and_dim(mapper, ris, pis)
    Nr = length(ris)
    Np = length(pis)
    hit = fill(false, Nr, Np)
    # 
    Nr2 = 3 * Nr ÷ 4
    Np2 = 3 * Np ÷ 4
    ris2 = LinRange(ris[begin], ris[end], Nr2)
    pis2 = LinRange(pis[begin], pis[end], Np2)
    hit2 = fill(false, Nr2, Np2)

    nohit_count = 0
    hit_count = 0
    for z ∈ 1:(4*Nr*Np)
        int = next_intersection!(mapper)
        # Map to distance from midline
        r = int.r_int
        if r > 0.5
            r = 1.0 - r
        end
        p = abs(int.p_int)
        rint = nearest_index(ris, r)
        pint = nearest_index(pis, p)
        if 1 <= rint <= Nr && 1 <= pint <= Np
            if hit[rint, pint]
                nohit_count += 1
                if nohit_count > 4*sqrt(Np * Nr)
                    println("stop at $z")
                    break
                end
            else
                nohit_count = 0
                hit_count += 1
                hit[rint, pint] = true
            end
        end
        rint2 = nearest_index(ris2, r)
        pint2 = nearest_index(pis2, p)
        if 1 <= rint2 <= Nr2 && 1 <= pint2 <= Np2
            hit2[rint2, pint2] = true
        end
    end
    # 
    # f(N)/f(N₂) = (N/N₂)^d
    n_ratio = sqrt(Nr*Np/(Nr2*Np2))
    d = log(n_ratio, sum(hit)/sum(hit2))

    return hit,d
end

function section_and_correlation_dim(mapper, ris, pis)
    Nr = length(ris)
    Np = length(pis)
    hit = fill(false, Nr, Np)

    N = 20000
    points = zeros(2, N)
    for i ∈ 1:N
        int = next_intersection!(mapper)
        # Map to distance from midline
        r = int.r_int
        if r > 0.5
            r = 1.0 - r
        end
        p = abs(int.p_int)
        rint = nearest_index(ris, r)
        pint = nearest_index(pis, p)
        if 1 <= rint <= Nr && 1 <= pint <= Np
            hit[rint, pint] = true
        end
        points[1, i] = r
        points[2, i] = p
    end
    
    d = correlation_dimension(points, 0.01)
    return hit,d
end

#=
function section_and_dim(mapper, ris, pis)
    hit = get_section_hits(mapper, ris, pis)
    dim = fractal_dimension(hit
    if dim > 1.1 && sum(hit) < 2 * (length(ris) + length(pis))
        # This is likely a stable orbit.
        # To get more accurate dimension, compute
        # using increased resolution.
        ris2 = LinRange(ris[begin], ris[end], 2 * length(ris))
        pis2 = LinRange(pis[begin], pis[end], 2 * length(pis))
        println("Likely stable, computing with double res")
        hit_for_dim = get_section_hits(mapper, ris2, pis2)
        dim2 = fractal_dimension(hit_for_dim)
        println("dim: $dim => $dim2")
        dim = dim2
    end
    return hit, dim
end
=#

parsed_args = parse_args(ARGS, s)

potentials = get_potentials_from_parsed_args(parsed_args, 1, 1)
if length(potentials) != 1
    println("potentials=$potentials")
    error("Exactly one potential is required, got $(length(potentials))")
end
pot = potentials[1].instances[1]
pot_name = potentials[1].name

Q = parsed_args["grid_size"]
dt = parsed_args["dt"]

section = fill(RGB(1, 1, 1), Q, Q)
# r0 = SA[0.0, 0.18]
# # r0 = SA[0.0, 0.13]
# p0 = SA[1.0, 0.0]

intersect = [0.0, 1.0]
step = [1.0, 0.0]
offset = [0.0, 0.0]

ris = LinRange(0, 0.5, Q)
pis = LinRange(0, 1, Q)

dim_tot = zeros(Q, Q)
dim_min = fill(10.0, Q, Q)
dim_max = fill(0.0, Q, Q)
dim_count = zeros(Q, Q)

progress_plot = isinteractive()

hit_maps = BitMatrix[]
hit_dims = Float64[]
traj_start = Vector{Float64}[]
## 
for (i, r) ∈ enumerate(ris)
    maxp = max_momentum(pot, r * intersect)
    for (j, p) ∈ enumerate(pis)
        if p >= maxp
            continue
        end
        if dim_count[i, j] == 0
            # Not yet covered, shoot a ray
            println("Shoot ($r,$p)")
            mapper = PoincareMapper(
                pot,
                r, p,
                intersect,
                step,
                offset,
                dt
            )
            # @time "Poincaré sim" hit, dim = section_and_correlation_dim(mapper, ris, pis)
            @time "Poincaré sim" hit, dim = section_and_dim(mapper, ris, pis)
            push!(hit_maps, BitMatrix(hit))
            push!(hit_dims, dim)
            push!(traj_start, [r, p])
            @printf "dim=%.2f\n" dim
            for idx ∈ eachindex(hit)
                if hit[idx]
                    dim_tot[idx] += dim
                    dim_min[idx] = min(dim_min[idx], dim)
                    dim_max[idx] = max(dim_max[idx], dim)
                    dim_count[idx] += 1
                end
            end
            mean_dim = dim_tot ./ dim_count
            if progress_plot
                fig = Figure()
                ax = Axis(fig[1, 1], title="$r, $p, dim=$dim")
                hm = heatmap!(ax, ris, pis, mean_dim,
                    colorrange=(1.0, 2.0))
                # contour!(ax, ris, pis, hit, color=:red)
                image!(ax, ris, pis, hit,
                    colormap=[
                        RGBA(0.0, 0.0, 0.0, 0.0),
                        RGBA(1.0, 0.0, 0.0, 0.5)
                    ])
                scatter!(ax, r, p, marker=:xcross, markersize=15, color=:red)
                Colorbar(fig[1, 2], hm)
                display(fig)
            end
            # display(hit)
        end
    end
end

## Write outputs

dir = parsed_args["output_dir"]
if dir[end] != '/'
    push!(dir, '/')
end

pot_dir = "$(potentials[1].name)_$(potentials[1].params["v0"])"
if potentials[1].name == "fermi_lattice"
    pot_dir = @sprintf("%s_%.2f_%.2f",
        pot_dir,
        potentials[1].params["dot_radius"],
        potentials[1].params["softness"])
end
dir = dir * pot_dir

dim_mean = dim_tot ./ dim_count

mkpath(dir)

h5open("$dir/data.h5", "w") do f
    f["grid_size"] = Q
    f["dt"] = Float64(dt)
    f["rs"] = Vector{Float64}(ris)
    f["ps"] = Vector{Float64}(pis)
    f["dim_mean"] = dim_mean
    f["dim_max"] = dim_max
    f["dim_min"] = dim_min
    f["dim_count"] = dim_count
    f["trajectory_dim"] = hit_dims
    f["trajectory_start"] = hcat(traj_start...)
    pot = create_group(f, "potential")
    for (k, v) ∈ pairs(potentials[1].params)
        pot[k] = v
    end
    sections = create_group(f, "trajectory_plot")
    for (k,map) ∈ enumerate(hit_maps)
        sections["$k"] = one_indices(map)
    end
end

# @time "Poincaré sim" ris,pis,hit = get_section_hits(mapper, Q)
# for i ∈ eachindex(section)
#     if hit[i] != 0
#         section[i] = RGB(0,0,0)
#     end
# end
# # hit_count::Int64 = 0
# println("Fractal dimension: $(fractal_dimension(hit))")
# # heatmap(ris, pis, section)
# display(section)
# image(ris, pis, section)
# save("outputs/classical/chaos.png", section')
# 