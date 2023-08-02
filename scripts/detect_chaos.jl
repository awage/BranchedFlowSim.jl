using DifferentialEquations
using BranchedFlowSim
using ColorTypes
using ArgParse
using CurveFit
using CairoMakie
using Makie
using StaticArrays
using LinearAlgebra
using ColorSchemes

function count_nonzero(img, scale) :: Int
    w = size(img)[1] ÷ scale
    h = size(img)[2] ÷ scale
    c::Int = 0
    for x ∈ 0:w-1
        for y ∈ 0:h-1
            if !iszero(@view img[(1 + scale*x):(scale*(x+1)),
                (1 + scale*y):(scale*(y+1))])
                c += 1
            end
        end
    end
    return c
end

function fractal_dimension(section) :: Float64
    max_scale = 4
    w = size(section)[1]
    grid_width = [w / s for s ∈ 1:max_scale]
    num_cells = [count_nonzero(section, s) for s ∈ 1:max_scale]
    a, d = power_fit(grid_width, num_cells)
    #=
    @show grid_width,num_cells
    fig = Figure()
    ax = Axis(fig[1,1])
    scatter!(ax, grid_width, num_cells)
    xs = LinRange(0, w, 100)
    lines!(ax, xs, a * xs.^d)
    display(fig)
    @show a,d
    =#
    return d
end

function get_section_hits(mapper, ris, pis)
    dr = ris[2] - ris[1]
    dp = pis[2] - pis[1]
    Nr = length(ris)
    Np = length(pis)
    hit = fill(false, Nr, Np)
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
        rint = round(Int, 1 + (r - ris[1]) / dr)
        pint = round(Int, 1 + (p - pis[1]) / dp)
        if 1 <= rint <= Nr && 1 <= pint <= Np
            if hit[rint, pint]
                nohit_count += 1
                if nohit_count > sqrt(Np*Nr)
                    println("stop at $z")
                    break
                end
            else
                nohit_count = 0
                hit_count += 1
                hit[rint, pint] = true
            end
            # if last_hit[rint, pint] == 0
            #     hit_count += 1
            # elseif last_hit[rint,pint] == hit_count && z > (Q*Q)
            #     println("Stop at z=$z")
            #     break
            # end
           #  hit[rint, pint] = true
        end
    end
    return hit
end

s = ArgParseSettings()
@add_arg_table s begin
    "--grid_size"
    help = "width and height of the grid used for the Poincaré section plot"
    arg_type = Int
    default = 600
    "--min_time"
    help = "end time"
    arg_type = Float64
    default = 500.0
    "--dt"
    help = "time step of integration"
    arg_type = Float64
    default = 0.005
end

add_potential_args(s,
    default_potentials="fermi_lattice",
    defaults=Dict(
        "num_angles" => 1,
        "num_sims" => 1,
        "lattice_a" => 1.0,
        "fermi_dot_radius" => 0.25,
    )
)

parsed_args = parse_args(ARGS, s)

potentials = get_potentials_from_parsed_args(parsed_args, 1, 1)
pot = potentials[1].instances[1]

Q = parsed_args["grid_size"]
T = parsed_args["min_time"]
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

dim_tot = zeros(Q,Q)
dim_min = fill(10.0, Q, Q)
dim_max = fill(0.0, Q, Q)
dim_count = zeros(Q,Q)

progress_plot = false

## 
for (i, r) ∈ enumerate(ris)
    maxp = max_momentum(pot, r * intersect)
    for (j,p) ∈ enumerate(pis)
        if p >= maxp
            continue
        end
        if dim_count[i,j] == 0
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
            @time "Poincaré sim" hit = get_section_hits(mapper, ris,pis)
            dim = fractal_dimension(hit)
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
                ax = Axis(fig[1,1], title="$r, $p")
                hm = heatmap!(ax, ris, pis, mean_dim,
                    colorrange=(0.0, 2.0))
                contour!(ax, ris, pis, hit, color=:red)
                scatter!(ax, r, p, marker='+', color=:blue)
                Colorbar(fig[1,2], hm)
                display(fig)
            end
            # display(hit)
        end
    end
end

##

dim_mean = dim_tot ./ dim_count
fig = Figure()
ax = Axis(fig[1,1], xticks=0.5:0.1:1.0)
hm = heatmap!(ax, ris, pis, dim_min, colorrange=(1.0, 2.0),
    colormap=ColorSchemes.viridis)
cm = Colorbar(fig[1,2], hm)
# display(fig)

save("outputs/classical/kam.png", fig, px_per_unit=2)


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