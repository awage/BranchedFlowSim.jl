# Temporary script to play around with classical simulations
# Move to src/ once more mature

using BranchedFlowSim
using CairoMakie
using ColorSchemes
using ColorTypes
using LinearAlgebra
using Makie

function normalize_momentum!(p, r, potential, E=0.5)
    @assert size(p) == size(r)
    for i ∈ 1:size(p)[2]
        V = potential(r[1,i], r[2,i])
        pnorm = sqrt(2 * (E - V))
        @assert isreal(pnorm)
        p[:,i] .*= pnorm / norm(p[:,i])
    end
end

lattice_a::Float64 = 0.2
dot_radius::Float64 = 0.25 * lattice_a
v0::Float64 = 0.1
softness = 0.1
dt::Float64 = 0.005

pot = LatticePotential(lattice_a * I, dot_radius, v0;
    softness=softness, offset=[lattice_a/2, lattice_a/2])
# pot = correlated_random_potential(4,4,0.1,v0)
# pot= fermi_dot_lattice_cos_series(1, lattice_a, dot_radius, -v0)

num_particles = 10000
# num_particles = 4
r0 = lattice_a * [0.1, 0.05] * ones(num_particles)'
angles = LinRange(0, 2pi, num_particles + 1)[1:end-1]
p0 = hcat(([cos(θ), sin(θ)] for θ ∈ angles)...)
    
normalize_momentum!(p0, r0, pot)

T = 5.0

## Plotting

function pixel_heatmap(path, data; kwargs...)
    data = transpose(data)
    scene = Scene(camera=campixel!, resolution=size(data))
    heatmap!(scene, data; kwargs...)
    save(path, scene)
    return scene
end

function heatmap_with_potential(path, data, potential; colorrange=extrema(data))
    fire = reverse(ColorSchemes.linear_kryw_0_100_c71_n256)
    pot_colormap = ColorSchemes.grays
    # data_colormap = ColorSchemes.viridis
    data_colormap = fire

    V = real(potential)
    minV, maxV = extrema(V)
    minD, maxD = colorrange
    print("datarange: [$(minD),$(maxD)]\n")
    pot_img = get(pot_colormap, (V .- minV) ./ (maxV - minV + 1e-9))
    data_img = get(data_colormap, (data .- minD) / (maxD - minD))
    
    img = mapc.((d,v) -> clamp(d - 0.2 * v, 0, 1), data_img, pot_img) 
    save(path, img)
    return img
end

Nh = 512
xs = LinRange(-2, 2, Nh)
hist :: Matrix{Float64} = zeros(Nh, Nh)

function nearest_idx(xs, x)
    dx = xs[2] - xs[1]
    return 1 + round(Int64, (x - xs[1]) / dx)
end

dx = xs[2] - xs[1]

@time "histo" ray_trajectories(r0, p0, pot, T, dt) do ts, rs, ps
    lx::Float64 = -1
    ly::Float64 = -1
    px::Int = 0
    py::Int = 0
    N = length(ts)
    for (x, y) ∈ eachcol(rs)
        if false
            xi = round(Int, 1 + (x - xs[1]) / dx)
            yi = round(Int, 1 + (y - xs[1]) / dx)
            if 1 < xi < Nh && 1 < yi < Nh && (xi != px || yi != py)
                hist[yi, xi] += 1
                px = xi
                py = yi
            end
        else
            xi = 1 + (x - xs[1]) / dx
            yi = 1 + (y - xs[1]) / dx
            if 1 < xi < Nh && 1 < yi < Nh && 1 < lx < Nh && 1 < ly < Nh
                # img[round(Int, yi), round(Int, xi)] += 2
                # plot_line(lx, ly, xi, yi) do x,y
                line_integer_cells(lx, ly, xi, yi) do x, y
                    if x != px || y != py
                        hist[y, x] += 1
                    end
                    px = x
                    py = y
                end
            end
            lx = xi
            ly = yi
        end
    end
end

## Visualize histogram

fire = reverse(ColorSchemes.linear_kryw_0_100_c71_n256)
# fig = pixel_heatmap(
#     "outputs/classical/dot.png", Makie.pseudolog10.(img);
#     colorrange=(0, Makie.pseudolog10(2 * num_particles)),
#     colormap=fire)
# display(fig)

img = heatmap_with_potential(
    "outputs/classical/dot.png",
    hist, 
    grid_eval(xs, xs, pot),
    colorrange=(0.0, 2.0 * num_particles / 100)
)
image(xs, xs, img, axis=(aspect=DataAspect(),))
# save("outputs/classical/dot.png", current_figure())
#fig = Figure()
#ax = Axis(fig[1,1])
#display(fig)
