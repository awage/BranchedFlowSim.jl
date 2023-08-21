#=
Figures to explain the branch count metric
=#

using BranchedFlowSim
using ColorSchemes
using LaTeXStrings
using ColorTypes
using Makie
using CairoMakie

function heatmap_with_potential(pot, xs, ys, data)
    pot_colormap = ColorSchemes.Greens
    data_colormap = ColorSchemes.grayC

    V = grid_eval(xs, ys, pot)

    minV, maxV = extrema(V)
    minD, maxD = extrema(data)
    print("datarange: [$(minD),$(maxD)]\n")
    pot_img = get(pot_colormap, 0.4 * (V .- minV) ./ (maxV - minV + 1e-9))
    data_img = get(data_colormap, (data .- minD) / (maxD - minD))

    img = mapc.((d, v) -> clamp(d * v, 0, 1), data_img, pot_img)
    return img
end

W = 2
v0 = 0.04
num_rays = 6000
dt = 0.001
pot = correlated_random_potential(2, 2, 0.1, v0, 2)

W = 512
xs = LinRange(0, 1, W)
ys = LinRange(0, 1, W)

int, (rmin, rmax) = quasi2d_intensity(num_rays, dt, xs, ys, pot)

# Compute rays at some fixed x
cross_idx = 2 * W ÷ 3
nb, ray_y = quasi2d_num_branches(ys, dt, [0, cross_x], pot, return_rays=true)

## Do plotting

fig = Figure(resolution=(800, 800))
ax = Axis(fig[1, 1], aspect=DataAspect(),
    limits=((xs[begin], xs[end]), (ys[begin], ys[end])),
    ylabel=L"y",
    xlabel=L"x=t",
    xticksvisible=false,
    yticksvisible=false,
    xticklabelsvisible=false,
    yticklabelsvisible=false,
    xgridvisible=false,
    ygridvisible=false,
)


img = heatmap_with_potential(pot, xs, ys, int)

# heatmap!(ax, xs, ys, int')
image!(ax, xs, ys, img')

cross_x = xs[cross_idx]
vlines!(ax, [cross_x], color=:red, label=L"t_1")

ax_int = Axis(fig[1, 2], aspect=1, limits=(nothing, (ys[begin], ys[end])),
    xlabel=L"\rho(t_1, y)",
    xgridvisible=false,
    ygridvisible=false,
)
hideydecorations!(ax_int)

lines!(ax_int, int[:, cross_idx], ys)

ax_ray = Axis(fig[2, 1],
    aspect=1,
    limits=((ys[begin], ys[end]), (ys[begin], ys[end])),
    ylabel=L"y(t_1)",
    xlabel=L"y(0)",
    xticksvisible=false,
    yticksvisible=false,
    xticklabelsvisible=false,
    yticklabelsvisible=false,
    xgridvisible=false,
    ygridvisible=false,
)
lines!(ax_ray, ys, ray_y)

ax_diff = Axis(fig[2, 2],
    aspect=1,
    limits=((ys[begin], ys[end]), nothing),
    xlabel=L"y(0)",
    ylabel=L"\partial y(t) / \partial y(0)",
    xticksvisible=false,
    xticklabelsvisible=false,
    xgridvisible=false,
    ygridvisible=false,
)
# hideydecorations!(ax_diff)
# Not a good differential, but good enough for an illustration
dray = (ray_y[2:end] - ray_y[1:end-1]) / (ys[2] - ys[1])
lines!(ax_diff, ys[1:end-1], dray)
hlines!(ax_diff, [0.0], color=:lightgray)
linkyaxes!(ax, ax_int)

# Mark caustics
caustic_y0 = Float64[]
caustic_yt = Float64[]
for j ∈ 2:length(dray)
    if xor(dray[j] < 0, dray[j-1] < 0)
        push!(caustic_y0, ys[j])
        push!(caustic_yt, ray_y[j])
    end
end

scatter!(ax, fill(cross_x, length(caustic_yt)), caustic_yt, color=:red,
    marker=Circle,
    markersize=6)
scatter!(ax_diff, caustic_y0, fill(0, length(caustic_y0)), color=:red, marker=Circle,
    markersize=6)
scatter!(ax_ray, caustic_y0, caustic_yt, color=:red, marker=Circle,
    markersize=6)
    
hlines!(ax_int, caustic_yt, color=:red, 
linestyle=:dot
)

display(fig)

save("outputs/quasi2d/branch_explain.png", fig, px_per_unit=2)
save("outputs/quasi2d/branch_explain.pdf", fig)

