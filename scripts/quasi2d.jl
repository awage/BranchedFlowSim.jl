using BranchedFlowSim
using CairoMakie
using Interpolations
using Makie
using ColorTypes


function pixel_heatmap(path, data; kwargs...)
    scene = Scene(camera=campixel!, resolution=size(data))
    heatmap!(scene, data; kwargs...)
    save(path, scene)
end

function count_zero_crossing(fx)
    c = 0
    for j ∈ 2:length(fx)
        if sign(fx[j-1]) != sign(fx[j])
        c += 1
        end
    end
    return c
end

function simulate_num_branches(xs, ys, potential)
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    dt = dx
    # Compute F(x,y) = -∂V(x,y)/∂y
    force = (circshift(potential, [-1, 0]) .- circshift(potential, [1, 0])) ./ (2 * dy)
    force_itp = interpolate(force, BSpline(Linear()))
    force_itp = scale(force_itp, ys, xs)
    force_itp = extrapolate(force_itp, Periodic())
    ray_y = Vector(ys)
    ray_py = zeros(num_rays)
    num_branches = zeros(length(xs))
    m = ray_y[2:end] - ray_y[1:end-1]
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* force_itp(ray_y, x)
        # drift
        ray_y .+= dt .* ray_py
        # Count branches
        caustics = count_zero_crossing(ray_y[2:end] - ray_y[1:end-1])
        num_branches[xi] = caustics / 2
    end
    return (num_branches, m)
end

sim_height = 1
sim_width = 2
num_rays = 512
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0 = ϵ_percent *0.01  * 0.5

num_sims = 50

Ny = num_rays
Nx = round(Int, num_rays * (sim_width / sim_height))
ys = LinRange(0, sim_height, Ny + 1)[1:end-1]
xs = LinRange(0, sim_width, Nx + 1)[1:end-1]

# potential = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
# nbz,m = simulate_num_branches(xs, ys, potential)
# lines(ys[1:end-1], m)

num_branches = zeros(length(xs), num_sims)
Threads.@threads for i ∈ 1:num_sims
    potential = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
    num_branches[:, i],_ = simulate_num_branches(xs, ys, potential)
end
nb = vec(sum(num_branches, dims=2)) ./ (num_sims * sim_height)

lines(xs, nb)
current_figure()