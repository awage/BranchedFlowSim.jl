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


sim_height = 10
sim_width = 20
num_rays = 512
correlation_scale = 1
v0 = 0.04

num_simulations = 100

Ny = num_rays
Nx = num_rays * (sim_width ÷ sim_height)
ys = LinRange(0, sim_height, Ny + 1)[1:end-1]
xs = LinRange(0, sim_width, Nx + 1)[1:end-1]

dx = xs[2] - xs[1]
dy = ys[2] - ys[1]
dt = dx

function make_grid_potential(num_rows)
    pot = zeros(length(ys), length(xs))
    onedot = zeros(length(ys), length(ys))
    # θ = deg2rad(5)
    θ = atan(1/num_rows)
    print("θ=$(rad2deg(θ))°\n")
    lattice_constant = sim_height / sqrt(1 + num_rows^2)
    lattice_matrix = lattice_constant * [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]
    radius = lattice_constant / 4
    points = lattice_points_in_a_box(
        lattice_matrix,
        -3 * radius, sim_width + 3 * radius,
        -3 * radius, sim_height + 3 * radius
    )
    for p ∈ eachcol(points)
        add_fermi_dot!(pot, xs, ys, p, radius)
    end
    return pot
end

function make_triangle_potential()
    pot = zeros(length(ys), length(xs))
    lattice_constant = sim_height / (2 * sin(π/3))
    lattice_matrix = lattice_constant * [
        1 cos(π / 3)
        0 sin(π / 3)
    ]
    radius = lattice_constant / 4
    points = lattice_points_in_a_box(
        lattice_matrix,
        -3 * radius, sim_width + 3 * radius,
        -3 * radius, sim_height + 3 * radius
    )
    for p ∈ eachcol(points)
        add_fermi_dot!(pot, xs, ys, p, radius)
    end
    return pot
end


potential = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
# @time "made potential" potential = v0 * make_angled_grid_potential(xs, ys, 0)
# @time "made potential" potential = v0 * make_triangle_potential()

# fig = Figure(resolution=(2*Nx,2*Ny))
# ax = Axis(fig[1,1])
# heatmap!(ax, xs, ys, potential')
# fig

# Compute F(x,y) = -∂V(x,y)/∂y
# 
force = (circshift(potential, [-1, 0]) .- circshift(potential, [1, 0])) ./ (2 * dy)
# fig = Figure(resolution=(2*Nx,2*Ny))
# ax = Axis(fig[1,1])
# heatmap!(ax, xs, ys, force')
# fig
#
force_itp = interpolate(force, BSpline(Linear()))
force_itp = scale(force_itp, ys, xs)
force_itp = extrapolate(force_itp, Periodic())

ray_y = Vector(ys)
# ray_y = rand(length(ys)) * sim_height
ray_py = zeros(num_rays)
# ray_py = rand(num_rays) .- 0.5

image = zero(potential)
num_branches = zeros(length(xs))
for (xi, x) ∈ enumerate(xs)
    # kick
    ray_py .+= dt .* force_itp(ray_y, x)
    # drift
    ray_y .+= dt .* ray_py
    # Collect
    for y ∈ ray_y
        yi = 1 + ((100Ny + round(Int, (y / sim_height) * Ny)) % Ny)
        image[yi, xi] += 1
    end
    # Wrap
    # ray_y .+= sim_height
    # ray_y .%= sim_height
    
    # Count branches
    caustics = 0
    for j ∈ 2:(num_rays-1)
        if sign(ray_y[j] - ray_y[j-1]) != sign(ray_y[j+1] - ray_y[j])
            caustics += 1
        end
    end
    num_branches[xi] = caustics / 2
end

# pixel_heatmap("outputs/kaplan_potential.png", potential', colormap=:Greens)
# pixel_heatmap("outputs/kaplan_rays.png", image', colormap=:grayC)

scene = Scene(camera=campixel!, resolution=size(potential'))
heatmap!(scene, potential'; colormap=:Greens_3)
heatmap!(scene, image'; colormap=[
        RGBA(0, 0, 0, 0), RGBA(0, 0, 0, 1),
    ],
    colorrange=(0, 20)
)
save("outputs/kaplan.png", scene)

# psquared = vec(sum((image ./ num_rays).^2, dims=1)) ./ num_rays
# lines(psquared, axis=(limits=(nothing, (0, 4e-5)),))
# save("outputs/kaplan_p2.png", current_figure())
lines(xs, num_branches)
current_figure()