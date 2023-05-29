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

v0 = 0.03

sim_height = 20
sim_width = 100
num_rays = 1024
correlation_scale = 1

Ny = num_rays
Nx = num_rays * (sim_width ÷ sim_height)
ys = LinRange(0, sim_height, Ny+1)[1:end-1]
xs = LinRange(0, sim_width, Nx)

dx = xs[2]-xs[1]
dy = ys[2]-ys[1]
dt = dx

# potential = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
potential = v0 * gaussian_correlated_random(xs, ys, correlation_scale)
for p ∈ 

# fig = Figure(resolution=(2*Nx,2*Ny))
# ax = Axis(fig[1,1])
# heatmap!(ax, xs, ys, potential')
# fig

# Compute F(x,y) = -∂V(x,y)/∂y
# 
force = (circshift(potential, [-1, 0]).-circshift(potential, [1, 0])) ./ (2*dy)
# fig = Figure(resolution=(2*Nx,2*Ny))
# ax = Axis(fig[1,1])
# heatmap!(ax, xs, ys, force')
# fig
#
force_itp = interpolate(force, BSpline(Linear()))
force_itp = scale(force_itp, ys, xs)
force_itp = extrapolate(force_itp, Periodic())

ray_y = Vector(ys)
ray_py = zeros(num_rays)

image = zero(potential)
for (xi, x) ∈ enumerate(xs)
    # kick
    ray_py .+= dt .* force_itp(ray_y, x)
    # drift
    ray_y .+= dt .* ray_py
    # Collect
    for y ∈ ray_y
        yi = 1 + ((Ny + round(Int, (y / sim_height)*Ny)) % Ny)
        image[yi, xi] += 1
    end
    # Wrap
    ray_y .+= sim_height
    ray_y .%= sim_height
end

# pixel_heatmap("outputs/kaplan_potential.png", potential', colormap=:Greens)
# pixel_heatmap("outputs/kaplan_rays.png", image', colormap=:grayC)

scene = Scene(camera=campixel!, resolution=size(potential'))
heatmap!(scene, potential'; colormap=:Greens_3)
heatmap!(scene, image'; colormap=[
RGBA(0,0,0,0), RGBA(0,0,0,1),
],
colorrange=(0, 20)
)
save("outputs/kaplan.png", scene)
