using BranchedFlowSim
using CairoMakie


sim_height = 10
sim_width = 100
num_rays = 512
Ny = num_rays
Nx = num_rays * (sim_width ÷ sim_height)
ys = LinRange(0, sim_height, Ny+1)[1:end-1]
xs = LinRange(0, sim_width, Nx)

potential = gaussian_correlated_random(xs, ys, 0.5)

function pixel_heatmap(path, data; kwargs...)
    heatmap!(scene, data; kwargs...)
    save(path, scene)
end
# scene = Scene(camera=campixel!, resolution=(Nx,Ny))
fig = Figure(resolution=(2*Nx,2*Ny))
ax = Axis(fig[1,1])
heatmap!(ax, xs, ys, potential')
fig