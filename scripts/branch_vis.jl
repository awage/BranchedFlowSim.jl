#=
An interactive program to inspect how branches look in phase space
=#
using LinearAlgebra, Statistics
using StaticArrays
using BranchedFlowSim
using GLMakie, Makie
using FileIO
using ColorTypes
using ColorSchemes
using HDF5

function potential_heatmap(xs, ys, potential)
    V = grid_eval(xs, ys, potential)
    minV, maxV = extrema(V)
    pot_colormap = reverse(ColorSchemes.greens)
    return get(pot_colormap, 0.4 * (V .- minV) ./ (maxV - minV + 1e-9))
end

function make_trajectory_histogram(trajectories, xs,ys)
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    num_particles = size(trajectories)[3]
    hist = zeros(length(ys), length(xs))
    for j ∈ 1:num_particles
        lx::Float64 = -1
        ly::Float64 = -1
        px::Int = 0
        py::Int = 0
        for (x, y) ∈ eachcol(@view trajectories[:,:,j])
            xi = 1 + (x - xs[1]) / dx
            yi = 1 + (y - ys[1]) / dy
            if 1 < xi < Nh && 1 < yi < Nh && 1 < lx < Nh && 1 < ly < Nh
                # Loop through all cells touching the line.
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
    return hist
end

fire = reverse(ColorSchemes.linear_kryw_0_100_c71_n256)
red_highlight = ColorScheme([RGBA(0,0,0,0), RGBA(1,0,0,0.5)])

function heatmap_with_potential(data, potential; colors=ColorSchemes.grayC, colorrange=extrema(data))
    pot_colormap = ColorSchemes.Greens

    V = real(potential)
    minV, maxV = extrema(V)
    minD, maxD = colorrange
    print("datarange: [$(minD),$(maxD)]\n")
    pot_img = get(pot_colormap, 0.4*(V .- minV) ./ (maxV - minV + 1e-9))
    data_img = get(colors, (data .- minD) / (maxD - minD))
    
    img = mapc.((d,v) -> clamp(d * v, 0, 1), data_img, pot_img) 
    return img
end

# TODO: Hack here to use a different monitor for the plot
GLMakie.activate!(inline=false)

# Modify these to run with different potentials
dot_radius = 0.25
v0 = 0.08
softness = 0.1
pot = LatticePotential(I, dot_radius, v0, softness=softness)
# pot = correlated_random_potential(1, 1, 0.2, v0, 0)

step = [1.0, 0.0]
intersect = [0.0, 1.0]

# Width of the visible box
W = 30
num_particles = 3000
T = 2W
dt = 0.005
saveat = 0.05

# Pixel grid used for real space plotting
Nh = 1024
xs = LinRange(-W/2, W/2, Nh)
ys = LinRange(-W/2, W/2, Nh)

trajectory_length = 1 + convert(Int, T / saveat)

fig = Figure()
display(fig)

real_ax = Axis(fig[1, 1], limits=((-W / 2, W / 2), (-W / 2, W / 2)),
    aspect=DataAspect())
psos_ax = Axis(fig[1, 2], limits=((0, 1), (-1,1)))

angles = LinRange(0, 2pi, num_particles + 1)[1:end-1]
r0::Vector{Float64} = [0.51, 0.47]
p0::Matrix{Float64} = hcat((normalized_momentum(pot, r0, [cos(θ), sin(θ)]) for θ ∈ angles)...)

trajectories::Array{Float64, 3} = zeros(2, trajectory_length, num_particles)
Threads.@threads for j ∈ 1:num_particles
    ts, rs, ps = ray_trajectory(r0, p0[:,j], pot, T, dt, saveat)
    trajectories[:,:,j] = rs
end

histogram = make_trajectory_histogram(trajectories, xs, ys)
colorrange = (0.0, 2.0 * num_particles / 100)
img = heatmap_with_potential(
    histogram, 
    grid_eval(xs, ys, pot),
    colorrange=colorrange
)
image!(real_ax, xs, ys, img')
highlight_image = nothing

function select_trajectory(x,y)
    global highlight_image
    # look for trajectories passing through closer than some radius from given
    # point.
    R = 0.2
    R2 = R^2
    D2 = norm(r0-[x,y])^2
    selected = Int[]
    end_velocities = SVector{2, Float64}[]
    for j ∈ 1:num_particles
        for i ∈ 2:trajectory_length
            dist_x = trajectories[1,i,j] - x
            dist_y = trajectories[2,i,j] - y
            if dist_x^2 + dist_y^2 < R2
                push!(selected, j)
                vel = normalize(trajectories[:, i,j] - trajectories[:, i-1,j])
                push!(end_velocities, vel)
                break
            end
        end
    end

    # Only pick trajectories which point in similar direction.
    max_angle = pi/4
    new_selected = Int[]
    mean_vel = normalize(mean(end_velocities))
    for (ji, vel) ∈ enumerate(end_velocities)
        if dot(vel, mean_vel) > cos(max_angle)
            push!(new_selected, selected[ji])
        end
    end
    println("Picked $(length(new_selected)) trajectories from nearby $(length(selected))")
    selected = new_selected

    trajs = trajectories[:,:,selected]
    end_time = fill(convert(Float64, T), length(selected))
    for j ∈ 1:size(trajs)[3]
        # Only highlight the beginning part
        for i ∈ 2:trajectory_length
            if (trajs[1, i,j ] - r0[1])^2 + (trajs[2, i,j ] - r0[2])^2 > D2 + 1
                end_time[j] = i * saveat
                for k ∈ i:trajectory_length
                    trajs[:,k,j] = trajs[:,k-1,j]
                end
                break
            end
        end
    end

    hl_hist = make_trajectory_histogram(trajs, xs, ys)
    img = get(red_highlight, hl_hist / 5)
    if highlight_image !== nothing
        delete!(real_ax, highlight_image)
    end
    highlight_image = image!(real_ax, xs, ys, img')
    
    empty!(psos_ax)
    for (et, j) ∈ zip(end_time, selected)
        mapper = PoincareMapper(pot, r0, p0[:, j], intersect, step, [0,0], dt)
        points = Vector{Float64}[]
        while true
            int = next_intersection!(mapper)
            if int.t > et
                break
            end
            push!(points, [int.r_int, int.p_int])
        end
        if length(points) != 0
            scatter!(psos_ax, reduce(hcat, points), markersize=4)
        end
    end
end

on(events(real_ax).mousebutton) do mb
    # mb = events(psos_ax).mousebutton[]
    if mb.button == Mouse.left && mb.action == Mouse.press
        mp = mouseposition(real_ax)
        # Select this trajectory 
        println("mouse pressed at $mp")
        select_trajectory(mp[1], mp[2])
        return Consume(true)
    end
end
