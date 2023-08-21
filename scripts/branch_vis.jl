#=
An interactive program to inspect how branches look in phase space.
=#
using GeometryBasics
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
v0 = 0.04
softness = 0.20
lattice_a = 1.0
# pot = LatticePotential(I, dot_radius, v0, softness=softness)
pot = correlated_random_potential(1, 1, 0.2, v0, 2)
# pot = fermi_dot_lattice_cos_series(1, lattice_a, dot_radius, v0; softness=softness)
# LatticePotential(I, dot_radius, v0, softness=softness)

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
fig[2, 1:2] = bottomgrid = GridLayout(height=200)

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

# Sliders
sg = SliderGrid(bottomgrid[1,1],
    (label="select radius", range=0.01:0.01:2, startvalue=0.2))
select_radius = sg.sliders[1].value

markersize_default = 8
markersize_highlight = 10

function select_trajectory(x,y)
    global highlight_image
    # look for trajectories passing through closer than some radius from given
    # point.
    R = select_radius[]
    R2 = R^2
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
    # Only highlight the beginning part.
    end_time = fill(convert(Float64, T), length(selected))
    D2 = norm(r0-[x,y])^2
    for j ∈ 1:size(trajs)[3]
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
    println("trajectories trimmed")

    hl_hist = make_trajectory_histogram(trajs, xs, ys)
    img = get(red_highlight, hl_hist / 5)
    if highlight_image !== nothing
        delete!(real_ax, highlight_image)
    end
    highlight_image = image!(real_ax, xs, ys, img')
    
    # Plot Poincare surface of section of selected trajectories up 
    empty!(psos_ax)
    for (et, j) ∈ zip(end_time, selected)
        mapper = PoincareMapper(pot, r0, p0[:, j], intersect, step, [0,0], dt)
        points = Vector{Float64}[]
        while true
            int = next_intersection!(mapper)
            if isnothing(int)
                println("Particle with p=$(p0[:,j]) didn't intersect")
                break
            end
            if int.t > et
                break
            end
            push!(points, [int.r_int, int.p_int])
        end
        if length(points) != 0
            scatter!(psos_ax, reduce(hcat, points), 
                # Color is based on the angle
                # color = RGB(HSV(rad2deg(angles[j]), 1, 1)),
                markersize=markersize_default,
                # We can just add a custom attribute here to remember the trajectory
                trajectory_idx = j
            )
        end
    end
end

last_picked = nothing
selected_trajectory_line = nothing
on(events(real_ax).mousebutton) do mb
    # mb = events(psos_ax).mousebutton[]
    if mb.button == Mouse.left && mb.action == Mouse.press
        pixel_pos = mouseposition(fig)
        if pixel_pos ∈ pixelarea(real_ax)[]
            mp = mouseposition(real_ax)
            # Select this trajectory 
            println("real_ax pressed at $mp")
            select_trajectory(mp[1], mp[2])
            return Consume(true)
        elseif pixel_pos ∈ pixelarea(psos_ax)[]
            mp = mouseposition(psos_ax)
            println("psos_ax pressed at $mp")
            picked,pidx = pick(psos_ax, pixel_pos)
            println("picked $picked")
            if picked isa Scatter
                global last_picked, selected_trajectory_line
                if !isnothing(last_picked)
                    if :markersize ∈ keys(last_picked.attributes)
                        last_picked.markersize[] = markersize_default
                        last_picked.strokewidth[] = 0
                    end
                end
                last_picked = picked
                picked.markersize[] = markersize_highlight
                picked.strokewidth[] = 2
                if !isnothing(selected_trajectory_line)
                    delete!(real_ax, selected_trajectory_line)
                end
                traj_idx = picked.trajectory_idx[]
                selected_trajectory_line = lines!(real_ax, trajectories[:,:,traj_idx], color=:blue)
            end
        end
    end
    return Consume(false)
end

# Draw branch selection marker
mousemarker = lift(events(fig).mouseposition) do mp
    area = pixelarea(real_ax)[]
    mp = Point2f0(mp) .- minimum(area)
    r = [Point2f0(to_world(real_ax.scene, mp))]
    return r
end
mouse_scatter = scatter!(real_ax,
    mousemarker,
    markersize=lift(x->2*x, select_radius),
    marker = Circle,
    markerspace=:data,
    color=RGBA(1,0,0, 0.3)
    )

