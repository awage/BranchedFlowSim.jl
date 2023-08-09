using BranchedFlowSim
using GLMakie, Makie
using FileIO
using ColorTypes
using ColorSchemes
using HDF5

function matrix_from_one_indices(size, one_indices)
    mat = falses(size...)
    for I ∈ eachcol(one_indices)
        mat[I...] = true
    end
    return mat
end

function potential_heatmap(xs, ys, potential)
    V = grid_eval(xs, ys, potential)
    minV, maxV = extrema(V)
    pot_colormap = reverse(ColorSchemes.grays)
    return get(pot_colormap, 0.4 * (V .- minV) ./ (maxV - minV + 1e-9))
end

# TODO: Hack here to use a different monitor for the plot
GLMakie.activate!(inline=false)

# data = load("../csc_outputs/classical/fermi_lattice_0.04_0.25_0.20/data.h5")
data = load("outputs/classical/fermi_lattice_0.04_0.25_0.20_correlation/data.h5")

psos_rs = data["rs"]
psos_ps = data["ps"]

# TODO: Write and load these from the file
step = [1.0, 0.0]
intersect = [0.0, 1.0]

fig = Figure()
display(fig)

W = 20

real_ax = Axis(fig[1, 1], limits=((-W / 2, W / 2), (-W / 2, W / 2)),
    aspect=DataAspect())
psos_ax = Axis(fig[1, 2],
    xzoomlock=true, yzoomlock=true,
    xrectzoom=false, yrectzoom=false
    )
fig[2, 1:2] = bottomgrid = GridLayout(height=200)

dim_max = data["dim_max"]
dim_count = data["dim_count"]

trajectory_dim = data["trajectory_dim"]
trajectory_start = data["trajectory_start"]
trajectory_plots = [
    data["trajectory_plot/$x"] for x ∈ 1:length(trajectory_dim)
]
trajectory_bitmaps = [
    matrix_from_one_indices(
        (length(psos_rs), length(psos_ps)), t)
    for t ∈ trajectory_plots
]

# Plot
d2 = copy(dim_max)
for idx ∈ eachindex(d2)
    if dim_count[idx] == 0
        d2[idx] = NaN
    end
end
hm = heatmap!(psos_ax, psos_rs, psos_ps, d2, colorrange=(1.0, 2.0),
    colormap=ColorSchemes.viridis)
cm = Colorbar(fig[1, 3], hm)

highlight_hms = []
traj_lines = []

pot = potential_from_h5_data(data)

bg_xs = LinRange(-W, W, 1024)
pot_hm = potential_heatmap(bg_xs, bg_xs, pot)
image!(real_ax, bg_xs, bg_xs, pot_hm)

function add_traj(ry, py)
    ridx = round(Int, 1 + (ry - psos_rs[1]) / (psos_rs[2] - psos_rs[1]))
    pidx = round(Int, 1 + (py - psos_ps[1]) / (psos_ps[2] - psos_ps[1]))
    if ridx < 1 || ridx > length(psos_rs) || pidx < 1 || pidx > length(psos_ps)
        return false
    end
    # Find first matching trajectory
    for t ∈ trajectory_bitmaps
        if t[ridx, pidx]
            global highlight_hms
            global traj_lines
            highlight_hm =
                heatmap!(psos_ax,
                    psos_rs,
                    psos_ps,
                    t,
                    colormap=[RGBA(0, 0, 0, 0), RGBA(1, 0, 0, 1)]
                )
            push!(highlight_hms, highlight_hm)
            r = ry * intersect
            V = pot(r[1], r[2])
            px = sqrt(1 - 2V - py^2)

            _, traj_rs, _ = ray_trajectory(r, [px, py], pot, 100, 0.005, 1.00)
            traj_line = lines!(real_ax, traj_rs, color=:red)
            push!(traj_lines, traj_line)
            break
        end
    end
    return true
end

on(events(psos_ax).mousebutton) do mb
    # mb = events(psos_ax).mousebutton[]
    if mb.button == Mouse.left && (mb.action == Mouse.press || mb.action == Mouse.repeat)
        mp = mouseposition(psos_ax)
        # Select this trajectory 
        println("mp=$mp")
        # add_traj(mp[1], mp[2])
        return Consume(add_traj(mp[1], mp[2]))
    end
end
#=
on(events(psos_ax).mouseposition) do mp
    mb = events(psos_ax).mousebutton[]
    println("move event: mb=$mb")
    if mb.button == Mouse.left && (mb.action == Mouse.press || mb.action == Mouse.repeat)
        println("input mp=$mp")
        mp = mouseposition(psos_ax)
        # Select this trajectory 
        println("new mp=$mp")
        return Consume(add_traj(mp[1], mp[2]))
    end
end
=#

clear_button = bottomgrid[1, 1] = Button(fig, label="Clear")
on(clear_button.clicks) do n
    global traj_lines
    global highlight_hms
    println("clearing")
    for t ∈ traj_lines
        # Delete previous highlight
        delete!(real_ax, t)
        traj_lines = []
    end
    for hm ∈ highlight_hms
        # Delete previous highlight
        delete!(psos_ax, hm)
    end
    highlight_hms = []
end

reset_view = bottomgrid[1, 2] = Button(fig, label="Reset view")
on(reset_view.clicks) do n
    println("reset view")
end


# TODO: Add UI for selecting the potential.