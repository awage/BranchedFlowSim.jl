using BranchedFlowSim
using LinearAlgebra
using StaticArrays
using LaTeXStrings
using CairoMakie
using ColorSchemes
using Makie

Vec2 = SVector{2,Float64}

function squared_norm(vec)
    s = 0.0
    for z ∈ vec
        s += z^2
    end
    return s
end

function plot_potential!(ax, xs, ys, pot)
    V = grid_eval(xs, ys, pot)
    minV, maxV = extrema(V)
    pot_colormap = reverse(ColorSchemes.grays)
    pot_img = get(pot_colormap, 0.4 * (V .- minV) ./ (maxV - minV + 1e-9))
    image!(ax, xs, xs, pot_img)
end

function circle_line_intersect_angle(
    r1::Vec2,
    r2::Vec2,
    center::Vec2,
    radius::Real)::Float64
    r1 = r1 - center
    r2 = r2 - center

    delta = r2 - r1
    a = delta[1]^2 + delta[2]^2
    b = 2 * dot(r1, delta)
    c = r1[1]^2 + r1[2]^2 - radius^2
    if b^2 - 4 * a * c < 0
        @show r1, r2, radius
        @show a, b, c
    end
    if b > 0
        m = (-b + sqrt(b^2 - 4 * a * c)) / (2a)
    else
        m = (-b - sqrt(b^2 - 4 * a * c)) / (2a)
    end
    # if 0 <= m  <= 1
    crossing = r1 + m * delta
    # println("R=$(norm(crossing))")
    return atan(crossing[2], crossing[1])
    # else
    #     return nothing
    # end
end

"""
    detect_circle_crossings(pot, r0, p, distances)

Returns a vector of (idx, θ, flow_sign) pairs. `idx` refers to the circle of
radius distances[idx], `θ` is the angle along the circle and `flow_sign`
is +1.0 when the particle is exiting the circle and -1.0 when entering.
"""
function detect_circle_crossings(pot, r0, p0, distances)::Vector{Tuple{Int,Float64,Float64}}
    r0 = Vec2(r0)
    p0 = Vec2(p0)
    Δd = distances[2] - distances[1]
    @assert(distances[end] ≈ distances[1] + Δd * (length(distances) - 1),
        "distances should be a linear range")
    crossings = Tuple{Int,Float64,Float64}[]
    dt = 0.005
    pint = ParticleIntegrator(pot, r0, p0, dt)
    # println("E=$(particle_energy(pint))")

    dist_idx::Int = 0
    r1 = particle_position(pint)
    while true
        particle_step!(pint)
        r2 = particle_position(pint)
        dist_idx2 = floor(Int, 1 + (norm(r2 - r0) - distances[1]) / Δd)
        if dist_idx2 != dist_idx
            di = max(dist_idx, dist_idx2)
            if di > length(distances)
                break
            end
            d = distances[di]
            # We've moved past a circle, write it down.
            ϕ = circle_line_intersect_angle(r1, r2, r0, d)
            sign = if dist_idx2 > dist_idx
                1.0
            else
                -1.0
            end
            push!(crossings, (di, ϕ, sign))
            dist_idx = dist_idx2
        end
        r1 = r2
    end
    return crossings
end

function max_intensity_at_range(pot, r0, num_rays, distances, b)
    # Pairs (θ, sign)
    # Sign is +1.0 if the trajectory exited the circle, -1.0 if the trajectory
    # went back into the circle (subtracting flux).
    crossings = [Vector{Tuple{Float64,Float64}}() for _ ∈ distances]

    angles = LinRange(0, 2pi, num_rays + 1)[1:end-1]
    Δd = distances[2] - distances[1]
    @assert(distances[end] ≈ distances[1] + Δd * (length(distances) - 1),
        "distances should be a linear range")
    # Compute crossings in parallel. Locking is required.
    lk = ReentrantLock()
    Threads.@threads for i ∈ 1:num_rays
        θ = angles[i]
        p = normalized_momentum(pot, r0, Vec2(cos(θ), sin(θ)))
        ray_crossings = detect_circle_crossings(pot, r0, p, distances)

        lock(lk)
        try
            for (idx, ϕ, flow_sign) ∈ ray_crossings
                push!(crossings[idx], (ϕ, flow_sign))
            end
        finally
            unlock(lk)
        end
    end
    max_intensity = zeros(length(distances))
    max_angles = zeros(length(distances))
    for (i, dist) ∈ enumerate(distances)
        sort!(crossings[i])
        max_flow = 0.0
        max_angle = 0.0
        cross = crossings[i]
        if 2pi * dist <= b
            # Weird case: distances[i] is so small that b covers the whole
            # circle perimeter.
            max_flow = num_rays
        else
            # Sweep through crossings and count maximum flow through
            # an arc of length at most b.
            start = 1
            flow::Float64 = cross[1][2]
            M = length(cross)
            for j ∈ 2:2M
                ji = 1 + (j - 1) % M
                arc = cross[ji][1] * dist
                if j > M
                    # Second passing
                    arc += 2pi * dist
                end
                flow += cross[ji][2]
                while start <= M && arc - dist * cross[start][1] > b
                    flow -= cross[start][2]
                    start += 1
                end
                if start > M
                    # We've gone a full circle, the following cases
                    # were all handled the first time around.
                    break
                end
                @assert start <= j
                if flow > max_flow
                    max_flow = flow
                    max_angle = cross[ji][1] - 0.5 * b / (dist)
                end
                max_flow = max(max_flow, flow)
            end
        end
        max_intensity[i] = max_flow / num_rays
        max_angles[i] = max_angle
    end
    return max_intensity, max_angles
end

function test_figure(pot, r0, num_rays, distances)
    fig = Figure()

    r0 = Vec2(r0)
    xmin = r0[1] - distances[end]
    xmax = r0[1] + distances[end]
    ymin = r0[2] - distances[end]
    ymax = r0[2] + distances[end]
    ax = Axis(fig[1, 1],
        limits=((xmin, xmax), (ymin, ymax)),
        aspect=DataAspect()
    )
    xs = LinRange(xmin, xmax, 800)
    ys = LinRange(ymin, ymax, 800)
    plot_potential!(ax, xs, ys, pot)

    angles = LinRange(0, 2pi, num_rays + 1)[1:end-1]
    Δd = distances[2] - distances[1]
    @assert(distances[end] ≈ distances[1] + Δd * (length(distances) - 1),
        "distances should be a linear range")
    for θ ∈ angles
        p = normalized_momentum(pot, r0, Vec2(cos(θ), sin(θ)))
        pint = ParticleIntegrator(pot, r0, p, 0.002)
        traj_xs = Float64[]
        traj_ys = Float64[]
        for step ∈ 0:typemax(Int64)
            particle_step!(pint)
            if step % 10 == 0
                r = particle_position(pint)
                push!(traj_xs, r[1])
                push!(traj_ys, r[2])
                if norm(r) > distances[end] + Δd
                    break
                end
            end
        end
        lines!(ax, traj_xs, traj_ys)

        crossings = detect_circle_crossings(pot, r0, p, distances)
        for (idx, ϕ, flow_sign) ∈ crossings
            d = distances[idx]
            crossing = r0 + d * Vec2(cos(ϕ), sin(ϕ))
            color = if flow_sign > 0
                :red
            else
                :blue
            end
            scatter!(ax, crossing[1], crossing[2], marker=:xcross, color=color)
        end
    end

    unit_circle_points = hcat(([cos(θ), sin(θ)] for θ ∈ LinRange(0, 2pi, 200))...)
    for d ∈ distances
        points = r0 .+ d * unit_circle_points
        lines!(ax, points, linestyle=:dot, color=:green)
    end
    return fig
end

## Do the computation

# zpot = ZeroPotential()
lpot = LatticePotential(I, 0.25, 0.04)
pot = correlated_random_potential(10, 10, 0.5, 0.04, 0)
# pot = correlated_random_potential(1, 1, 0.25, 0.04, 0)
distances = LinRange(1, 80, 80)
#distances = 1:15
# fig = test_figure(pot, [0.51, 0.45], 10, distances)
# display(fig)

num_rays = 2000
b = 1
r0 = [0.47, 0.51]
int, ma_pot = max_intensity_at_range(pot, r0, num_rays, distances, b)
lint, ma_lpot = max_intensity_at_range(lpot, r0, num_rays, distances, b)

## Do the plot

fig = Figure()
ax = Axis(fig[1, 1], title=L"Max intensity, $b=%$b$")

xs = LinRange(distances[1], distances[end], 200)
lines!(ax, distances, int, label=LaTeXString("Random"))
lines!(ax, distances, lint, label=LaTeXString("Periodic"))
lines!(ax, xs, b ./ (2pi * xs), linestyle=:dash, label=L"\frac{b}{2\pi r}")

axislegend(ax)

display(fig)

# 

W = 40
PLOT_W = 1024
xs = LinRange(-W / 2, W / 2, PLOT_W)
ys = LinRange(-W / 2, W / 2, PLOT_W)
r0s = r0 * ones(num_rays)'
angles = LinRange(0, 2pi, num_rays + 1)[1:end-1]

## Plot visualization of the max angles
# pot = lpot
# name = rand
# ma = ma_lpot
for (pot, name, ma) ∈ [
    (pot, "rand", ma_pot),
    (lpot, "fermi_lattice", ma_lpot)
]
    p0s = foldl(hcat,
        (
            normalized_momentum(pot, r0, [cos(θ), sin(θ)])
            for θ ∈ angles
        ))
    @time "visualize $name" img = BranchedFlowSim.classical_visualize_rays(
        xs, ys, pot, r0s, p0s, 20, 0.005)

    fig = Figure(resolution=(PLOT_W,PLOT_W))
    ax = Axis(fig[1,1], aspect=DataAspect())
    # hidedecorations!(ax)
    image!(ax, xs, ys, img')
    
    points = [r0 + d*[cos(a),sin(a)] for (d,a)∈ zip(distances, ma) if d < W/2]
    points = foldl(hcat, points)

    scatter!(ax, points, color=:blue)

    path = "outputs/classical/longevity_$name.png"
    save(path, fig)
    println("Wrote $path")
    display(fig)
end