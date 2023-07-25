# Temporary script to play around with classical simulations
# Move to src/ once more mature

using BranchedFlowSim
using ColorSchemes
using CairoMakie
using DifferentialEquations
using LinearAlgebra
using Makie

function ray_f!(ddu, du, u, pot, t)
    # r[i] = u[:,i]
    sz = size(u)
    @assert sz[1] == 2
    num_particles = sz[2]
    for i ∈ 1:num_particles
        x = u[1,i]
        y = u[2,i]
        ddu[:,i] .= force(pot, x, y)
    end
    nothing
end

lattice_a::Float64 = 0.2
dot_radius::Float64 = 0.25 * lattice_a
v0::Float64 = 0.04
dt::Float64 = 0.01

# pot = LatticePotential(lattice_a * I, dot_radius, v0;
#     offset=lattice_a * [0.45,0.51])
pot = correlated_random_potential(4,4,0.1,v0)
# pot= fermi_dot_lattice_cos_series(1, lattice_a, dot_radius, -v0)

num_particles = 10000
# num_particles = 4
r0 = lattice_a * [0.1, 0.3] * ones(num_particles)'
angles = LinRange(0, 2pi, num_particles+1)[1:end-1]
p0 = hcat(([cos(θ),sin(θ)] for θ ∈ angles)...)

tspan = (0.0, 6.0)

prob = SecondOrderODEProblem{true}(ray_f!, p0, r0, tspan, pot)

saveat = 0.05
@time "ode solve" sol = solve(prob, Yoshida6(), saveat=saveat, dt=dt);

## Plot
function plot_line(f, x0, y0, x1, y1)
    x = round(Int32, x0)
    y = round(Int32, y0)
    xe = round(Int32, x1)
    ye = round(Int32, y1)
    # dx = round(Int32, abs(x1 - x0))
    # dx = abs(x1 - x0)
    dx = abs(xe - x)
    sx ::Int32 = x < xe ? 1 : -1
    # sx ::Int32 = x0 < x1 ? 1 : -1
    # dy = round(Int32, -abs(y1 - y0))
    # dy = -abs(y1 - y0)
    dy = -abs(ye - y)
    sy ::Int32 = y < ye ? 1 : -1
    # sy ::Int32 = y0 < y1 ? 1 : -1
    error::Float64 = dx + dy
    while true
        @inline f(x, y)
        if x == xe && y == ye
            break
        end
        e2::Float64 = 2 * error
        if e2 >= dy
            if x == xe
                break
            end
            error = error + dy
            x = x + sx
        end
        if e2 <= dx
            if y == ye
                break
            end
            error = error + dx
            y = y + sy
        end
    end
end

function iter_pixels(f, x0, y0, x1, y1)
    # Always draw in positive x direction
    if x0 > x1
        iter_pixels(-x0, y0, -x1, y1) do x,y
            @inline f(-x, y)
        end
        return nothing
    end
    ix :: Int = round(Int, x0)
    iy :: Int = round(Int, y0)
    # Form line equation y = mx + b
    m :: Float64 = (y1 - y0) / (x1 - x0)
    b :: Float64 = y0 - m * x0
    # End pixel coordinate
    ixe :: Int = round(Int, x1)
    iye :: Int = round(Int, y1)

    if abs(x0 - x1) < 1e-9
        for y ∈ iy:iye
            f(y, ix)
        end
        return nothing
    end

    ny ::Float64  = b + m * (ix + 0.5) - iy
    if y0 < y1 
        # Moving upwards
        while true
            @inline f(ix, iy)
            if ix == ixe && iy == iye
                break
            end
            # We move either up or to the right
            if ny > 0.5
                iy += 1
                ny -= 1
            else
                ix += 1
                ny += m
            end
        end
    else
        # Moving downwards
        while true
            @inline f(ix, iy)
            if ix == ixe && iy == iye
                break
            end
            # We move either down or to the right
            if ny < -0.5
                iy -= 1
                ny += 1
            else
                ix += 1
                ny += m
            end
        end
    end
    return nothing
end

Nh = 512
xs = LinRange(-4, 4, Nh)
img = zeros(Nh, Nh)

function nearest_idx(xs, x)
    dx = xs[2] - xs[1]
    return 1 + round(Int64, (x-xs[1]) / dx)
end

dx = xs[2] - xs[1]
for i ∈ 1:num_particles
    lx::Float64 = -1
    ly::Float64 = -1
    px::Int32 = 0
    py::Int32 = 0
    for j ∈ 1:length(sol)
        x, y = sol[j].x[2][:,i]
        xi = 1 + (x-xs[1]) / dx
        yi = 1 + (y-xs[1]) / dx
        if 1<xi<Nh && 1<yi<Nh && 1<lx<Nh && 1<ly<Nh
            # img[round(Int, yi), round(Int, xi)] += 2
            # plot_line(lx, ly, xi, yi) do x,y
            iter_pixels(lx, ly, xi, yi) do x,y
                if x != px || y != py
                    img[y,x] += 1
                end
                  px = x
                  py = y
            end
        end
        lx = xi
        ly = yi
    end
end


fire = reverse(ColorSchemes.linear_kryw_0_100_c71_n256)
f = heatmap(xs, xs, img,
    colorscale=Makie.pseudolog10,
    colorrange=(0, 200),
    colormap=fire)
display(current_figure())
#fig = Figure()
#ax = Axis(fig[1,1])
#display(fig)

R = 1.0
cross_angles = Float64[]
weights = Float64[]
for i ∈ 1:num_particles
    pos = hcat([s.x[2][:,1] for s ∈ sol.u]...)
end

