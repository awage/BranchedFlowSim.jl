using DifferentialEquations
using StaticArrays

export ray_trajectories
export ray_trajectories2
export lattice_intersections

Vec2 = SVector{2,Float64}

function ray_f(dr::Vec2, r::Vec2, pot, t)
    return @inline force(pot, r[1], r[2])
end

function ray_f!(ddu, du, u, pot, t)
    # r[i] = u[:,i]
    sz = size(u)
    # @assert sz[1] == 2
    num_particles = sz[2]
    for i ∈ 1:num_particles
        x = u[1, i]
        y = u[2, i]
        ddu[:, i] .= force(pot, x, y)
    end
    nothing
end


function ray_trajectory(r::Vec2, p::Vec2, potential::AbstractPotential, T, dt)
    tspan = (0.0, Float64(T))
    prob = SecondOrderODEProblem{false}(ray_f, p, r, tspan, potential)
    sol = solve(prob, Yoshida6(), dt=dt)
    # sol = solve(prob, VelocityVerlet(), dt=dt)
    # sol = solve(prob, dt=dt)

    N = length(sol)
    ts = sol.t
    rs = zeros(2, N)
    ps = zeros(2, N)
    for j ∈ 1:N
        rs[:, j] .= sol[j].x[2]
        ps[:, j] .= sol[j].x[1]
    end
    return ts, rs, ps
end



function ray_trajectories(
    f,
    r0::Matrix{Float64},
    p0::Matrix{Float64},
    potential::AbstractPotential,
    T,
    dt)
    num_particles = size(r0)[2]
    @assert size(p0)[2] == num_particles

    # TODO: if there is a huge number of particles, split the group to limit
    # memory use.
    tspan = (0.0, Float64(T))
    prob = SecondOrderODEProblem{true}(ray_f!, p0, r0, tspan, potential)
    sol = solve(prob, Yoshida6(), dt=dt)

    # Perform solution handling in a separate method.
    # This is an important performance optimization, since Julia
    # can specialize this method for the specific solution type.
    function collect_sol(sol)
        N = length(sol)
        ts = Vector{Float64}(sol.t)
        rs::Matrix{Float64} = zeros(Float64, 2, N)
        ps::Matrix{Float64} = zeros(Float64, 2, N)
        for i ∈ 1:num_particles
            for (j, s) ∈ enumerate(sol.u)
                for k ∈ 1:2
                    @inbounds rs[k, j] = s.x[2][k, i]
                    @inbounds ps[k, j] = s.x[1][k, i]
                end
            end
            f(ts, rs, ps)
        end
    end
    collect_sol(sol)
    nothing
end

"""
    ray_trajectories2(
    f,
    r0::Matrix{Float64},
    p0::Matrix{Float64},
    potential::AbstractPotential,
    T,
    dt)

Version of ray_trajectories which performs numerical integration one particle at a time.

TODO: Get rid of this, not worth keeping if this is slower than
ray_trajectories. Currently this is kept for testing.
"""
function ray_trajectories2(
    f,
    r0::Matrix{Float64},
    p0::Matrix{Float64},
    potential::AbstractPotential,
    T,
    dt)
    num_particles = size(r0)[2]
    @assert size(p0)[2] == num_particles

    tspan = (0.0, Float64(T))
    N = ceil(Int, 1 + T / dt)
    rs = zeros(2, N)
    ps = zeros(2, N)

    function collect_sol(sol)
        @assert N == length(sol)
        for j ∈ 1:N
            rs[:, j] .= sol[j].x[2]
            ps[:, j] .= sol[j].x[1]
        end
        f(sol.t, rs, ps)
    end

    for i ∈ 1:num_particles
        r = Vec2(@view r0[:,i])
        p = Vec2(@view p0[:,i])
        prob = SecondOrderODEProblem{false}(ray_f, p, r, tspan, potential)
        sol = solve(prob, Yoshida6(), dt=dt)
        collect_sol(sol)
    end
    nothing
end

struct IntersectDetector
end

function lattice_intersections(f, intersect, step, offset, rs, ps)
    A = hcat(intersect, step)
    Ainv = SMatrix{2,2,Float64,4}(inv(A))
    num_points = size(rs)[2]

    for i ∈ 1:(num_points-1)
        r0 = Vec2(@view rs[:, i])
        r1 = Vec2(@view rs[:, i+1])
        p0 = Vec2(@view ps[:, i])
        p1 = Vec2(@view ps[:, i+1])

        q0 = Ainv * (r0+offset)
        q1 = Ainv * (r1+offset)

        # Look for t such that q0+t*(q1-q0) == (x, k)
        # where k is an integer.
        q0i = floor(Int, q0[2])
        q1i = floor(Int, q1[2])
        if q0i != q1i
            t = if q0i < q1i
                (q1i - q0[2]) / (q1[2] - q0[2])
            else
                (q0[2]-q0i) / (q0[2] - q1[2])
            end
            q = q0 + t * (q1 - q0)
            p = p0 + t * (p1 - p0)
            # w = Ainv * p
            w = dot(intersect, p) / norm(intersect)
            f(r0 + t * (r1 - r0), p0 + t * (p1 - p0),
                q[1] - floor(q[1]), w[1])
        end
    end
end

# struct Hist2D
#     xs :: LinRange{Floa64, Int64}
#     ys :: LinRange{Floa64, Int64}
#     count :: Matrix{}
#     function Hist2D(x0, y0, x1, y1, Nx, Ny=Nx)
#         
#     end
# end
