using DifferentialEquations
using StaticArrays

Vec2 = SVector{2, Float64}

function ray_f(dr::Vec2, r :: Vec2, pot, t)
    return @inline force(pot, r[1], r[2])
end

function ray_f!(ddu, du, u, pot, t)
    # r[i] = u[:,i]
    sz = size(u)
    # @assert sz[1] == 2
    num_particles = sz[2]
    for i ∈ 1:num_particles
        x = u[1,i]
        y = u[2,i]
        ddu[:,i] .= force(pot, x, y)
    end
    nothing
end


function ray_trajectory(r :: Vec2, p :: Vec2, potential :: AbstractPotential, T, dt)
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
    r0 :: Matrix{Float64},
    p0 :: Matrix{Float64}, 
    potential :: AbstractPotential,
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