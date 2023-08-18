using DifferentialEquations
using StaticArrays

export ray_trajectory
export ray_trajectories
export ray_trajectories2
export lattice_intersections

export PoincareMapper
export PoincareIntersection
export next_intersection!
export set_position_on_surface!

export max_momentum
export ParticleIntegrator
export particle_step!
export particle_position
export particle_momentum
export particle_energy
export normalized_momentum
export normalize_particle_energy!
export local_lyapunov_exponent

const Vec2 = SVector{2,Float64}
const AbstractFloatVec = AbstractVector{<:Real}

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


function ray_trajectory(r::AbstractVector{<:Real}, p::AbstractVector{<:Real},
    potential::AbstractPotential, T, dt, saveat=dt)
    r = Vec2(r)
    p = Vec2(p)
    tspan = (0.0, Float64(T))
    prob = SecondOrderODEProblem{false}(ray_f, p, r, tspan, potential)
    sol = solve(prob, Yoshida6(), dt=dt, saveat=saveat)
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
        r = Vec2(@view r0[:, i])
        p = Vec2(@view p0[:, i])
        prob = SecondOrderODEProblem{false}(ray_f, p, r, tspan, potential)
        sol = solve(prob, Yoshida6(), dt=dt)
        collect_sol(sol)
    end
    nothing
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

        q0 = Ainv * (r0 + offset)
        q1 = Ainv * (r1 + offset)

        # Look for t such that q0+t*(q1-q0) == (x, k)
        # where k is an integer.
        q0i = floor(Int, q0[2])
        q1i = floor(Int, q1[2])
        if q0i != q1i
            t = if q0i < q1i
                (q1i - q0[2]) / (q1[2] - q0[2])
            else
                (q0[2] - q0i) / (q0[2] - q1[2])
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

function max_momentum(pot, r, E=0.5)
    V = pot(r[1], r[2])
    return sqrt(2 * (E - V))
end

"""
Iterator for advancing along a ray, one timestep at a time.

"""
struct ParticleIntegrator{Integrator<:OrdinaryDiffEq.ODEIntegrator}
    integrator::Integrator
    function ParticleIntegrator(pot::AbstractPotential,
        r::AbstractFloatVec,
        p::AbstractFloatVec,
        dt::Real)
        r0 = Vec2(r)
        p0 = Vec2(p)
        prob = SecondOrderODEProblem{false}(ray_f, p0, r0, (0, Inf), pot)
        integrator = init(prob, Yoshida6(), dt=dt, save_everystep=false)
        return new{typeof(integrator)}(
            integrator
        )
    end
end

"""
    particle_position(pint::ParticleIntegrator)::Vec2

Returns position of the particle.
"""
function particle_position(pint::ParticleIntegrator)::Vec2
    return pint.integrator.u.x[2]
end

"""
    particle_momentum(pint :: ParticleIntegrator) :: Vec2

Returns momentum (equal to velocity) of the particle.
"""
function particle_momentum(pint::ParticleIntegrator)::Vec2
    return pint.integrator.u.x[1]
end

function particle_t(pint::ParticleIntegrator)::Float64
    return pint.integrator.t
end

"""
    particle_step!(pint::ParticleIntegrator)

Advance ray by one timestep.
"""
function particle_step!(pint::ParticleIntegrator)
    step!(pint.integrator)
    return nothing
end

function normalized_momentum(
    pot::AbstractPotential,
    r,
    p, E=0.5
)
    V = pot(r[1], r[2])
    pnorm = sqrt(2 * (0.5 - V))
    return (pnorm / norm(p)) * p
end

"""
    normalize_particle_energy!(pint::ParticleIntegrator, E=0.5)


"""
function normalize_particle_energy!(pint::ParticleIntegrator, E=0.5)
    r = particle_position(pint)
    p = particle_momentum(pint)
    pot = pint.integrator.sol.prob.p
    p_new = normalized_momentum(pot, r, p, E)
    set_u!(pint.integrator,
        ArrayPartition((
            p_new, particle_position(pint)
        ))
    )
    return nothing
end

function reset_particle!(pint::ParticleIntegrator, r, p)
    set_u!(pint.integrator,
        ArrayPartition((
            Vec2(p), Vec2(r)
        ))
    )
end

function particle_energy(pint::ParticleIntegrator)
    r = particle_position(pint)
    p = particle_momentum(pint)
    pot = pint.integrator.sol.prob.p
    V = pot(r[1], r[2])
    return V + (p[1]^2 + p[2]^2) / 2
end

struct PoincareMapper{Integrator<:OrdinaryDiffEq.ODEIntegrator}
    integrator::Integrator
    offset::Vec2
    A::SMatrix{2,2,Float64,4}
    Ainv::SMatrix{2,2,Float64,4}

    function PoincareMapper(pot::AbstractPotential,
        r_int::Real,
        p_int::Real,
        intersect,
        step,
        offset,
        dt)
        r = Vec2(offset + r_int * intersect)

        # Normalize p:
        V = pot(r[1], r[2])
        E = 0.5
        py = p_int
        px = sqrt(2 * (E - V) - py^2)
        rot90 = rotation_matrix(-pi / 2)
        ey = intersect / norm(intersect)
        ex = rot90 * ey
        p = Vec2(px * ex + py * ey)
        return PoincareMapper(pot, r, p, intersect, step, offset, dt)
    end

    function PoincareMapper(pot::AbstractPotential,
        r0::AbstractVector{<:Real},
        p0::AbstractVector{<:Real},
        intersect,
        step,
        offset,
        dt)
        r0 = Vec2(r0)
        p0 = Vec2(p0)
        prob = SecondOrderODEProblem{false}(ray_f, p0, r0, (0, Inf), pot)
        A = SMatrix{2,2,Float64,4}(hcat(intersect, step))
        Ainv = inv(A)
        integrator = init(prob, Yoshida6(), dt=dt,
            save_everystep=false)
        return new{typeof(integrator)}(
            integrator,
            offset,
            A,
            Ainv
        )
    end
end

struct PoincareIntersection
    t::Float64

    # Relative position coordinate along the intersect vector.
    # r_int ∈ [0,1]
    r_int::Float64
    # Component of momentum coordinate parallel to the intersect vector
    # p_int ∈ [-1,1]
    p_int::Float64
end

"""
    next_intersection!(mapper::PoincareMapper, max_time = 1000.0)::Union{PoincareIntersection, Nothing}

Returns the next point 

Parameter `max_time` is used to avoid infinite loops for trajectories that never intersect
the surface. After `max_time`, `nothing` is returned. Callers should therefore 
"""
function next_intersection!(mapper::PoincareMapper, max_time=100.0)::Union{PoincareIntersection,Nothing}
    end_t = mapper.integrator.t + max_time
    while mapper.integrator.t < end_t
        r0 = mapper.integrator.u.x[2]
        p0 = mapper.integrator.u.x[1]
        step!(mapper.integrator)
        r1 = mapper.integrator.u.x[2]
        p1 = mapper.integrator.u.x[1]

        q0 = mapper.Ainv * (r0 + mapper.offset)
        q1 = mapper.Ainv * (r1 + mapper.offset)

        # Look for t such that q0+t*(q1-q0) == (x, k)
        # where k is an integer.
        q0i = floor(Int, q0[2])
        q1i = floor(Int, q1[2])
        if q0i != q1i
            t = if q0i < q1i
                (q1i - q0[2]) / (q1[2] - q0[2])
            else
                (q0[2] - q0i) / (q0[2] - q1[2])
            end
            q = q0 + t * (q1 - q0)
            p = p0 + t * (p1 - p0)
            intersect = mapper.A[:, 1]
            w = dot(intersect, p) / norm(intersect)

            # Map back to unit cell to retain accuracy
            qfrac = q[1] - floor(q[1])
            r1_new = mapper.A * (q1 - floor.(q1)) - mapper.offset
            # Normalize energy back to 0.5
            pot = mapper.integrator.sol.prob.p
            V = pot(r1_new[1], r1_new[2])
            pnorm = sqrt(2 * (0.5 - V))
            p1_new = (pnorm / norm(p1)) * p1

            set_u!(mapper.integrator,
                ArrayPartition((
                    p1_new, r1_new
                ))
            )
            return PoincareIntersection(mapper.integrator.t, qfrac, w[1])
        end
    end
    return nothing
end

"""
    set_position_on_surface!(mapper::PoincareMapper, r_int, p_int)

Reset Poincare mapper coordinates to given point on the Poincaré surface.
"""
function set_position_on_surface!(mapper::PoincareMapper, r_int, p_int)
    intersect = mapper.A[:, 1]
    r = Vec2(mapper.offset + r_int * intersect)

    pot = mapper.integrator.sol.prob.p
    V = pot(r[1], r[2])
    E = 0.5
    py = p_int
    px = sqrt(2 * (E - V) - py^2)
    rot90 = rotation_matrix(-pi / 2)
    ey = intersect / norm(intersect)
    ex = rot90 * ey
    p = Vec2(px * ex + py * ey)

    set_u!(mapper.integrator,
        ArrayPartition((
           p, r 
        ))
    )
    nothing
end

function total_energy(mapper::PoincareMapper)
    pot = mapper.integrator.sol.prob.p
    r = mapper.integrator.u.x[2]
    p = mapper.integrator.u.x[1]
    return pot(r[1], r[2]) + sum(p .^ 2) / 2
end

# struct Hist2D
#     xs :: LinRange{Floa64, Int64}
#     ys :: LinRange{Floa64, Int64}
#     count :: Matrix{}
#     function Hist2D(x0, y0, x1, y1, Nx, Ny=Nx)
#         
#     end
# end


"""
    classical_visualize_rays(xs, ys, pot, r0, p0, T, dt)

Returns an visualization of rays 
"""
function classical_visualize_rays(xs, ys, pot, r0, p0, T, dt=0.005)
    Nx = length(xs)
    Ny = length(ys)
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    hist::Matrix{Float64} = zeros(Nx, Ny)
    ray_trajectories(r0, p0, pot, T, dt) do ts, rs, ps
        lx::Float64 = -1
        ly::Float64 = -1
        px::Int = 0
        py::Int = 0
        for (x, y) ∈ eachcol(rs)
            xi = 1 + (x - xs[1]) / dx
            yi = 1 + (y - ys[1]) / dy
            if 1 < xi < Nx && 1 < yi < Ny && 1 < lx < Nx && 1 < ly < Ny
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

    # Produce a RGB image
    colorrange = (0.0, 2.0 * size(r0)[2] / 100)
    fire = reverse(ColorSchemes.linear_kryw_0_100_c71_n256)
    pot_colormap = ColorSchemes.grays
    data_colormap = fire

    V = grid_eval(xs, ys, pot)
    minV, maxV = extrema(V)
    minD, maxD = colorrange
    pot_img = get(pot_colormap, (V .- minV) ./ (maxV - minV + 1e-9))
    data_img = get(data_colormap, (hist .- minD) / (maxD - minD))

    img = mapc.((d, v) -> clamp(d - 0.2 * v, 0, 1), data_img, pot_img)
    return img
end

# Inner function in order to specialize based on integrator type
# TODO: Double check if this separation is actually useful.
function lyapunov_inner(pint0, pint1, T, d_start, perform_rescaling)
    d_max = d_start * 10
    d_min = d_start / 10
    d2_max = d_max^2
    d2_min = d_min^2

    log_sum::Float64 = 0.0
    rescalings = 0
    while particle_t(pint0) < T
        particle_step!(pint0)
        particle_step!(pint1)
        if perform_rescaling
            # Potentially renormalize pint1
            r0 = particle_position(pint0)
            p0 = particle_momentum(pint0)
            r1 = particle_position(pint1)
            p1 = particle_momentum(pint1)
            dr = r1 - r0
            dp = p1 - p0
            dvec = SVector{4,Float64}(dr[1], dr[2], dp[1], dp[2])
            d2 = dot(dvec, dvec)
            if d2 < d2_min || d2 > d2_max
                rescalings += 1
                d = sqrt(d2)
                dvec = d_start * (dvec / d)
                r1 = r0 + Vec2(dvec[1], dvec[2])
                p1 = p0 + Vec2(dvec[3], dvec[4])
                reset_particle!(pint1, r1, p1)

                dist_scaling = d / d_start
                log_sum += log(dist_scaling)
            end
        end
    end
    @show rescalings
    @assert particle_t(pint0) == particle_t(pint1)
    # println("end t=$(particle_t(pint0))")

    # Add last segment
    r0 = particle_position(pint0)
    p0 = particle_momentum(pint0)
    r1 = particle_position(pint1)
    p1 = particle_momentum(pint1)
    dr = r1 - r0
    dp = p1 - p0
    dvec = SVector(dr[1], dr[2], dp[1], dp[2])
    d2 = dot(dvec, dvec)
    d = sqrt(d2)
    dist_scaling = d / d_start
    log_sum += log(dist_scaling)

    return (1 / particle_t(pint0)) * log_sum
end

"""
See 
https://physics.stackexchange.com/questions/388676/how-to-calculate-the-maximal-lyapunov-exponents-of-a-multidimensional-system
"""
function local_lyapunov_exponent(pot, r0, p0, dt, T;
    perform_rescaling=true, d_norm=1e-6)::Float64
    # perform_rescaling = true
    # Starting distance between the two trajectories

    #     dvec = d_start * normalize([pi, exp(1), 3, -sqrt(2)])
    dvec = d_norm * normalize(rand(4) .- 0.5)
    pint0 = ParticleIntegrator(pot, r0, p0, dt)
    pint1 = ParticleIntegrator(pot, r0 + dvec[1:2], p0 + dvec[3:4], dt)
    return @inline lyapunov_inner(pint0, pint1, T, d_norm, perform_rescaling)
end