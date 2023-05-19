module BranchedFlowSim

# Helpers
export fermi_dot, gaussian_packet, absorbing_potential, lattice_points_in_a_box, lattice_potential, triangle_potential

using LinearAlgebra
using FFTW
using Trapz
using ColorSchemes
using ColorTypes # For RGB
using FixedPointNumbers # For N0f8

# Qualified imports
import VideoIO
import ImageIO
import FileIO

const mass::Float64 = 1.0

ANIM_FPS = 15

function fermi_step(x, α)
    return 1 / (1 + exp(-x / α))
end

function fermi_dot(xs, pos, radius, α=radius / 10)
    # N x N matrix of distance from `pos`
    dist = sqrt.(((xs .- pos[2]) .^ 2) .+ transpose(((xs .- pos[1]) .^ 2)))
    α = radius / 10
    return fermi_step.(radius .- dist, α)
end

function gaussian_packet(xs, pos, momentum, Δ)
    x0 = pos[1]
    y0 = pos[2]
    px = momentum[1]
    py = momentum[2]
    return (
        exp.(
            -((xs .- y0) .^ 2) / (2 * Δ^2)
            +
            1im * (py .* xs)
        ) * transpose(exp.(
            -((xs .- x0) .^ 2) / (2 * Δ^2)
            +
            1im * (px .* xs)
        ))
    ) / (√π * Δ)
end

"""
    braket(xs, Ψ, Φ)

For 2D wavefunctions `a` and `b` in coordinate basis as given by `xs`, returns
<a|b> = ∫∫ Ψ*(x,y)Φ(x,y) dx dy.
"""
function braket(xs, Ψ, Φ)
    ca = conj.(Ψ)
    ca .*= Φ
    return trapz((xs, xs), ca)
end

function total_prob(xs, Ψ)
    return trapz((xs, xs), abs.(Ψ) .^ 2)
end

function absorbing_potential(xs, strength, width)
    N = length(xs)
    L = xs[end] - xs[1]

    # Width in grid points
    width_n = round(Int, N * width / L)
    pot = zeros((N, N))
    pot[1:width_n, :] .+= LinRange(strength, 0, width_n)
    pot[:, 1:width_n] .+= LinRange(strength, 0, width_n)'
    pot[N-width_n+1:N, :] .+= LinRange(0, strength, width_n)
    pot[:, N-width_n+1:N] .+= LinRange(0, strength, width_n)'
    return -1im * pot
end


"""
    SplitOperatorStepper

foofoo
"""
struct SplitOperatorStepper
    dt :: Float64
    V_step :: Matrix{ComplexF64}
    T_step :: Matrix{ComplexF64}
    
    prepared_fft :: Any
    prepared_ifft :: Any

    function SplitOperatorStepper(xs, Δt, potential)
        N = length(xs)
        @assert size(potential) == (N, N)
        dx = xs[2] - xs[1]
        # Find momentum values corresponding to each fft value
        px = 2π * fftfreq(N, 1 / dx)
        # Kinetic operator is diagonal in p space:
        p2 = (px .^ 2) .+ transpose(px .^ 2)
        T_step = exp.(-Δt .* 1im .* p2 / 2mass)
        # Potential operator is diagonal in x space:
        V_step = exp.(-1im .* Δt .* potential)

        # Prepare in-place FFT to make repeated application faster.
        psi_placeholder = zeros(ComplexF64, N, N)
        my_fft = plan_fft!(psi_placeholder)
        my_ifft= plan_ifft!(psi_placeholder)
        return new(Δt, V_step, T_step, my_fft, my_ifft)
    end
end

function timestep!(Ψ, stepper)
    @assert size(Ψ) == size(stepper.V_step) "$(size(Ψ)) != $(size(stepper.V_step))"
    # Potential step in x-space:
    Ψ .*= stepper.V_step
    # Kinetic step in p-space:
    stepper.prepared_fft * Ψ
    Ψ .*= stepper.T_step
    stepper.prepared_ifft * Ψ
end

function time_evolve(xs, potential, Ψ, T, Δt)
    N = length(xs)
    stepper = SplitOperatorStepper(xs, Δt, potential)
    # Make a copy since we'll be modifying Ψ in-place.
    Ψ = copy(Ψ)
    ts = 0:Δt:T
    steps = length(ts)
    result_Ψs = zeros(ComplexF64, N, N, steps)
    result_Ψs[:, :, 1] = Ψ
    for i ∈ 2:steps
        timestep!(Ψ, stepper)
        result_Ψs[:, :, i] = Ψ
    end
    return result_Ψs, Vector(0:Δt:T)
end

"""
    lattice_points_in_box(avec, bvec, L)

Returns all lattice points in L-by-L sized box (2D) centered at (0,0). `avec`
and `bvec` are the generating vectors. Returns a matrix with 2 rows and one
column per lattice point.
"""
function lattice_points_in_a_box(A, L)
    # Start with a vector of 2d vectors
    points = Vector{Vector{Float64}}()
    avec = A[:, 1]
    bvec = A[:, 2]
    halfL = L / 2
    # Solve the maximum integer multiple of each vector by mapping each corner
    # of the box to the lattice basis.
    Na = 0
    Nb = 0
    for corner ∈ [[halfL, halfL], [halfL, -halfL]]
        ca, cb = A \ corner
        Na = max(Na, ceil(Int, abs(ca)))
        Nb = max(Nb, ceil(Int, abs(cb)))
    end
    for ka ∈ -Na:Na
        for kb ∈ -Nb:Nb
            v = ka * avec + kb * bvec
            if maximum(abs.(v)) <= halfL
                push!(points, v)
            end
        end
    end
    # Convert to a N-by-2 matrix
    return reduce(hcat, points)
end

function triangle_lattice_points_in_a_box(a, L)
end

function lattice_potential(xs, A, dot_height, dot_radius, offset=[0,0], dotfunc = fermi_dot)
    N = length(xs)
    L = (xs[end] - xs[1]) + 10 * dot_radius + abs(maximum(offset))
    V = zeros(N, N)
    for point ∈ eachcol(lattice_points_in_a_box(A, L))
        V .+= fermi_dot(xs, point + offset , dot_radius)
    end
    return V * dot_height
end


function triangle_potential(xs, a, dot_height, dot_radius=0.1 * a)
    # Lattice matrix
    A = a * [
        1 cos(π / 3)
        0 sin(π / 3)
    ]
    # Add offset such that 0,0 is in the center of a triangle.
    # Computed by taking the average of three lattice points,
    # one of which is (0,0).
    offset = (A[:, 1] + A[:, 2]) / 3
    return lattice_potential(xs, A, dot_height, dot_radius, offset)
end

function simple_potential(xs)
    # Build potential function
    N = length(xs)
    potential = zeros(ComplexF64, N, N)
    # Add absorbing potential at edges
    absorb_width = 2
    absorb_strength = 20
    potential += absorbing_potential(xs, absorb_strength, absorb_width)

    dot_height = 100
    dot_radius = 0.25
    potential += triangle_potential(xs, 2, dot_height, dot_radius)
    return potential
end

function psi_heatmap(xs, Ψ, potential, maxval=:none)
    if maxval == :none
        maxval = maximum(abs.(real(Ψ)))
    end
    max_pot = maximum(real(potential))
    img = get(ColorSchemes.gray1, real(potential) / max_pot)
    img .+= get(ColorSchemes.berlin, 0.5 * (real(Ψ) / maxval .+ 1))
    clampcolor(c) = clamp(c, 0.0, 1.0)
    return mapc.(clampcolor, img)
    # return 
    # plot(img, axis=nothing, size=size(Ψ))
    # heatmap(xs, xs, real(Ψ[:, :]),
    #     seriescolor=:berlin, clims=(-maxval, maxval),
    #     aspect_ratio=1.0,
    #     size=size(Ψ),
    #     axis=nothing,
    #     colorbar=nothing,
    #     margin = 0.0 * Plots.mm)
end

function frame_indices(ts, duration, fps)
    if length(ts) < fps * duration
        duration = length(ts) ÷ fps
    end
    T = ts[end]
    dt = ts[2] - ts[1]
    return [min(length(ts), Int(floor(1 + t / dt)))
            for t ∈ LinRange(0, T, fps * duration)]
end

function animate_evolution(fname, xs, potential, Ψs, ts)
    duration = 10
    fps = ANIM_FPS
    maxval = maximum(real(Ψs))

    imgstack = (
        convert(Matrix{RGB{N0f8}}, psi_heatmap(xs, Ψs[:, :, i], potential, maxval))
        for i ∈ frame_indices(ts, duration, ANIM_FPS)
    )
    VideoIO.save(fname, imgstack, framerate=ANIM_FPS)
end

function simple_anim()
    L = 20
    Nx = 1000
    T = 5
    dt = T / 150
    momentum = [5, 0]
    xs = LinRange(-L / 2, L / 2, Nx)

    # Build initial wavefunction
    Ψ = gaussian_packet(xs, [0, 0], momentum, 1)

    potential = simple_potential(xs)
    Ψs, ts = time_evolve(xs, potential, Ψ, T, dt)
    num_steps = size(Ψs)[2]

    fps = 15
    dur = 10
    animate_evolution("packet_2d.mp4", xs, potential, Ψs, ts)
end

function compute_correlation_function(xs, Ψs)
    N = length(xs)
    Nx, Ny, Nt = size(Ψs)
    @assert N == Nx
    @assert N == Ny
    return [braket(xs, Ψs[:, :, 1], Ψs[:, :, i]) for i ∈ 1:Nt]
end

end # module BranchedFlowSim
