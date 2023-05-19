module BranchedFlowSim

# Helpers
export fermi_dot, gaussian_packet, absorbing_potential, lattice_points_in_a_box, lattice_potential, triangle_potential

# Simulation
export SplitOperatorStepper, timestep!, time_evolution

# Visualization
export wavefunction_to_image, save_animation

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

function fermi_step(x, α)
    return 1 / (1 + exp(-x / α))
end

function fermi_dot(xgrid, pos, radius, softness=1)
    # N x N matrix of distance from `pos`
    dist = sqrt.(((xgrid .- pos[2]) .^ 2) .+ transpose(((xgrid .- pos[1]) .^ 2)))
    α = radius / (softness)
    return fermi_step.(radius .- dist, α)
end

function gaussian_packet(xgrid, pos, momentum, Δ)
    x0 = pos[1]
    y0 = pos[2]
    px = momentum[1]
    py = momentum[2]
    return (
        exp.(
            -((xgrid .- y0) .^ 2) / (2 * Δ^2)
            +
            1im * (py .* xgrid)
        ) * transpose(exp.(
            -((xgrid .- x0) .^ 2) / (2 * Δ^2)
            +
            1im * (px .* xgrid)
        ))
    ) / (√π * Δ)
end

"""
    braket(xgrid, Ψ, Φ)

For 2D wavefunctions `a` and `b` in coordinate basis as given by `xgrid`, returns
<a|b> = ∫∫ Ψ*(x,y)Φ(x,y) dx dy.
"""
function braket(xgrid, Ψ, Φ)
    ca = conj.(Ψ)
    ca .*= Φ
    return trapz((xgrid, xgrid), ca)
end

function total_prob(xgrid, Ψ)
    return trapz((xgrid, xgrid), abs.(Ψ) .^ 2)
end

function absorbing_potential(xgrid, strength, width)
    N = length(xgrid)
    L = xgrid[end] - xgrid[1]

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

    function SplitOperatorStepper(xgrid, Δt, potential)
        N = length(xgrid)
        @assert size(potential) == (N, N)
        dx = xgrid[2] - xgrid[1]
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

function time_evolution(xgrid, potential, Ψ, T, Δt)
    N = length(xgrid)
    stepper = SplitOperatorStepper(xgrid, Δt, potential)
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

function lattice_potential(xgrid, A, dot_height, dot_radius, offset=[0,0],
    dot_softness = 1)
    N = length(xgrid)
    L = (xgrid[end] - xgrid[1]) + 10 * dot_radius + abs(maximum(offset))
    V = zeros(N, N)
    for point ∈ eachcol(lattice_points_in_a_box(A, L))
        V .+= fermi_dot(xgrid, point + offset , dot_radius, dot_softness)
    end
    return V * dot_height
end


function triangle_potential(xgrid, a, dot_height, dot_radius=0.1 * a, dot_softness=1)
    # Lattice matrix
    A = a * [
        1 cos(π / 3)
        0 sin(π / 3)
    ]
    # Add offset such that 0,0 is in the center of a triangle.
    # Computed by taking the average of three lattice points,
    # one of which is (0,0).
    offset = (A[:, 1] + A[:, 2]) / 3
    return lattice_potential(xgrid, A, dot_height, dot_radius, offset, dot_softness)
end

function simple_potential(xgrid)
    # Build potential function
    N = length(xgrid)
    potential = zeros(ComplexF64, N, N)
    # Add absorbing potential at edges
    absorb_width = 2
    absorb_strength = 20
    potential += absorbing_potential(xgrid, absorb_strength, absorb_width)

    dot_height = 100
    dot_radius = 0.25
    potential += triangle_potential(xgrid, 2, dot_height, dot_radius)
    return potential
end

function wavefunction_to_image(xgrid, Ψ, potential, maxval=0.0)
    if maxval == 0.0
        maxval = maximum(abs.(real(Ψ)))
    end
    max_pot = maximum(real(potential))
    img = get(ColorSchemes.gray1, real(potential) / max_pot)
    
    img .+= get(ColorSchemes.berlin, 0.5 * (real(Ψ) / maxval .+ 1))
    # Convert to 8-bit channels. Clamping is needed since values over 1.0
    # cannot be represented in fixed point format.
    return map(img) do pixel
        convert(RGB{N0f8}, mapc((c)->clamp(c, 0.0, 1.0), pixel))
    end
end

function save_animation(fname, xgrid, potential, Ψs, ts,
        duration=10,
        constant_colormap=false)
    # Figure out how many frames to render.
    max_fps = 40
    num_frames = length(ts)
    skip = 1
    fps = round(Int, num_frames / duration)
    while fps > max_fps
        # Instead of rendering each frame, only render half.
        skip *= 2
        num_frames ÷= 2
        fps = round(Int, num_frames / duration)
    end

    maxval = if constant_colormap maximum(real(Ψs)) else 0.0 end

    imgstack = (
        wavefunction_to_image(xgrid, Ψs[:, :, i], potential, maxval)
        for i ∈ 1:skip:length(ts)
    )
    VideoIO.save(fname, imgstack, framerate=fps)
end

function simple_anim()
    L = 20
    Nx = 1000
    T = 5
    dt = T / 150
    momentum = [5, 0]
    xgrid = LinRange(-L / 2, L / 2, Nx)

    # Build initial wavefunction
    Ψ = gaussian_packet(xgrid, [0, 0], momentum, 1)

    potential = simple_potential(xgrid)
    Ψs, ts = time_evolve(xgrid, potential, Ψ, T, dt)
    num_steps = size(Ψs)[2]

    fps = 15
    dur = 10
    animate_evolution("packet_2d.mp4", xgrid, potential, Ψs, ts)
end

function compute_correlation_function(xgrid, Ψs)
    N = length(xgrid)
    Nx, Ny, Nt = size(Ψs)
    @assert N == Nx
    @assert N == Ny
    return [braket(xgrid, Ψs[:, :, 1], Ψs[:, :, i]) for i ∈ 1:Nt]
end

end # module BranchedFlowSim
