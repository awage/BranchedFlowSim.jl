module BranchedFlowSim

# Helpers
export fermi_dot, gaussian_packet, absorbing_potential, lattice_points_in_a_box, lattice_potential, triangle_potential
export braket
export total_prob

# Simulation and analysis
export SplitOperatorStepper, timestep!, time_evolution
export compute_energy_spectrum, compute_eigenfunctions

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

include("colormap.jl")

const mass::Float64 = 1.0
# colorscheme = colorscheme_daza
colorscheme = ColorSchemes.berlin

"""
    fermi_step(x, α)

Smooth step function, i.e. goes from 0 to 1 as x goes from negative to
positive. α is a term controlling how smooth the transition from 0 to 1 is.
Equal to Fermi-Dirac distribution where α=kT, so large values of α correspond
to smooth transition and small values to an abrupt one.
"""
function fermi_step(x, α)
    return 1 / (1 + exp(-x / α))
end

"""
    fermi_dot(xgrid, pos, radius, softness=1)

Return a potential function matrix of size length(xgrid)×length(xgrid) with
value tending to 1 inside a circle of given position and radius.
"""
function fermi_dot(xgrid, pos, radius, softness=1)
    # N x N matrix of distance from `pos`
    dist = sqrt.(((xgrid .- pos[2]) .^ 2) .+ transpose(((xgrid .- pos[1]) .^ 2)))
    α = radius / (5 * softness)
    return fermi_step.(radius .- dist, α)
end

"""
    gaussian_packet(xgrid, pos, momentum, Δ)

Returns a 2D gaussian packet with given mean position and momentum. Δ is
standard deviation from the center along one axis.
"""
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

"""
    total_prob(xgrid, Ψ)

Returns total probability of Ψ, i.e. <Ψ|Ψ>.
"""
function total_prob(xgrid, Ψ)
    return trapz((xgrid, xgrid), abs.(Ψ) .^ 2)
end

"""
    absorbing_potential(xgrid, strength, width)

Returns a length(xgrid)×length(xgrid) sized matrix representing an absorbing
potential. `strength` is the maximum absolute value of the potential and
`width` is the width of the absorbing wall in same units as `xgrid`.
"""
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

Type for efficient time propagation of 2D quantum wavefunctions. 
"""
struct SplitOperatorStepper
    dt::Float64
    V_step::Matrix{ComplexF64}
    T_step::Matrix{ComplexF64}

    prepared_fft::Any
    prepared_ifft::Any

    """
    SplitOperatorStepper(xgrid, Δt, potential)

Creates a stepper object for a 2D grid where coordinates are discretized
according to `xgrid` (in both x and y coordinates). Applying this stepper with
`timestep!` advances the picture by `Δt`.
"""
function SplitOperatorStepper(xgrid :: AbstractVector{Float64}, Δt :: Float64, potential)
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
        my_ifft = plan_ifft!(psi_placeholder)
        return new(Δt, V_step, T_step, my_fft, my_ifft)
    end
end

"""
    timestep!(Ψ, stepper)

Perform a time step using given SplitOperatorStepper. This operation mutates Ψ.
"""
function timestep!(Ψ, stepper :: SplitOperatorStepper)
    @assert size(Ψ) == size(stepper.V_step) "$(size(Ψ)) != $(size(stepper.V_step))"
    # Potential step in x-space:
    Ψ .*= stepper.V_step
    # Kinetic step in p-space:
    stepper.prepared_fft * Ψ
    Ψ .*= stepper.T_step
    stepper.prepared_ifft * Ψ
end

"""
    time_evolution(xgrid, potential, Ψ_initial, T, Δt)

Performs time evolution for a 2D quantum system from t=0 to t=T in steps of Δt.
System is discretized in coordinate space according to `xgrid``.
"""
function time_evolution(xgrid, potential, Ψ_initial, T, Δt)
    N = length(xgrid)
    stepper = SplitOperatorStepper(xgrid, Δt, potential)
    # Make a copy since we'll be modifying Ψ in-place.
    Ψ = copy(Ψ_initial)
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
    lattice_points_in_a_box(A, L)

Returns all lattice points in L-by-L sized box (2D) centered at (0,0). A is a
2×2 matrix with generating vectors of the lattice as the columns. Returns a
matrix with 2 rows and one column per lattice point.
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

"""
    lattice_potential(xgrid, A, dot_height, dot_radius, offset=[0, 0],
    dot_softness=1)

Returns a potential matrix.
"""
function lattice_potential(xgrid, A, dot_height, dot_radius, offset=[0, 0],
    dot_softness=1)
    N = length(xgrid)
    L = (xgrid[end] - xgrid[1]) + 10 * dot_radius + abs(maximum(offset))
    V = zeros(N, N)
    for point ∈ eachcol(lattice_points_in_a_box(A, L))
        V .+= fermi_dot(xgrid, point + offset, dot_radius, dot_softness)
    end
    return V * dot_height
end


"""
    triangle_potential(xgrid, a, dot_height, dot_radius=0.1 * a, dot_softness=1)

Convenience function for creating a lattice potential with a triangle lattice.
"""
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

"""
    wavefunction_to_image(xgrid, Ψ, potential, maxval=0.0)

Produces a RGB image showing given wavefunction Ψ and potential. If `maxval` is
nonzero, Ψ values greater or equal to `maxval` are mapped to highest colors.
"""
function wavefunction_to_image(xgrid, Ψ, potential; maxval=0.0)
    if maxval == 0.0
        maxval = maximum(abs.(real(Ψ)))
    end
    max_pot = maximum(real(potential))
    img = get(ColorSchemes.gray1, real(potential) / max_pot)

    img .+= get(colorscheme, 0.5 * (real(Ψ) / maxval .+ 1))
    # Convert to 8-bit channels. Clamping is needed since values over 1.0
    # cannot be represented in fixed point format.
    return map(img) do pixel
        convert(RGB{N0f8}, mapc((c) -> clamp(c, 0.0, 1.0), pixel))
    end
end

"""
    save_animation(fname, xgrid, potential, Ψs, ts;
    duration=10,
    constant_colormap=false)

Produces an animation and stores it to a file at `fname`. `Ψs` should be a
length(xgrid)×length(xgrid)×length(ts) matrix and `ts` the 
"""
function save_animation(fname, xgrid, potential, Ψs;
    duration=10,
    constant_colormap=false)
    # Figure out how many frames to render.
    max_fps = 40
    num_steps = size(Ψs)[3]
    num_frames = num_steps
    skip = 1
    fps = round(Int, num_frames / duration)
    while fps > max_fps
        # Instead of rendering each frame, only render half.
        skip *= 2
        num_frames ÷= 2
        fps = round(Int, num_frames / duration)
    end

    maxval = if constant_colormap
        maximum(real(Ψs[:,:, 1 + num_steps ÷ 10]))
    else
        0.0
    end

    imgstack = (
        wavefunction_to_image(xgrid, Ψs[:, :, i], potential, maxval=maxval)
        for i ∈ 1:skip:num_steps
    )
    VideoIO.save(fname, imgstack, framerate=fps,
        encoder_options=(crf=23, preset="fast"))
end

function compute_correlation_function(xgrid, Ψs)
    N = length(xgrid)
    Nx, Ny, Nt = size(Ψs)
    @assert N == Nx
    @assert N == Ny
    return [braket(xgrid, Ψs[:, :, 1], Ψs[:, :, i]) for i ∈ 1:Nt]
end

function compute_energy_spectrum(xgrid, Ψs, ts)
    dt = ts[2] - ts[1]
    pt = compute_correlation_function(xgrid, Ψs)
    # pt .*= 1 .- cos.((2π / T) * ts)

    Es = fftshift(fftfreq(length(ts), 1 / dt))
    pE = fftshift(ifft(pt))
    return pE, Es
end

function compute_eigenfunctions(xgrid, Ψs, ts, Es)
    Nx = size(Ψs)[1]
    ΨE = zeros(ComplexF64, (Nx, Nx, length(Es)))
    T = ts[end]
    for (i, t) ∈ enumerate(ts)
        w = 1 - cos(2pi * t / T)
        w = 1
        for (j, E) ∈ enumerate(Es)
            ΨE[:, :, j] .+= Ψs[:, :, i] * w * exp(1im * t * E)
        end
    end
    # Normalize each computed eigenfunction
    for Ψ ∈ eachslice(ΨE, dims=3)
        Ψ ./= sqrt(total_prob(xgrid, Ψ))
    end
    return ΨE
end

# TODO:
# - docstrings
# - General commenting and cleanliness

end # module BranchedFlowSim
