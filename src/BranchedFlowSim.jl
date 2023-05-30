module BranchedFlowSim

# Helpers
export gaussian_packet, braket
export total_prob
export gaussian_correlated_random

# Simulation and analysis
export SplitOperatorStepper, timestep!, time_evolution
export compute_energy_spectrum, compute_eigenfunctions
export collect_eigenfunctions

# Visualization
export wavefunction_to_image, save_animation
export make_animation

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

include("colorschemes.jl")
include("potentials.jl")

default_colorscheme = complex_berlin

const mass::Float64 = 1.0

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
    total_prob(xgrid, ygrid, Ψ)

Returns total probability of Ψ, i.e. <Ψ|Ψ>.
"""
function total_prob(xgrid, ygrid, Ψ)
    return trapz((ygrid, xgrid), abs.(Ψ) .^ 2)
end

"""
    total_prob(xgrid, Ψ)

Returns total probability of Ψ, i.e. <Ψ|Ψ>.
"""
function total_prob(xgrid, Ψ)
    return total_prob(xgrid, xgrid, Ψ)
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
    SplitOperatorStepper(
        xgrid::AbstractVector{Float64},
        ygrid::AbstractVector{Float64},
         Δt::Float64, potential,
        hbar::Float64=1.0)

TBW
"""
    function SplitOperatorStepper(
        xgrid::AbstractVector{Float64},
        ygrid::AbstractVector{Float64},
        Δt::Float64, potential,
        hbar::Real=1.0)
        Nx = length(xgrid)
        Ny = length(ygrid)
        @assert size(potential) == (Ny, Nx)
        dx = xgrid[2] - xgrid[1]
        dy = ygrid[2] - ygrid[1]
        # Find momentum values corresponding to each fft value
        px = 2π * fftfreq(Nx, 1 / dx)
        py = 2π * fftfreq(Ny, 1 / dy)
        # Kinetic operator is diagonal in p space:
        p2 = (py .^ 2) .+ transpose(px .^ 2)
        T_step = exp.(-Δt .* hbar * 1im .* p2 / 2mass)
        # Potential operator is diagonal in x space:
        V_step = exp.(-1im .* Δt .* potential / hbar)

        # Prepare in-place FFT to make repeated application faster.
        psi_placeholder = zeros(ComplexF64, Ny, Nx)
        my_fft = plan_fft!(psi_placeholder)
        my_ifft = plan_ifft!(psi_placeholder)
        return new(Δt, V_step, T_step, my_fft, my_ifft)
    end
    """
    SplitOperatorStepper(xgrid, Δt, potential)

Creates a stepper object for a 2D grid where coordinates are discretized
according to `xgrid` (in both x and y coordinates). Applying this stepper with
`timestep!` advances the picture by `Δt`.
"""
    function SplitOperatorStepper(xgrid::AbstractVector{Float64}, Δt::Float64, potential,
        hbar::Float64=1.0)
        return SplitOperatorStepper(xgrid, xgrid, Δt, potential, hbar)
    end
end

"""
    timestep!(Ψ, stepper)

Perform a time step using given SplitOperatorStepper. This operation mutates Ψ.
"""
function timestep!(Ψ, stepper::SplitOperatorStepper)
    @assert size(Ψ) == size(stepper.V_step) "$(size(Ψ)) != $(size(stepper.V_step))"
    # Potential step in x-space:
    Ψ .*= stepper.V_step
    # Kinetic step in p-space:
    stepper.prepared_fft * Ψ
    Ψ .*= stepper.T_step
    stepper.prepared_ifft * Ψ
end

struct TimeEvolution
    stepper::SplitOperatorStepper
    dt::Float64
    num_steps::Int64
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    potential::Matrix{ComplexF64}
    Ψ_initial::Matrix{ComplexF64}
end

function time_evolution(
    xgrid::AbstractVector{Float64},
    ygrid::AbstractVector{Float64},
    potential, Ψ, dt, num_steps, hbar=1.0)
    stepper = SplitOperatorStepper(xgrid, ygrid, dt, potential, hbar)
    return TimeEvolution(stepper, dt, num_steps, xgrid, ygrid, potential, Ψ)
end

function Base.iterate(it::TimeEvolution)
    return ((0.0, it.Ψ_initial), (copy(it.Ψ_initial), 0))
end

function Base.iterate(it::TimeEvolution, (Ψ, step))
    if step == it.num_steps
        return nothing
    end
    timestep!(Ψ, it.stepper)
    return ((step + 1) * it.dt, copy(Ψ)), (Ψ, step + 1)
end

function Base.length(it::TimeEvolution)
    return 1 + it.num_steps
end


"""
    time_evolution(xgrid, potential, Ψ_initial, T, Δt, hbar=1.0)

Performs time evolution for a 2D quantum system from t=0 to t=T in steps of Δt.
System is discretized in coordinate space according to `xgrid``.
"""
function time_evolution(xgrid, potential, Ψ_initial, T, Δt, hbar=1.0)
    N = length(xgrid)
    stepper = SplitOperatorStepper(xgrid, Δt, potential, hbar)
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
    wavefunction_to_image(xgrid, Ψ, potential, max_modulus=0.0)

Produces a RGB image showing given wavefunction Ψ and potential. If `max_modulus` is
nonzero, Ψ values greater or equal to `max_modulus` are mapped to highest colors.
"""
function wavefunction_to_image(Ψ;
    potential=nothing,
    max_modulus=0.0,
    colorscheme=default_colorscheme
)
    if max_modulus == 0.0
        max_modulus = maximum(abs.(real(Ψ)))
    end
    img = zeros(RGB{Float64}, size(Ψ))
    if potential !== nothing
        min_pot = minimum(real(potential))
        max_pot = maximum(real(potential)) + 1e-8
        img .+= get(ColorSchemes.gray1,
            0.5 * (real(potential) .- min_pot) / (max_pot - min_pot))
    end

    img .+= get(colorscheme, Ψ / max_modulus)
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
    constant_colormap=false,
    colorscheme=default_colorscheme)
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

    max_modulus = if constant_colormap
        maximum(abs(Ψs[:, :, 1+num_steps÷10]))
    else
        0.0
    end

    imgstack = (
        wavefunction_to_image(Ψs[:, :, i]; potential=potential, max_modulus=max_modulus,
            colorscheme=colorscheme)
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
        # w = 1 - cos(2pi * t / T)
        for (j, E) ∈ enumerate(Es)
            ΨE[:, :, j] .+= Ψs[:, :, i] .* exp(1im * t * E)
        end
    end
    # Normalize each computed eigenfunction
    for Ψ ∈ eachslice(ΨE, dims=3)
        Ψ ./= sqrt(total_prob(xgrid, Ψ))
    end
    return ΨE
end

function middle(xs)
    if length(xs) % 2 == 1
        return xs[1+end÷2]
    end
    return (xs[end÷2] + xs[1+end÷2]) / 2
end

function gaussian_correlated_random(xs, ys, scale)
    ymid = middle(ys)
    xmid = middle(xs)

    dist2 = ((ys .- ymid) .^ 2) .+ transpose((xs .- xmid) .^ 2)
    # Not sure why, but 
    corr = 2 * exp.(-dist2 / (scale^2))
    # XXX Explain this
    num_points = length(xs) * length(ys)
    fcorr = fft(corr) / num_points
    phase = rand(length(ys), length(xs))
    vrand = ifft(num_points * sqrt.(fcorr) .* exp.(im * 2pi * phase))
    return real(vrand)
end

"""
    collect_eigenfunctions(xgrid, Ψ_initial, potential, Δt, steps, energies)

Returns eigenfunctions for given `energies` by running a simulation for given
initial wavefunction, potential, Δt, and number of steps.
"""
function collect_eigenfunctions(xgrid, Ψ_initial, potential, Δt, steps, energies)
    stepper = SplitOperatorStepper(xgrid, Δt, potential)
    Ψ = copy(Ψ_initial)
    # Collect eigenfunctions
    ΨE = zeros(ComplexF64, length(xgrid), length(xgrid), length(energies))
    t = 0
    tmp = copy(Ψ)
    for step = 1:steps
        timestep!(Ψ, stepper)
        t += Δt
        Threads.@threads for ei ∈ 1:length(energies)
            E = energies[ei]
            @views ΨE[:, :, ei] .+= Ψ .* exp(1im * t * E)
        end
    end
    # Normalize eigenfunctions
    for Ψ ∈ eachslice(ΨE, dims=3)
        Ψ ./= sqrt(total_prob(xgrid, Ψ))
    end
    return ΨE
end

"""
    collect_eigenfunctions(evolution :: TimeEvolution, energies)

Returns eigenfunctions for given `energies` by running a given time evolution.
"""
function collect_eigenfunctions(evolution::TimeEvolution, energies;
    window=false)
    ΨE = zeros(ComplexF64, length(evolution.ygrid), length(evolution.xgrid),
        length(energies))
    T = evolution.num_steps * evolution.dt
    for (t, Ψ) ∈ evolution
        w = 1.0
        if window
            w = 1.0 - cos(t*2pi / T)
        end
        Threads.@threads for ei ∈ 1:length(energies)
            E = energies[ei]
            @views ΨE[:, :, ei] .+= Ψ .* (w * exp(1im * t * E))
        end
    end
    print("$(length(evolution.xgrid)) ---- $(length(evolution.ygrid))\n")
    # Normalize eigenfunctions
    for Ψ ∈ eachslice(ΨE, dims=3)
        print("$(size(Ψ))\n")
        prob = total_prob(evolution.xgrid, evolution.ygrid, Ψ)
        Ψ ./= sqrt(prob)
    end
    return ΨE
end

"""
    make_animation(fname, xgrid, potential, Ψs, ts;
    duration=10,
    constant_colormap=false)

Produces an animation and stores it to a file at `fname`. `Ψs` should be a
length(xgrid)×length(xgrid)×length(ts) matrix and `ts` the 
"""
function make_animation(fname, evolution::TimeEvolution, potential;
    duration=10,
    colorscheme=default_colorscheme,
    max_modulus=0.0
)
    # Figure out how many frames to render.
    max_fps = 40
    num_steps = length(evolution)
    num_frames = num_steps
    skip = 1
    fps = round(Int, num_frames / duration)
    while fps > max_fps
        # Instead of rendering each frame, only render half.
        skip *= 2
        num_frames ÷= 2
        fps = round(Int, num_frames / duration)
    end

    # XXX

    imgstack = (
        wavefunction_to_image(Ψ; potential=potential, max_modulus=max_modulus,
            colorscheme=colorscheme)
        for (t, Ψ) ∈ evolution
    )
    VideoIO.save(fname, imgstack, framerate=fps,
        encoder_options=(crf=23, preset="fast"))
end

end # module BranchedFlowSim
