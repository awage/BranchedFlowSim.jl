# Comparing nonintegrable periodic potential and a (possibly?) integrable
# approximation of it.
using BranchedFlowSim
using CairoMakie
using Makie
using ColorTypes
using DataStructures
using ColorSchemes
using FFTW

complex_colorscheme = BranchedFlowSim.RealPartColor(BranchedFlowSim.sigmoid_berlin)


function make_prob_colorscheme()
    colors = reverse(ColorSchemes.linear_kryw_0_100_c71_n256.colors)
    # 
    # colors_with_alpha = [
    #     RGBA(c.r, c.g, c.b, a)
    #     for (c, a) ∈ zip(colors, LinRange(0, 1, length(colors)))
    # ]
    return ColorScheme(colors)
end

prob_colorscheme = make_prob_colorscheme()

function heatmap_with_potential(path, data, potential; kwargs...)
    scene = Scene(camera=campixel!, resolution=size(data))
    if potential !== nothing
        potential = real(potential)
        a = minimum(potential)
        b = maximum(potential)
        heatmap!(scene, potential, colorrange=(a, 2 * b), colormap=:grayC)
    end
    heatmap!(scene, data; colormap=prob_colorscheme.colors, kwargs...)
    save(path, scene)
end

function pixel_heatmap(path, data; kwargs...)
    scene = Scene(camera=campixel!, resolution=size(data))
    heatmap!(scene, data; kwargs...)
    save(path, scene)
end

function analyze(path_prefix, xgrid, Ψ, potential, Δt, num_steps, energies)
    pixel_heatmap(path_prefix * "_potential.png", real(potential))

    @time "eigenfunctions done" ΨE = collect_eigenfunctions(
        xgrid, Ψ, potential, Δt, num_steps, energies)
    for (ei, eigenfunc) ∈ enumerate(eachslice(ΨE, dims=3))
        E = eigenfunction_E[ei]
        save(
            path_prefix * "_eigenfunc_$(round(Int, E)).png",
            wavefunction_to_image(xgrid, eigenfunc,
                colorscheme=complex_colorscheme)
        )
        prob = abs.(eigenfunc) .^ 2
        pixel_heatmap(
            path_prefix * "_eigenfunc_$(round(Int, E))_prob.png",
            log.(prob),
            colormap=prob_colorscheme.colors,
            colorrange=(-9, 0))
    end
    return ΨE
end

scale = 4
# Simulation parameters
L = 20 * scale
num_grid_points = 1024
num_steps = 500 * scale
V0 = 100
maxE = 5 * V0
# Fix 
dt = 1 / maxE
T = dt * num_steps
# Potential parameters
lattice_constant = L / 20
dot_width = 0.2 * scale
dot_softness = 0.5
wall_strength = 100
# Initial wavefunction parameters
packet_pos = lattice_constant * [0.12, 0.23]
packet_momentum = [0, 0]
# Match Alvar's params
# packet_width = 0.1581
# This is kept constant w.r.t. `scale`, which can cause problems 
# due to spatial resolution.
packet_width = 0.08
# Output parameters
eigenfunction_E = LinRange(V0, maxE, 8)

xgrid = LinRange(-L / 2, L / 2, num_grid_points)

rm("outputs/square/", recursive=true, force=true)
mkpath("outputs/square/")

Ψ_initial = gaussian_packet(xgrid, packet_pos, packet_momentum, packet_width)

absorbing_walls = absorbing_potential(xgrid, wall_strength, 2)
square_potential = lattice_potential(xgrid, lattice_constant * [1 0; 0 1], 1,
    dot_width;
    offset=lattice_constant * [0.5, 0.5],
    dot_softness=dot_softness)

# analyze("outputs/free", xgrid, Ψ_initial, absorbing_walls, eigenfunction_E)

for approx ∈ [0, 5, 9]
    # for approx ∈ [0]
    println("Running for approx=$(approx)")

    path_prefix = "outputs/square/square$(approx)"
    potential = copy(square_potential)
    if approx != 0
        fpot = fft(potential)
        fvec = vec(fpot)
        # Only keep top terms
        top = approx
        perm = reverse(sortperm(abs.(fvec)))[1:top]
        new_fpot = zero(fpot)
        new_fpot[perm] = fpot[perm]
        potential = real(ifft(new_fpot))
    end

    potential *= V0
    potential += absorbing_walls

    analyze(path_prefix, xgrid, Ψ_initial, potential,
        dt, num_steps,
        eigenfunction_E)
end
