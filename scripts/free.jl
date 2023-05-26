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
    return ColorScheme(colors)
end

prob_colorscheme = make_prob_colorscheme()

function pixel_heatmap(path, data; kwargs...)
    scene = Scene(camera=campixel!, resolution=size(data))
    heatmap!(scene, data; kwargs...)
    save(path, scene)
end

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
        for (ei,E) ∈ enumerate(energies)
            @views ΨE[:, :, ei] .+= Ψ .* exp(1im * t * E)
        end
    end
    # Normalize eigenfunctions
    for Ψ ∈ eachslice(ΨE, dims=3)
        Ψ ./= sqrt(total_prob(xgrid, Ψ))
    end
    return ΨE
end


# Simulation parameters
L = 20
num_grid_points = 1024
V0 = 100
maxE = 5 * V0
num_steps = 400
dt = 1 / maxE
T = dt * num_steps
# Potential parameters
wall_strength = 100
# Initial wavefunction parameters
packet_pos = [0, 0]
packet_momentum = [0, 0]
# Match Alvar's params
# packet_width = 0.1581
packet_width = 0.05 * 10
# Output parameters
eigenfunction_E = LinRange(V0, maxE, 10)

xgrid = LinRange(-L / 2, L / 2, num_grid_points)

path_prefix = "outputs/free"
mkpath("outputs/")

Ψ = gaussian_packet(xgrid, packet_pos, packet_momentum, packet_width)
potential = absorbing_potential(xgrid, wall_strength, 2)
# This consumes a lot of memory, free up what we can beforehand to escape
# the OOM killer.
GC.gc()
pixel_heatmap(path_prefix * "_potential.png", real(potential))
@time "time evolution" Ψs, ts = time_evolution(xgrid, potential, Ψ, T, dt)
print("Start <Ψ|Ψ>=$(total_prob(xgrid, Ψs[:,:,1]))\n")
print("End <Ψ|Ψ>=$(total_prob(xgrid, Ψs[:,:,end]))\n")

@time "video done" save_animation(
    path_prefix * ".mp4", xgrid, potential, Ψs, duration=20, constant_colormap=false)

# Also compute an eigenfunction for some energy.
ΨE = compute_eigenfunctions(xgrid, Ψs, ts, eigenfunction_E)
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
        prob,
        colormap=prob_colorscheme.colors,
        colorrange=(0, 0.03))
end