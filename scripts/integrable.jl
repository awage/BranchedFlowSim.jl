# Comparing nonintegrable periodic potential and a (possibly?) integrable
# approximation of it.
using BranchedFlowSim
using CairoMakie
using Makie
using DataStructures
using ColorSchemes
using FFTW


# Simulation parameters
L = 20
num_grid_points = 1024
num_steps = 400
T = 0.5
dt = T / num_steps
# Potential parameters
lattice_constant = 1
dot_width = 0.2
dot_softness = 0.5
V0 = 100
# Initial wavefunction parameters
packet_pos = [0.2, 0.1]
packet_momentum = [0, 0]
packet_width = 0.05
# Output parameters
eigenfunction_E = vec(LinRange(200, 800, 10))
complex_colorscheme = BranchedFlowSim.RealPartColor(BranchedFlowSim.sigmoid_berlin)

xgrid = LinRange(-L / 2, L / 2, num_grid_points)

mkpath("outputs/")

function pixel_heatmap(path, data; kwargs...)
    scene = Scene(camera=campixel!, resolution=size(data))
    heatmap!(scene, data; kwargs...)
    save(path, scene)
end

square_potential = lattice_potential(xgrid, lattice_constant * [1 0; 0 1], 1,
    dot_width;
    offset=lattice_constant * [0.5, 0.5],
    dot_softness=dot_softness)

for do_approx ∈ [true, false]
# This consumes a lot of memory, free up what we can beforehand to escape the
# OOM killer.
GC.gc()
# do_approx = false
println("Running for do_approx=$(do_approx)")

path_prefix = "outputs/square"
potential = copy(square_potential)
if do_approx
    path_prefix = "outputs/square_approx"
    fpot = fft(potential)
    fvec = vec(fpot)
    # Only keep top terms
    top = 5
    perm = reverse(sortperm(abs.(fvec)))[1:top]
    new_fpot = zero(fpot)
    new_fpot[perm] = fpot[perm]
    potential = real(ifft(new_fpot))
end

potential *= V0
potential += absorbing_potential(xgrid, 100, 2)

pixel_heatmap(path_prefix * "_potential.png", real(potential))

Ψ = gaussian_packet(xgrid, packet_pos, packet_momentum, packet_width)

Ψs = nothing # Just to free memory
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
    intensity = abs.(eigenfunc).^2
    pixel_heatmap(
        path_prefix * "_eigenfunc_$(round(Int, E))_intensity.png",
        intensity,
        colormap=:lajolla,
        colorrange=(0, 0.03))
end

function get_flow(Ψs)
    return sum(
        abs.(Ψs[:, :, i]) .^ 2
        for i ∈ 1:size(Ψs)[3]
    )
end

# Produce an intensity picture
intensity = get_flow(Ψs)
pixel_heatmap(path_prefix * "_intensity.png",
    log.(intensity),
    colormap=:lajolla
)
end
