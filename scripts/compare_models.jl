#=
Code to simulate the same potential using 3 models and compare the results.
=#
using ColorSchemes
using BranchedFlowSim
using Makie
using Colors
using CairoMakie
using FFTW

# Speed up quantum sim using multiple threads
FFTW.set_num_threads(8)

function quantum_visualize(xs, ys, pot, ħ)
    Nx = length(xs)
    Ny = length(ys)
    width = Nx * (xs[2] - xs[1])
    height = Ny * (ys[2] - ys[1])

    # Double the width of the grid and only keep the second part.
    # This allows us to put the absorbing wall in the first half.
    Nx *= 2
    xs = LinRange(0, 2width, Nx + 1)[1:end-1]
    qm_pot = zeros(ComplexF64, Ny, Nx)
    qm_pot += grid_eval(xs, ys, pot)
    # Create an absorbing wall in the first half of the 
    absorbing_wall_width = 0.3
    absorbing_wall_strength = 3
    # Add absorbing walls
  #   absorbing_profile = -1im * absorbing_wall_strength .*
  #                       (max.(0, (absorbing_wall_width .- xs) / absorbing_wall_width) .^ 2 .+
  #                        max.(0, (xs .- (2width - absorbing_wall_width)) / absorbing_wall_width) .^ 2)
    absorbing_profile = zeros(ComplexF64, length(xs))
    abs_cells = round(Int, absorbing_wall_width/(xs[2]-xs[1]))
    absorbing_profile[1:abs_cells] = -1im * absorbing_wall_strength * LinRange(0,1,abs_cells).^2
    absorbing_profile[abs_cells+1:2abs_cells] = -1im * absorbing_wall_strength * LinRange(1,0,abs_cells).^2

    px = 1
    x0 = width
    E = 0.5
    packet_Δx = 0.1
    T = 2 * width
    dt = ħ / E
    num_steps = round(Int, T / dt)
    # Make planar gaussian packet moves to the right
    Ψ_initial = ones(Ny) * transpose(exp.(-((xs .- x0) ./ (packet_Δx * √2)) .^ 2 +
                                          1im * px * xs / ħ))
    Ψ_initial ./= sqrt(total_prob(xs, ys, Ψ_initial))

    qm_pot += ones(Ny) * transpose(absorbing_profile)
    evolution = time_evolution(xs, ys, qm_pot, Ψ_initial, dt, num_steps, ħ)

    @time "eigenfunctions" ΨE = collect_eigenfunctions(evolution, [E],
        window=false)
    Ψ_prob = abs.(ΨE[:, :, 1]) .^ 2

    img_data = Ψ_prob[:,Nx÷2:end]
    return heatmap_with_potential(img_data, real(qm_pot[:,Nx÷2:end]))
end

function heatmap_with_potential(data, potential; colorrange=extrema(data))
    fire = reverse(ColorSchemes.linear_kryw_0_100_c71_n256)
    pot_colormap = ColorSchemes.grays
    # data_colormap = ColorSchemes.viridis
    data_colormap = fire

    V = real(potential)
    minV, maxV = extrema(V)
    minD, maxD = colorrange
    print("datarange: [$(minD),$(maxD)]\n")
    pot_img = get(pot_colormap, (V .- minV) ./ (maxV - minV + 1e-9))
    data_img = get(data_colormap, (data .- minD) / (maxD - minD))
    
    img = mapc.((d,v) -> clamp(d - 0.2 * v, 0, 1), data_img, pot_img) 
    return img
end

v0 = 0.04

width = 2
height = 1
pot = correlated_random_potential(width, height, 0.1, v0, 0)
Nx = 2000
Ny = round(Int, Nx * height / width)

xs = LinRange(0, width, Nx + 1)[1:end-1]
ys = LinRange(0, height, Ny + 1)[1:end-1]
num_rays = 5000

## Classical
r0 = [0.0, 1.0] * LinRange(0, 1, num_rays)'
p0 = [1.0, 0.0] * ones(num_rays)'
for i ∈ 1:num_rays
    p0[:, i] = normalized_momentum(pot, r0[:, i], p0[:, i])
end
println("p0=$p0")
img_cl = BranchedFlowSim.classical_visualize_rays(xs, ys, pot, r0, p0, 2 * width)

## Quantum 

ħ = 0.0005
qm_img = quantum_visualize(xs, ys, pot, ħ)

## Quasi-2D

# This is slower than necessary, really don't need the smoothness stuff.
int,(rmin,rmax) = quasi2d_intensity(num_rays, 0.005, xs, ys, pot)
quasi_img = heatmap_with_potential(int, grid_eval(xs, ys, pot))

## Plot

fig = Figure()
ax_qm = Axis(fig[1, 1], aspect=DataAspect())
ax_c = Axis(fig[2, 1], aspect=DataAspect())
ax_quasi = Axis(fig[3, 1], aspect=DataAspect())
hidedecorations!(ax_qm)
hidedecorations!(ax_c)
hidedecorations!(ax_quasi)

image!(ax_qm, xs, ys, qm_img')
image!(ax_c, xs, ys, img_cl')
image!(ax_quasi, xs, ys, quasi_img')

display(fig)
save("outputs/compare_models.pdf", fig)
