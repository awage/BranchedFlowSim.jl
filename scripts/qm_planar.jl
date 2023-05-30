using BranchedFlowSim
using CairoMakie
using Makie

path_prefix = "outputs/qm_planar"

ħ = 0.5
aspect_ratio = 5

Ny = 256
Nx = aspect_ratio * Ny

H = 2
W = H * aspect_ratio

px = 10
x0 = 1.5
packet_Δx = 0.25
v0 = 20
# These are tuned to get rid of reflections
absorbing_wall_width = 0.75
absorbing_wall_strength = 300

with_walls = false

if !with_walls
    path_prefix = path_prefix * "_periodic"
end

function pixel_heatmap(path, data; kwargs...)
    data = transpose(data)
    scene = Scene(camera=campixel!, resolution=size(data))
    heatmap!(scene, data; kwargs...)
    save(path, scene)
end

E = px^2 / 2
print("E=$(E)\n")

@assert W / Nx == H / Ny

xgrid = range(0, step=W / Nx, length=Nx)
ygrid = range(0, step=H / Ny, length=Ny)

potential = zeros(ComplexF64, Ny, Nx)

xstart = if with_walls
    3
else
    1
end
for x ∈ xstart:W
    for y ∈ 1:H
        p = [x - 0.5, y - 0.5]
        add_fermi_dot!(potential, xgrid, ygrid, p, 0.25)
    end
end
potential *= v0
# potential *= 0.0
# Make absorbing walls on the left and the right
if with_walls
    absorbing_profile = -1im * absorbing_wall_strength .*
                        (max.(0, (absorbing_wall_width .- xgrid) / absorbing_wall_width) .^ 2 .+
                         max.(0, (xgrid .- (W - absorbing_wall_width)) / absorbing_wall_width) .^ 2)
    potential += ones(Ny) * transpose(absorbing_profile)
end


dt = 1 / (3 * E)
num_steps = if with_walls 400 else 1000 end

# Make planar gaussian packet moves to the right
Ψ_initial = ones(Ny) * transpose(exp.(-((xgrid .- x0) ./ (packet_Δx * √2)) .^ 2 + 1im * px * xgrid / ħ))
Ψ_initial ./= sqrt(total_prob(xgrid, ygrid, Ψ_initial))

pixel_heatmap(path_prefix * "_potential.png", real(potential))
pixel_heatmap(path_prefix * "_initial_real.png", real(Ψ_initial))

evolution = time_evolution(xgrid, ygrid, potential, Ψ_initial, dt, num_steps, ħ)

@time "making animation" make_animation(path_prefix * ".mp4", evolution, potential,
    max_modulus=0.5)

energies = LinRange(v0, 5 * v0, 5)
@time "eigenfunctions" ΨE = collect_eigenfunctions(evolution, energies,
    window=!with_walls)
for (ei, E) ∈ enumerate(energies)
    pixel_heatmap(path_prefix * "_eigenfunc_$(round(Int, E)).png",
        abs.(ΨE[:, :, ei]) .^ 2)
end