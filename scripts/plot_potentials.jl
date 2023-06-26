using BranchedFlowSim
using Makie
using CairoMakie

path_prefix="outputs/potentials/"
mkpath(path_prefix)

function random_dot_potential(xmin, xmax, ymin, ymax, lattice_a, dot_radius, v0)
    box_dims =
        num_dots = round(Int, (xmax - xmin) * (ymax - ymin) / lattice_a^2)
    locs = zeros(2, num_dots)
    for i ∈ 1:num_dots
        locs[:, i] = [xmin, ymin] + rand(2) .* [xmax - xmin, ymax - ymin]
    end
    return RepeatedPotential(
        locs,
        FermiDotPotential(dot_radius, v0),
        lattice_a
    )
end

function plot_potential(pot)
    res = 512
    xs = LinRange(0, 2, res)
    ys = LinRange(0, 2, res)
    g = grid_eval(xs, ys, pot)

    fig = Figure()
    ax = Axis(fig[1,1], aspect=DataAspect())
    ming, maxg = extrema(g)
    mintick = round(Int, ming)
    maxtick = round(Int, maxg)
    tickpoints = mintick:maxtick
    if maxtick - mintick < 2
        tickpoints = mintick:0.5:maxtick
    end

    hm = heatmap!(xs, ys, g, interpolate=true,
        colorrange=(min(mintick, ming), max(maxtick, maxg))
    )
    ticks = (
        tickpoints,
        [if x != 0 L"%$(x)v_0" else L"0" end for x ∈ tickpoints])
    cb = Colorbar(fig[1,2], hm, ticks = ticks)
        # Makie.MultiplesTicks(5, 1, "v₀"))
    return fig
end

# Keep a seed to keep the same potential
seed = 123

rpot = correlated_random_potential(3, 3, 0.1, 1, seed)
save(path_prefix*"correlated_random.pdf", plot_potential(rpot))

lattice_a = 0.2
dot_radius = 0.25 * lattice_a
lpot = LatticePotential(rotation_matrix(0)*lattice_a, dot_radius, 1)
save(path_prefix*"fermi_lattice.pdf", plot_potential(lpot))

fermi_rand= 
    random_dot_potential(-1, 5, -1, 5, lattice_a, dot_radius, 1)
save(path_prefix*"fermi_random.pdf", plot_potential(fermi_rand))

for deg ∈ 1:6
    pot = fermi_dot_lattice_cos_series(
        deg, lattice_a, dot_radius, 1
    )
    save(path_prefix*"cos_series_$deg.pdf", plot_potential(pot))
end
