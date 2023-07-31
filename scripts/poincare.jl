using BranchedFlowSim
using CairoMakie
using LinearAlgebra
using Colors
using LaTeXStrings
using FixedPointNumbers
using Printf
using Makie
using ColorSchemes
using ColorTypes

function normalize_momentum!(p, r, potential, E=0.5)
    @assert size(p) == size(r)
    for i ∈ 1:size(p)[2]
        V = potential(r[1, i], r[2, i])
        pnorm = sqrt(2 * (E - V))
        @assert isreal(pnorm)
        p[:, i] .*= pnorm / norm(p[:, i])
    end
end

function potential_heatmap(xs, ys, potential)
    V = grid_eval(xs, ys, potential)
    minV, maxV = extrema(V)
    pot_colormap = reverse(ColorSchemes.grays)
    return get(pot_colormap, 0.4 * (V .- minV) ./ (maxV - minV + 1e-9))
end

function heatmap_with_potential(xs, ys, data, potential; colorrange=extrema(data))
    fire = reverse(ColorSchemes.linear_kryw_0_100_c71_n256)
    pot_colormap = ColorSchemes.grays
    # data_colormap = ColorSchemes.viridis
    data_colormap = fire

    V = grid_eval(xs, ys, potential)
    minV, maxV = extrema(V)
    minD, maxD = colorrange

    pot_img = get(pot_colormap, (V .- minV) ./ (maxV - minV + 1e-9))
    data_img = get(data_colormap, (data .- minD) / (maxD - minD))

    img = mapc.((d, v) -> clamp(d - 0.2 * v, 0, 1), data_img, pot_img)
    return img
end

function make_explanation_figure(pot, step, intersect, offset, angles)
    num_particles = size(angles)
    r0 = lattice_a * [0.1, 0.05] * ones(num_particles)'
    p0 = hcat(([cos(θ), sin(θ)] for θ ∈ angles)...)
    normalize_momentum!(p0, r0, pot)

    T = 20.0

    fig = Figure(resolution=(1024, 512))

    M = 1.0
    xs = LinRange(-M, M, 512)

    ax_traj = Axis(fig[1, 1], limits=((-1, 1), (-1, 1)),
        aspect=DataAspect(),
        xlabel=L"$x",
        ylabel=L"$y"
    )
    ax_section = Axis(fig[1, 2],
        xlabel=L"$q",
        ylabel=L"$w",
        limits=((0, 1), (-1, 1))
    )

    pot_img = potential_heatmap(xs, xs, pot)
    image!(ax_traj, xs, xs, pot_img)

    for k ∈ -10:10
        lines!(ax_traj,
            hcat(offset + k * step - 100 * intersect, offset + k * step + 100 * intersect),
            color=:black,
            linestyle=:dot
        )
        lines!(ax_traj,
            hcat(offset + k * intersect - 100 * step, offset + k * intersect + 100 * step),
            color=:gray,
            linestyle=:dash
        )
    end
    dt = 0.005

    palette_i = 1
    palette = Makie.wong_colors()
    ray_trajectories(r0, p0, pot, T, dt) do ts, rs, ps
        color = palette[palette_i]
        palette_i += 1
        lines!(ax_traj, rs, color=color)
        lattice_intersections(intersect, step, offset, rs, ps) do r, p, q, w
            if norm(r) < 2
                scatter!(ax_traj, r, color=:red)
                text!(ax_traj, r, text=@sprintf("q=%.2f\nw=%.2f", q, w), fontsize=12)
            end
            scatter!(ax_section, q, w, color=color, markersize=3)
        end
    end

    return fig
end

function make_trajectory_plot(pot, step, intersect, offset, r0, p0, T)
    num_particles = size(r0)[2]
    normalize_momentum!(p0, r0, pot)

    fig = Figure(resolution=(1024, 1024), figure_padding=0)
    ax_section = Axis(fig[2, 1],
        xlabel=L"$q",
        ylabel=L"$w",
        limits=((0, 1), (-1.0, 1.0)),
        aspect=2
    )

    hist_width = 512
    section_width = 512
    xs = LinRange(0, 8, hist_width)
    ys = LinRange(-2, 2, hist_width)
    qs = LinRange(0, 1, section_width)
    ws = LinRange(-1, 1, section_width)

    hist = zeros(hist_width, hist_width)
    section_img = fill(RGB{N0f8}(1, 1, 1), section_width, section_width)
    ray_img = potential_heatmap(ys, xs, pot)

    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    dq = qs[2] - qs[1]
    dw = ws[2] - ws[1]

    dt = 0.005

    traj_i = 1
    ray_trajectories(r0, p0, pot, T, dt) do ts, rs, ps
        color = convert(RGB{N0f8}, HSV(traj_i * 360 / num_particles, 1, 0.7))
        traj_i += 1
        lattice_intersections(intersect, step, offset, rs, ps) do r, p, q, w
            qi = round(Int, 1 + (q - qs[1]) / dq)
            wi = round(Int, 1 + (w - ws[1]) / dw)
            if 1 < qi < section_width && 1 < wi < section_width
                section_img[qi, wi] = color
            end
            # scatter!(ax_section, q, w, color=color, markersize=1)
        end
        lx::Float64 = 0.0
        ly::Float64 = 0.0
        for (x, y) ∈ eachcol(rs)
            xi = 1 + (x - xs[1]) / dx
            yi = 1 + (y - ys[1]) / dy
            if 1 < xi < hist_width && 1 < yi < hist_width &&
               1 < lx < hist_width && 1 < ly < hist_width
                first = true
                line_integer_cells(lx, ly, xi, yi) do x, y
                    if !first
                        hist[x, y] += 1
                        ray_img[x, y] = color
                    end
                    first = false
                end
            end
            lx = xi
            ly = yi
        end
    end
    ax_traj = Axis(fig[1, 1], limits=((xs[1], xs[end]), (ys[1], ys[end])),
        aspect=DataAspect(),
        xlabel=L"$x",
        ylabel=L"$y"
    )
    hist_img = heatmap_with_potential(xs, ys, hist, pot,
        colorrange=(0.0, 2.0 * num_particles / 100)
    )
    # image!(ax_traj, xs, xs, hist_img)
    image!(ax_traj, xs, ys, ray_img)
    image!(ax_section, qs, ws, section_img)
    hidedecorations!(ax_traj)
    hidedecorations!(ax_section)
    hidespines!(ax_traj)
    hidespines!(ax_section)
    colgap!(fig.layout, 0.0)
    rowgap!(fig.layout, 0.0)
    return fig
end

lattice_a::Float64 = 0.2
dot_radius::Float64 = 0.25 * lattice_a
v0::Float64 = 0.05
softness = 0.2
num_particles = 30
T = 500

offset = [lattice_a / 2, lattice_a / 2]
# pot = fermi_dot_lattice_cos_series(1, lattice_a, dot_radius, -v0)

intersect = [0.0, 0.2]
step = [0.2, 0.0]
# angles = LinRange(-pi/8, pi/8, num_particles + 1)[1:end-1]
# r0 = lattice_a * [0.1, 0.05] * ones(num_particles)'
# p0 = hcat(([cos(θ), sin(θ)] for θ ∈ angles)...)
r0 = lattice_a * [0.0, 1] * LinRange(-0.5, 0.5, num_particles)'
p0 = [1.0, 0] * ones(num_particles)'

# Integrable and almost integrable potentials
for deg ∈ 1:4
    pot = fermi_dot_lattice_cos_series(deg, lattice_a, dot_radius, v0)
    pot = TranslatedPotential(pot, lattice_a / 2, lattice_a / 2)
    # cos_pot = CosSeriesPotential(-[-v0/2 v0/4; v0/4 0], 2pi/lattice_a)
    fig = make_trajectory_plot(pot, step, intersect, offset, r0, p0, T)
    display(fig)
    save("outputs/classical/poincare_cos_$deg.pdf", fig)
    save("outputs/classical/poincare_cos_$deg.png", fig, px_per_unit=1)
end

# Complex separable (and thus integrable?) potential
cint = complex_separable_potential(5, lattice_a, dot_radius, v0)
cint = TranslatedPotential(cint, lattice_a / 2, lattice_a / 2)
fig = make_trajectory_plot(cint, step, intersect, offset, r0, p0, T)
display(fig)
save("outputs/classical/poincare_cint.pdf", fig)
save("outputs/classical/poincare_cint.png", fig, px_per_unit=1)


begin
lattice_pot = LatticePotential(lattice_a * I, dot_radius, v0;
    softness=softness, offset=offset)
fig = make_trajectory_plot(lattice_pot, step, intersect, offset, r0, p0, T)
display(fig)
save("outputs/classical/poincare_lattice.pdf", fig)
save("outputs/classical/poincare_lattice.png", fig, px_per_unit=1)
end

# Make a single explanation plot
fig = make_explanation_figure(lattice_pot, step, intersect, offset,
    LinRange(0, 2pi, 6)[1:end-1])
display(fig)
save("outputs/classical/poincare_explanation.pdf", fig)