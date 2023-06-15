using BranchedFlowSim
using CairoMakie
using Interpolations
using Statistics
using FFTW
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Makie
using ColorTypes


path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)
sim_height = 1
sim_width = 4
num_rays = 1024
dt = 1/512
# num_rays = 512
correlation_scale = 0.1
# To match Metzger, express potential as percents from particle energy (1/2).
ϵ_percent = 8
v0::Float64 = ϵ_percent * 0.01 * 0.5
softness = 0.2

num_sims = 100

Ny = num_rays
Nx = round(Int, sim_width / dt)
ys = LinRange(0, sim_height, Ny + 1)[1:end-1]
xs = LinRange(0, sim_width, Nx + 1)[1:end-1]

# Time steps to count branches.
ts = LinRange(0, xs[end], 100)

# Periodic potential
lattice_a::Float64 = 0.2
dot_radius = 0.25 * lattice_a
dot_v0 = 0.08 * 0.5

# Compute average over all angles
function get_lattice_mean_nb()::Vector{Float64}
    angles = LinRange(0, π / 2, 51)[1:end-1]
    grid_nb = zeros(length(ts), length(angles))
    Threads.@threads for di ∈ 1:length(angles)
        θ = angles[di]
        potential = LatticePotential(lattice_a * rotation_matrix(θ),
            dot_radius, dot_v0; offset=[lattice_a / 2, 0], softness=softness)
        grid_nb[:, di] = quasi2d_num_branches(xs, ys, ts, potential) / sim_height
    end
    return vec(sum(grid_nb, dims=2) / length(angles))
end

# Integrable potential
function integrable_pot(x::Real, y::Real)::Float64
    v0 * (cos(2pi * x / lattice_a) + cos(2pi * y / lattice_a))
end

function integrable_pot2(x::Real, y::Real)::Float64
    a0 = 0.4441648435271566 / 2
    a1 = 0.2632722982611676
    a2 = 0.1491129866125936
    k = 2pi / lattice_a
    return v0 * (a0+a1*(cos(k * x) + cos(k * y)) - a2 * cos(k * x)*cos(k*y))
end

function get_nb_int(V)::Vector{Float64}
    num_angles = 50
    angles = LinRange(0, π / 2, num_angles + 1)[1:end-1]
    int_nb_arr = zeros(length(ts), length(angles))
    Threads.@threads for di ∈ 1:length(angles)
        θ = angles[di]
        potential = RotatedPotential(θ, V)
        # tys = LinRange(0, 1, 2num_rays)
        int_nb_arr[:, di] = quasi2d_num_branches(xs, ys, ts, potential) / sim_height
    end
    return vec(sum(int_nb_arr, dims=2) / length(angles))
end

struct CosSeriesPotential{MatrixType}
    w::MatrixType
    k::Float64
end

function (V::CosSeriesPotential)(x, y)
    len = size(V.w)[1]
    xcos = MVector{len, Float64}(undef)
    ycos = MVector{len, Float64}(undef)
    xcos[1] = ycos[1] = 1.0
    for k ∈ 2:len
        kk = (k-1)*V.k
        xcos[k] = cos(kk*x)
        ycos[k] = cos(kk*y)
    end
    xycos = ycos * transpose(xcos)
    return sum(
        V.w .* xycos
    )
end

function BranchedFlowSim.force(V::CosSeriesPotential, x, y)
    # TODO: This can be greatly optimized
    return BranchedFlowSim.force_diff(V, x, y)
end

function potential_label(degree)
    lbl = L"\sum_{n+m\le%$(degree)}a_{nm}\cos(nkx)\cos(mky)"
end

function make_integrable_potential(degree)
    pot = LatticePotential(lattice_a * rotation_matrix(0),
    dot_radius, v0)
    # Find coefficients numerically by doing FFT
    N = 128
    xs = LinRange(0, lattice_a, N+1)[1:N]
    g = grid_eval(xs, xs, pot)
    # Fourier coefficients
    w = (2/length(g))*fft(g)
    # Convert from exp coefficients to cosine coefficients
    w = real(w[1:(degree+1), 1:(degree+1)])
    w[1,1] /= 2
    w[2:end, 2:end] *= 2
    for i ∈ 1:degree
        for j ∈ 1:degree
            if (i+j) > degree
                w[1+i,1+j] = 0
            end
        end
    end
    k =  2pi/lattice_a
    # Use statically sized arrays.
    static_w = SMatrix{1+degree,1+degree,Float64,(1+degree)^2}(w)
    return CosSeriesPotential{typeof(static_w)}(static_w, k)
end

## Actually evaluate

@time "lattice sim" nb_lattice = get_lattice_mean_nb()
max_degree = 6
int_potentials = [make_integrable_potential(degree) for degree ∈ 1:max_degree]
@time "all int sims" nb_int = [get_nb_int(pot) for pot ∈ int_potentials]

## Make a comparison between rand, lattice, and int

fig = Figure(resolution=(1024, 768))
ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
    title=LaTeXString("Number of branches"), limits=((0, sim_width), (0, nothing)))

lines!(ax, ts, nb_lattice, label=L"periodic lattice, $a=%$(lattice_a)$ (mean)")
for degree ∈ 1:max_degree
    #lbl=L"degree %$(degree) $\cos$ approx $a=%$(lattice_a)$ (mean)")
    lbl = potential_label(degree)
    lines!(ax, ts, nb_int[degree], label=lbl)
end
axislegend(ax, position=:lt)
save(path_prefix * "int_branches.png", fig, px_per_unit=2)
display(fig)

## Visualize simulations
#
lattice_mat = lattice_a * rotation_matrix(-pi / 10)
rot_lattice = LatticePotential(lattice_mat, dot_radius, v0)
quasi2d_visualize_rays(path_prefix * "lattice_rot_sim.png", xs, ys,
    rot_lattice,
)
for degree ∈ 1:max_degree
    quasi2d_visualize_rays(path_prefix * "int_$(degree)_rot_sim.png", xs,
        LinRange(0, 1, 1024),
        RotatedPotential(pi / 10, int_potentials[degree]),
    )
end