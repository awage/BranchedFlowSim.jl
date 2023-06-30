# Plot gaussian_correlatedrng_random and make sure statistics are as expected
using BranchedFlowSim
using CairoMakie
using Statistics
using LaTeXStrings

Lx = 4
Ly = 4
Nx = 1024
Ny = 1024

# This is the "feature scale" of the random potential
scale = 0.1

# xs = LinRange(-Lx / 2, Lx / 2, Nx)
# ys = LinRange(0, Ly, Ny)
# Use kind of strange values for xs and ys
xs = LinRange(0, Lx, Nx)
ys = LinRange(-Ly / 2, Ly / 2, Ny)


V = gaussian_correlated_random(xs, ys, scale)

## Plot
fig = Figure(resolution=(1024,1024))
ax = Axis(fig[1, 1], aspect=DataAspect())
# rowsize!(fig.layout, 1, Relative(10))
hmap = heatmap!(ax, ys, xs, V)
cbar = Colorbar(fig[1, 2], hmap)

flat = [V[i] for i ∈ eachindex(V)]

# Plot histogram and compare to normal distribution pdf
ax = Axis(fig[2, 1:2], limits=((-3,3), nothing))
# rowsize!(fig.layout, 2, Relative(5))
hist!(ax, flat, bins=20, normalization=:pdf)
ts = LinRange(-3, 3, 512)
normal_pdf = (1 / (sqrt(2pi))) * exp.(-0.5 * ts .^ 2)
lines!(ax, ts, normal_pdf,
    label=L"\exp(-x^2/2)/\sqrt{2\pi}"
)
axislegend(ax)


# Correlation
function random_corr(r)
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    x1i = rand(eachindex(xs))
    y1i = rand(eachindex(ys))
    # Pick a point in a random direction with distance `r`
    dir = rand() * 2π
    x2i = round(Int, x1i + cos(dir) * r / dx)
    y2i = round(Int, y1i + sin(dir) * r / dy)
    x2i = 1 + (x2i + Nx - 1) % Nx
    y2i = 1 + (y2i + Ny - 1) % Ny
    return V[y1i, x1i] * V[y2i, x2i]
end

rs = LinRange(0, min(Lx, Ly) / 2, 256)
num_pairs = 40000
corr = [
    mean(
        random_corr(r) for i ∈ 1:num_pairs
    )
    for r ∈ rs
]
ax = Axis(fig[3, 1:2], xlabel=L"r")
lines!(ax, rs, corr,
    label=L"\overline{V(\vec{r})V(\vec{r} + \vec{r}')}")
lines!(ax, rs, exp.(-(rs / scale) .^ 2),
    label=L"\exp(-(r^2/s^2)")

axislegend(ax)

save("outputs/verify_correlated_random.png", fig)
display(fig)