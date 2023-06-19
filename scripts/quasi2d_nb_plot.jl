using CairoMakie
using LaTeXStrings
using Makie
using JLD2
using CurveFit

path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)

data = load(path_prefix*"nb_data_20k.jld2")
ts = data["ts"]
nbs = data["nb"]
num_rays = data["num_rays"]
dt = data["dt"]

## Plots

xmax = ts[end]
fig = Figure(resolution=(1024, 768))
ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
    title=LaTeXString("Number of branches, $num_rays rays, dt=$dt"),
    limits=((0, xmax), (0, nothing)),
    yticks=[1,10,100,1000],
    yscale=Makie.pseudolog10
)

included_ts = ts .≤ xmax

palette = Makie.wong_colors()

for (j, (label, nb)) ∈ enumerate(nbs)
    lines!(ax, ts[included_ts], nb[included_ts], label=LaTeXString(label), color=palette[j])
end

# Exponential fit
exp_ts = (ts .≥ 2) .&& (ts .≤4)
for (j, (label, nb)) ∈ enumerate(nbs)
    a1,a2 = exp_fit(ts[exp_ts], nb[exp_ts])
    lines!(ax, ts[included_ts], a1*exp.(a2*ts[included_ts]), linestyle=:dash, color=palette[j])
end

axislegend(ax, position=:lt)
fname = "int_branches.png"
save(path_prefix * fname, fig, px_per_unit=2)
fname = "int_branches_$(floor(Int, num_rays / 1000))k.png"
save(path_prefix * fname, fig, px_per_unit=2)

display(fig)