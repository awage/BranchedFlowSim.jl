using Makie
using ArgParse
using LinearAlgebra
using CairoMakie
using HDF5
using FileIO 

s = ArgParseSettings()
@add_arg_table s begin
    "input"
    help = "Input h5 file to generate the plot from"
    arg_type = String
    nargs = '+'
end

parsed_args = parse_args(ARGS, s)
for fname ∈ parsed_args["input"]
    dir = dirname(fname)
    data = load(fname)
    le_mean = data["le_mean"]
    le_max = data["le_max"]
    ris = data["ris"]
    pis = data["pis"]
    fig = Figure()
    ax = Axis(fig[1,1], xticks=0.0:0.1:0.5)
    hm = heatmap!(ax, ris, pis, le_mean, colorrange=(0.0, maximum(le_max)))
    cm = Colorbar(fig[1,2], hm)

    save("$dir/lyapunov.png", fig, px_per_unit=2)
    save("$dir/lyapunov.pdf", fig)
    println("Wrote $dir/lyapunov.pdf")
end
