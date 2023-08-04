using BranchedFlowSim
using ColorTypes
using LaTeXStrings
using ArgParse
using Printf
using CairoMakie
using Makie
using LinearAlgebra
using ColorSchemes
using FileIO

s = ArgParseSettings()
@add_arg_table s begin
    "input"
    help = "Input h5 file to generate the plot from"
    arg_type = String
end

parsed_args = parse_args(ARGS, s)

fname = parsed_args["input"]
dir = dirname(fname)

println("dir=$dir")

h5_data = load(fname)
label = potential_label_from_h5_data(h5_data,
    ["type", "softness"])

dim_mean = h5_data["dim_mean"]
dim_min = h5_data["dim_min"]
dim_max = h5_data["dim_max"]
dim_count = h5_data["dim_count"]
ris = h5_data["rs"]
pis = h5_data["ps"]

for (name, d) ∈ [
    ("mean", dim_mean),
    ("min", dim_min),
    ("max", dim_max)
]
    fig = Figure()
    ax = Axis(fig[1, 1], xticks=0.0:0.1:0.5,
        title=LaTeXString("$label, fractal dimension ($name)"),
        xlabel=L"y",
        ylabel=L"p_y"
        )
    # Write NaN at each value that was not at all computed.
    d2 = copy(d)
    for idx ∈ eachindex(d2)
        if dim_count[idx] == 0
            d2[idx] = NaN
        end
    end
    hm = heatmap!(ax, ris, pis, d2, colorrange=(1.0, 2.0),
        colormap=ColorSchemes.viridis)
    cm = Colorbar(fig[1, 2], hm)
    save("$dir/fractal_dim_$name.png", fig, px_per_unit=2)
end
# 