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
    "--grid_plot"
    help = ""
    arg_type = Bool
    default = false
    "input"
    help = "Input h5 file to generate the plot from"
    arg_type = String
    nargs = '+'
end


function plot_heatmap(target, h5_data, title, map_data)
    ris = h5_data["rs"]
    pis = h5_data["ps"]
    dim_count = h5_data["dim_count"]

    fig = Figure()
    ax = Axis(target, xticks=0.0:0.1:0.5,
        title=title, xlabel=L"y", ylabel=L"p_y"
    )
    # Write NaN at each value that was not at all computed.
    d2 = copy(map_data)
    for idx ∈ eachindex(d2)
        if dim_count[idx] == 0
            d2[idx] = NaN
        end
    end
    hm = heatmap!(ax, ris, pis, d2, colorrange=(1.0, 2.0),
        colormap=ColorSchemes.viridis)
    return hm, ax
end

parsed_args = parse_args(ARGS, s)
# For development
# parsed_args["input"] = [
#     "../csc_outputs/classical/fermi_lattice_0.04_0.25_0.05_correlation/data.h5",
#     "../csc_outputs/classical/fermi_lattice_0.04_0.25_0.10_correlation/data.h5",
#     "../csc_outputs/classical/fermi_lattice_0.04_0.25_0.20_correlation/data.h5",
#     "../csc_outputs/classical/fermi_lattice_0.04_0.33_0.05_correlation/data.h5",
#     "../csc_outputs/classical/fermi_lattice_0.04_0.33_0.10_correlation/data.h5",
#     "../csc_outputs/classical/fermi_lattice_0.04_0.33_0.20_correlation/data.h5",
#     "../csc_outputs/classical/fermi_lattice_0.04_0.40_0.05_correlation/data.h5",
#     "../csc_outputs/classical/fermi_lattice_0.04_0.40_0.10_correlation/data.h5",
#     "../csc_outputs/classical/fermi_lattice_0.04_0.40_0.20_correlation/data.h5",
# ]

if parsed_args["grid_plot"]
    println("Making a grid plot")
    @show parsed_args["input"]
    data = [
        load(fname) for fname ∈ parsed_args["input"]
    ]
    sigmas = [
        d["potential/softness"] for d ∈ data
    ]
    radii = [
        d["potential/dot_radius"] for d ∈ data
    ]
    unique_sigma = sort(unique(sigmas))
    unique_radii = sort(unique(radii))
    fig = Figure()
    for row ∈ 1:length(unique_radii)
        r = unique_radii[row]
        wstr = @sprintf "%.2f" 1/r - 2
        Label(fig[row, 1, Left()], L"$r=%$r$, $w=%$wstr$", rotation=pi/2)
        for col ∈ 1:length(unique_sigma)
            σ = unique_sigma[col]
            if row == 1
                Label(fig[1, col, Top()], L"\sigma=%$σ")
            end
            plot_index = [i for i ∈ 1:length(data) if r == radii[i] && σ == sigmas[i]]
            @show plot_index
            @assert length(plot_index) <= 1
            if length(plot_index) == 1
                d = data[plot_index[1]]
                hm, ax = plot_heatmap(fig[row, col], d, "", d["dim_max"])
                hidedecorations!(ax)
            end
        end
    end
    cm = Colorbar(fig[1:3, 4], colorrange=(1, 2))
    for path ∈ 
        [
        "outputs/classical/grid_fractal_dim.png",
        "outputs/classical/grid_fractal_dim.pdf"
        ]
        save(path, fig, px_per_unit=2)
        println("Wrote $path")
    end
    display(fig)
else
    println("Making individual plots")
    for fname ∈ parsed_args["input"]
        dir = dirname(fname)

        println("dir=$dir")

        h5_data = load(fname)
        label = potential_label_from_h5_data(h5_data,
            ["type", "softness", "dot_radius", "v0"])

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
            title = LaTeXString("$label, fractal dimension ($name)")
            fig = Figure()
            hm, ax = plot_heatmap(fig[1, 1], h5_data, title, d)
            cm = Colorbar(fig[1, 2], hm)
            for path ∈ ["$dir/fractal_dim_$name.png","$dir/fractal_dim_$name.pdf"]
                save(path, fig, px_per_unit=2)
                println("Wrote $path")
            end
        end
    end
end