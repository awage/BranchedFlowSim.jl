using ArgParse
using CairoMakie
using FileIO
using HDF5
using LaTeXStrings
using Statistics
using Makie

# For quick hacks / interactive testing
custom_args = [
#    "../csc_outputs/quasi2d/intensity_2000000_40.0_0.04/intensity_fermi_lattice.h5"
#    "../csc_outputs/quasi2d/intensity_2000000_40.0_0.04/intensity_fermi_rand.h5"
#    "../csc_outputs/quasi2d/intensity_2000000_40.0_0.04/intensity_rand.h5"
#    "../csc_outputs/quasi2d/intensity_2000000_40.0_0.04/intensity_cos_series_1.h5"
]
args = vcat(ARGS, custom_args)
s = ArgParseSettings()
@add_arg_table s begin
    "--time", "-T"
    help = "plot num branches at this time. Default is the end time"
    arg_type = Float64
    default = -1.0
    "--output_dir"
    help = "Output file for max intensity"
    arg_type = String
    default = "outputs/quasi2d/"
    "--legend_params"
    help = "Comma-separated list of params included in the legend"
    arg_type = String
    default = "type,correlation_scale,lattice_a"
    "input"
    help = "Input h5 file to generate the plot from"
    arg_type = String
    nargs = '+'
end
parsed_args = parse_args(args, s)
path_prefix = parsed_args["output_dir"]
if path_prefix[end] != '/'
    path_prefix *= "/"
end

# TODO: Move this to src for reuse
function make_label(data)::LaTeXString
    type = data["potential/type"]
    params = split(parsed_args["legend_params"], ",")
    label = L""
    if type == "rand"
        label = LaTeXString("Correlated random")
    elseif type == "fermi_lattice"
        a = data["potential/lattice_a"]
        label = "Periodic Fermi lattice"
    elseif type == "fermi_rand"
        a = data["potential/lattice_a"]
        label = LaTeXString("Random Fermi potential")
    elseif type == "cos_series"
        a = data["potential/lattice_a"]
        degree = data["potential/degree"]
        if degree == 1
            label = L"$c_1(\cos(2\pi x/a)+\cos(2\pi y/a))$"
        else
            label = L"$\sum_{n+m\le%$(degree)}c_{nm}\cos(2\pi nx/a)\cos(2\pi my/a)$"
        end
    elseif type == "cint"
        degree = data["potential/degree"]
        label = L"$\sum_{n=0}^{%$(degree)}c_{n}\left(\cos(nkx)+\cos(nky)\right)$"
    else
        error("Unknown/new potential type \"$type\"")
    end

    if "correlation_scale" ∈ params && "potential/correlation_scale" ∈ keys(data)
        lc = data["potential/correlation_scale"]
        label *= L", $l_c=%$lc$"
    end
    if "lattice_a" ∈ params && "potential/lattice_a" ∈ keys(data)
        a = data["potential/lattice_a"]
        label *= L", $a=%$a$"
    end
    if "v0" ∈ params && "potential/v0" ∈ keys(data)
        v0 = data["potential/v0"]
        label *= L", $v_0=%$v0$"
    end
    return LaTeXString(label)
end

## Load and analyze data
datas = [
    load(f) for f ∈ parsed_args["input"]
]
num_rays = datas[1]["num_rays"]
ys = datas[1]["ys"]
b = datas[1]["b"]
ts = datas[1]["ts"]
for d ∈ datas
    @assert d["ys"] == ys
    @assert d["ts"] == ts
    @assert d["num_rays"] == num_rays
    @assert d["b"] == b
end

ints = [
    d["intensity"] for d ∈ datas
]

max_int = [
    vec(mean(maximum(int, dims=1), dims=3))
    for int ∈ ints
]

## Plot
fig = Figure()
ax = Axis(fig[1, 1], title=L"Maximum smoothed intensity, $b=%$b$, $%$num_rays$ rays",
    xlabel=LaTeXString("time (a.u.)"), ylabel=L"I_b^\text{max}",
    yticks=0:10, limits=(nothing, (0, nothing))
)
for d ∈ datas
    max_int = vec(mean(maximum(d["intensity"], dims=1), dims=3))
    lines!(ax, ts, max_int, label=make_label(d))
end
axislegend(ax)
# display(fig)
save(path_prefix * "max_intensity.pdf", fig)
save(path_prefix * "max_intensity.png", fig, px_per_unit=2)

## Angle plots
for d ∈ datas
    if "potential/angles" ∈ keys(d)
        fig = Figure()
        title = L"Maximum smoothed intensity, $b=%$b$, $%$num_rays$ rays, "
        title = title * make_label(d)
        title = LaTeXString(title)
        angles = d["potential/angles"]
        yticks_n = LinRange(0, 0.5, 5)
        yticks_l = [L"%$x\pi" for x ∈ yticks_n]
        yticks = (pi * yticks_n, yticks_l)

        ax = Axis(fig[1, 1], title=title, xlabel=LaTeXString("time (a.u.)"),
            ylabel=L"Propagation angle $\theta$",
            yticks=yticks, limits=(nothing, (0, pi / 2))
        )
        climits = (1.0, 10.0)
        hm = heatmap!(ax, ts, angles, 
            maximum(d["intensity"], dims=1)[1,:,:],
            colorrange=climits
        )
        fig[1,2] = Colorbar(fig, hm)
        # display(fig)
        potname = d["potential/type"]
        save(path_prefix * "intensity_angle_$potname.pdf", fig)
        save(path_prefix * "intensity_angle_$potname.png", fig, px_per_unit=2)
    end
end