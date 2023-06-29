using ArgParse
using CairoMakie
using FileIO
using HDF5
using LaTeXStrings
using Makie

# For quick hacks / interactive testing
custom_args = [
]
args = vcat(ARGS, custom_args)
s = ArgParseSettings()
@add_arg_table s begin
    "--time", "-T"
        help = "plot num branches at this time. Default is the end time"
        arg_type = Float64
        default = -1.0
    "--output"
        help = "Output file"
        arg_type = String
        default = "outputs/quasi2d/nb_angles.pdf"
    "input"
        help = "Input h5 file to generate the plot from"
        arg_type = String
        nargs = '+'
end
parsed_args = parse_args(args, s)

## Do plotting

function make_label(data)::LaTeXString
    # TODO: Perhaps move this under src/ as well
    type = data["potential/type"]
    if type == "rand"
        lc = data["potential/correlation_scale"]
        return L"Correlated random, $l_c=%$lc$"
    elseif type == "fermi_lattice"
        a = data["potential/lattice_a"]
        return L"Periodic Fermi lattice, $a=%$a$"
    elseif type == "fermi_rand"
        a = data["potential/lattice_a"]
        return LaTeXString("Random Fermi potential")
    elseif type == "cos_series"
        a = data["potential/lattice_a"]
        degree = data["potential/degree"]
        if degree == 1
            return L"$c_1(\cos(2\pi x/a)+\cos(2\pi y/a))$, $a=%$a$"
        end
        return L"$\sum_{n+m\le%$(degree)}c_{nm}\cos(nkx)\cos(mky)$, $k=2\pi/%$a$"
    elseif type == "cint"
        a = data["potential/lattice_a"]
        degree = data["potential/degree"]
        return L"$\sum_{n=0}^{%$(degree)}c_{n}\left(\cos(nkx)+\cos(nky)\right)$, $k=2\pi/%$a$"
    end
    error("Unknown/new potential type \"$type\"")
end

time = parsed_args["time"]
fnames = parsed_args["input"]

fig = Figure()
@time "data loaded" datas = [load(f) for f ∈ fnames]

ts = datas[1]["ts"]
angles = datas[1]["potential/angles"]
num_rays = datas[1]["num_rays"]
for d ∈ datas
    @assert d["ts"] == ts
    @assert d["potential/angles"] == angles
    @assert d["num_rays"] == num_rays
end

if time < 0 || time > ts[end]
    time = ts[end]
end

idx = findfirst(x->(x >= time), ts)
@assert idx !== nothing
println("ts index $idx, t=$(ts[idx])")


title=L"Number of branches at $t=%$(time)$, %$(num_rays) rays"
if length(datas) == 1
    title = make_label(datas[1])
end
fig = Figure(resolution=(500,500))
tickpoints = LinRange(0, 0.5, 5)
ax = Axis(fig[1,1],
    title=title, xticks=(tickpoints*pi,[L"%$(x)\pi" for x ∈ tickpoints]),
    ylabel=L"Number of branches $N_b$",
    xlabel=L"Angle of propagation $\theta$",
    limits=((0, pi/2), nothing)
)
for d ∈ datas
    lbl = make_label(d)
    lines!(ax, angles, d["nb_all"][idx,:], label=lbl)
end
if length(datas) > 1
    axislegend(ax)
end

outfile = parsed_args["output"]
save(outfile, fig)
println("Wrote $outfile")
