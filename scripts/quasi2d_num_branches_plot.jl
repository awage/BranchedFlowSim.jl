using CairoMakie
using LaTeXStrings
using BranchedFlowSim
using Makie
using CurveFit
using FileIO
using Printf
using HDF5
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--time", "-T"
    help = "only plot until this time"
    arg_type = Float64
    default = 10000.0
    "--lin_time"
    help = "for linear plots, only plot until this time"
    arg_type = Float64
    default = 2.0
    "--data_dir"
    help = "directory with input data"
    arg_type = String
    default = "outputs/quasi2d/latest/"
    "--output_dir"
    help = "directory to write output data, by default same as data_dir"
    "--setname"
    help = "Prefix for the generated files."
    arg_type = String
    default = "num"
    "--legend_params"
    help = "Comma-separated list of params included in the legend"
    arg_type = String
    default = "type,correlation_scale,lattice_a"
    "--expfit"
    help = "If set, generate exp fit plots"
    action = :store_true
    "files"
    help = "Input files to genererate the plot from"
    arg_type = String
    nargs = '*'
end

parsed_args = parse_args(ARGS, s)

max_time = parsed_args["time"]
lin_max_time = parsed_args["lin_time"]
if lin_max_time <= 0
    lin_max_time = 1e8
end

input_prefix = parsed_args["data_dir"]
# Make sure the prefix ends in /
if input_prefix[end] != '/'
    input_prefix = input_prefix * "/"
end

output_prefix = input_prefix
if parsed_args["output_dir"] !== nothing
    output_prefix = parsed_args["output_dir"]
    # Make sure the prefix ends in /
    if output_prefix[end] != '/'
        output_prefix = output_prefix * "/"
    end
end
mkpath(output_prefix)

# data = load(path_prefix * "nb_data.jld2")
# ts = data["ts"]
# num_branches = data["nb"]
# labels = data["labels"]
# num_rays = data["num_rays"]
# dt = data["dt"]

## Plots

function generate_plots(output_prefix, setname, fnames)
    data = [
        load(n) for n ∈ fnames
    ]
    num_rays = data[1]["num_rays"]
    dt = data[1]["dt"]
    ts = [d["ts"] for d ∈ data]
    # Parameters should be equal between all files
    for d ∈ data
        @assert d["dt"] == dt
        @assert (d["ts"] == ts[1] || d["ts"][end] >= max_time)
        @assert d["num_rays"] == num_rays
    end
    legend_params = split(parsed_args["legend_params"], ",")
    labels = [
        potential_label_from_h5_data(d, legend_params) for d ∈ data
        ]
    num_branches = [d["nb_mean"] for d ∈ data]

    for (scalename, scaleparams) ∈ [
        ("", [])
        ("_log", [:yticks => [1, 10, 100, 1000], :yscale => Makie.pseudolog10])
    ]
        xmax = min(ts[1][end], max_time)
        if scalename == ""
            # Only show the first part in linear scale
            xmax = min(xmax, lin_max_time)
        end
        fig = Figure(resolution=(1024, 576))
        ax = Axis(fig[1, 1]; xlabel=LaTeXString("time (a.u.)"), ylabel=L"Number of branches $N_b$",
            title=LaTeXString("Number of branches, $num_rays rays, quasi-2D (\$p_x=1\$)"),
            limits=((0, xmax), (0, nothing)),
            scaleparams...
        )


        ls = [:solid for i ∈ 1:length(data)]
        # ls[1] = :dot
        palette = vcat([RGBAf(0, 0, 0, 1)], Makie.wong_colors())
        all_lines = []
        for j ∈ 1:length(data)
            label = labels[j]
            included_ts = ts[j] .≤ xmax
            nb = num_branches[j]
            l = lines!(ax, ts[j][included_ts],
                nb[included_ts], label=LaTeXString(label),
                color=palette[j],
                linestyle=ls[j])
            push!(all_lines, l)
        end


        # axislegend(ax, position=:lt)
        legend = Legend(fig, ax, "", framevisible=false, padding=1.0)
        fig[1, 2] = legend
        for ext ∈ ["png", "pdf"]
            fname = "$(setname)_branches$(scalename).$ext"
            save(output_prefix * fname, fig, px_per_unit=2)
        end

        # Plot matching exponential curve
        for j ∈ 1:length(data)
            label = labels[j]
            nb = num_branches[j]
            exp_ts = (ts[j] .≥ 2) .&& (ts[j] .≤ 4)
            a1, a2 = exp_fit(ts[j][exp_ts], nb[exp_ts])
            included_ts = ts[j] .≤ xmax
            lines!(ax, ts[j][included_ts],
                a1 * exp.(a2 * ts[j][included_ts]),
                linestyle=:dash, color=palette[j],
                label=LaTeXString(@sprintf("\$%.2f \\exp(%.2f t)\$", a1, a2))
                )
        end
        delete!(legend)
        legend = Legend(fig, ax, "", framevisible=false, padding=1.0)
        fig[1, 2] = legend

        for ext ∈ ["png", "pdf"]
            fname = "$(setname)_branches_exp$(scalename).$ext"
            save(output_prefix * fname, fig, px_per_unit=2)
        end
        if parsed_args["expfit"]

            # Scale all lines to make the exponential part match
            fig = Figure(resolution=(800, 600))
            ax = Axis(fig[1, 1]; xlabel=L"t/a_2", ylabel=L"N_b/a_1",
                limits=((0, xmax), (0, 1.5 * exp(xmax))),
                scaleparams...
            )
            included_ts = ts[1] .≤ xmax
            lines!(ax, ts[1][included_ts],
                exp.(ts[1][included_ts]), linestyle=:dash, color=:gray, label=L"e^t")
            for j ∈ 1:length(data)
                label = labels[j]
                nb = num_branches[j]
                exp_ts = (ts[j] .≥ 2) .&& (ts[j] .≤ 4)
                a1, a2 = exp_fit(ts[j][exp_ts], nb[exp_ts])
                lines!(ax, ts[j] * a2, nb / a1, label=LaTeXString(label),
                    color=palette[j])
            end
            fig[1, 2] = Legend(fig, ax, "", framevisible=false, padding=1.0)

            for ext ∈ ["png", "pdf"]
                fname = "$(setname)_branches_expfit$scalename.$ext"
                save(output_prefix * fname, fig, px_per_unit=2)
            end
        end
    end
end

apply_prefix(fs) = [input_prefix * f for f ∈ fs]

datasets = [
    ("num", apply_prefix([
        "nb_rand.h5",
        "nb_lattice.h5",
        "nb_fermi_rand.h5",
        "nb_cos_series_1.h5",
    ])),
    ("cos", apply_prefix([
        "nb_lattice.h5",
        "nb_cos_series_1.h5",
        "nb_cos_series_2.h5",
        "nb_cos_series_3.h5",
        "nb_cos_series_4.h5",
        "nb_cos_series_5.h5",
        "nb_cos_series_6.h5",
        "nb_cint_8.h5",
    ])),
]

if !isempty(parsed_args["files"])
    datasets = [
        (parsed_args["setname"], parsed_args["files"])
    ]
end

for (setname, fnames) ∈ datasets
    generate_plots(output_prefix, setname, fnames)
end