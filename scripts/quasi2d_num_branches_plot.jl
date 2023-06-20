using CairoMakie
using LaTeXStrings
using Makie
using JLD2
using CurveFit

path_prefix = "outputs/quasi2d/"
mkpath(path_prefix)

data = load(path_prefix * "nb_data.jld2")
ts = data["ts"]
num_branches = data["nb"]
labels = data["labels"]
num_rays = data["num_rays"]
dt = data["dt"]

## Plots

datasets = [
    ("num", [1, 2, 3, 4]),
    ("cos", [x for x ∈ eachindex(labels) if x ∉ [1, 3]]),
]

for (setname, indices) ∈ datasets
    for (scalename, scaleparams) ∈ [
        ("", [])
        ("_log", [:yticks => [1, 10, 100, 1000], :yscale => Makie.pseudolog10])
    ]
        xmax = ts[end]
        if scalename == ""
            # Only show the 
            xmax = 2
        end
        included_ts = ts .≤ xmax
        fig = Figure(resolution=(800, 600))
        ax = Axis(fig[1, 1]; xlabel=LaTeXString("time (a.u.)"), ylabel=L"Number of branches $N_b$",
            title=LaTeXString("Number of branches, $num_rays rays, quasi-2D (\$p_x=1\$)"),
            limits=((0, xmax), (0, nothing)),
            scaleparams...
        )


        ls = [:solid for i ∈ indices]
        # ls[1] = :dot
        palette = vcat([RGBAf(0, 0, 0, 1)], Makie.wong_colors())
        all_lines = []
        for (c, j) ∈ enumerate(indices)
            label = labels[j]
            nb = num_branches[:, j]
            l = lines!(ax, ts[included_ts], nb[included_ts], label=LaTeXString(label), color=palette[c],
                linestyle=ls[c])
            push!(all_lines, l)
        end


        # axislegend(ax, position=:lt)
        fig[1, 2] = Legend(fig, ax, "", framevisible=false, padding=1.0)
        for ext ∈ ["png", "pdf"]
            fname = "$(setname)_branches$(scalename).$ext"
            save(path_prefix * fname, fig, px_per_unit=2)
        end
        display(fig)

        # Exponential fit
        exp_ts = (ts .≥ 2) .&& (ts .≤ 4)
        for (c, j) ∈ enumerate(indices)
            label = labels[j]
            nb = num_branches[:, j]
            a1, a2 = exp_fit(ts[exp_ts], nb[exp_ts])
            lines!(ax, ts[included_ts], a1 * exp.(a2 * ts[included_ts]), linestyle=:dash, color=palette[c])
        end

        for ext ∈ ["png", "pdf"]
            fname = "$(setname)_branches_exp$(scalename).$ext"
            save(path_prefix * fname, fig, px_per_unit=2)
        end
        display(fig)
        
        # Scale all lines to make the exponential part match
        fig = Figure(resolution=(800, 600))
        ax = Axis(fig[1, 1]; xlabel=L"t/a_2", ylabel=L"N_b/a_1",
            limits=((0, xmax), (0, 1.5*exp(xmax))),
            scaleparams...
        )
        exp_ts = (ts .≥ 2) .&& (ts .≤ 4)
        lines!(ax, ts[included_ts], exp.(ts[included_ts]), linestyle=:dash, color=:gray, label=L"e^t")
        for (c, j) ∈ enumerate(indices)
            label = labels[j]
            nb = num_branches[:, j]
            a1, a2 = exp_fit(ts[exp_ts], nb[exp_ts])
            lines!(ax, ts * a2, nb / a1, label=LaTeXString(label), color=palette[c])
        end
        fig[1, 2] = Legend(fig, ax, "", framevisible=false, padding=1.0)

        for ext ∈ ["png", "pdf"]
            fname = "$(setname)_branches_expfit$scalename.$ext"
            save(path_prefix * fname, fig, px_per_unit=2)
        end
        display(fig)

    end

end
