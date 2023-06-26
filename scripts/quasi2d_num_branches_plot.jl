using CairoMakie
using LaTeXStrings
using Makie
using CurveFit
using FileIO
using HDF5

path_prefix = "outputs/quasi2d/latest/"

# data = load(path_prefix * "nb_data.jld2")
# ts = data["ts"]
# num_branches = data["nb"]
# labels = data["labels"]
# num_rays = data["num_rays"]
# dt = data["dt"]

## Plots

function make_label(data)::LaTeXString
    type = data["potential/type"]
    if type == "rand"
        lc = data["potential/correlation_scale"]
        return L"Correlated random (Metzger), $l_c=%$lc$"
    elseif type == "fermi_lattice"
        a = data["potential/lattice_a"]
        return L"Periodic Fermi lattice, $a=%$a$ (mean)"
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
    end
end

datasets = [
    ("num", [
        "nb_rand.h5",
        "nb_lattice.h5",
        "nb_fermi_rand.h5",
        "nb_int_1.h5",
    ]),
    ("cos", [
        "nb_lattice.h5",
        "nb_int_1.h5",
        "nb_int_2.h5",
        "nb_int_3.h5",
        "nb_int_4.h5",
        "nb_int_5.h5",
        "nb_int_6.h5",
    ]),
]

for (setname, fnames) ∈ datasets
    data = [
        load(path_prefix * n) for n ∈ fnames
    ]
    ts = data[1]["ts"]
    num_rays = data[1]["num_rays"]
    dt = data[1]["dt"]
    # Parameters should be equal between all files
    for d ∈ data
        @assert d["dt"] == dt
        @assert d["ts"] == ts
        @assert d["num_rays"] == num_rays
    end
    labels = [make_label(d) for d ∈ data]
    num_branches = hcat([d["nb_mean"] for d ∈ data]...)

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
            nb = num_branches[:, j]
            l = lines!(ax, ts[included_ts],
                nb[included_ts], label=LaTeXString(label),
                color=palette[j],
                linestyle=ls[j])
            push!(all_lines, l)
        end


        # axislegend(ax, position=:lt)
        fig[1, 2] = Legend(fig, ax, "", framevisible=false, padding=1.0)
        for ext ∈ ["png", "pdf"]
            fname = "$(setname)_branches$(scalename).$ext"
            save(path_prefix * fname, fig, px_per_unit=2)
        end

        # Exponential fit
        exp_ts = (ts .≥ 2) .&& (ts .≤ 4)
        for j ∈ 1:length(data)
            label = labels[j]
            nb = num_branches[:, j]
            a1, a2 = exp_fit(ts[exp_ts], nb[exp_ts])
            lines!(ax, ts[included_ts],
                a1 * exp.(a2 * ts[included_ts]),
                linestyle=:dash, color=palette[j])
        end

        for ext ∈ ["png", "pdf"]
            fname = "$(setname)_branches_exp$(scalename).$ext"
            save(path_prefix * fname, fig, px_per_unit=2)
        end

        # Scale all lines to make the exponential part match
        fig = Figure(resolution=(800, 600))
        ax = Axis(fig[1, 1]; xlabel=L"t/a_2", ylabel=L"N_b/a_1",
            limits=((0, xmax), (0, 1.5 * exp(xmax))),
            scaleparams...
        )
        exp_ts = (ts .≥ 2) .&& (ts .≤ 4)
        lines!(ax, ts[included_ts], exp.(ts[included_ts]), linestyle=:dash, color=:gray, label=L"e^t")
        for j ∈ 1:length(data)
            label = labels[j]
            nb = num_branches[:, j]
            a1, a2 = exp_fit(ts[exp_ts], nb[exp_ts])
            lines!(ax, ts * a2, nb / a1, label=LaTeXString(label),
                color=palette[j])
        end
        fig[1, 2] = Legend(fig, ax, "", framevisible=false, padding=1.0)

        for ext ∈ ["png", "pdf"]
            fname = "$(setname)_branches_expfit$scalename.$ext"
            save(path_prefix * fname, fig, px_per_unit=2)
        end
    end
end
