using ArgParse

export add_potential_args
export get_potentials_from_parsed_args
export ParsedPotential
export potential_label_from_h5_data

"""
    add_potential_args(s::ArgParseSettings,
        default_potentials = "rand,fermi_lattice,fermi_rand,cos_series,cint")

TBW
"""
function add_potential_args(s::ArgParseSettings;
    default_potentials="rand,fermi_lattice,fermi_rand,cos_series,cint",
    defaults=Dict{String,Any}()
)
    @add_arg_table s begin
        "--num_angles"
        help = "number of angles used for periodic potentials"
        arg_type = Int
        default = get(defaults, "num_angles", 50)
        "--num_sims"
        help = "number of realizations used for random potentials"
        arg_type = Int
        default = get(defaults, "num_sims", 100)
        "--v0"
        help = "potential height. Default is 8% of energy."
        arg_type = Float64
        default = get(defaults, "v0", 0.04)
        "--correlation_scale"
        help = "Correlation scale (l_c) used for correlated random potential"
        arg_type = Float64
        default = get(defaults, "correlation_scale", 0.1)
        "--potentials"
        help = "comma-separated list of potentials to run"
        arg_type = String
        default = default_potentials
        "--cos_max_degree"
        help = "maximum degree for the cos_series potentials"
        arg_type = Int
        default = get(defaults, "cos_max_degree", 6)
        "--lattice_a"
        help = "lattice constant, used for periodic potentials"
        arg_type = Float64
        default = get(defaults, "lattice_a", 0.2)
        "--fermi_dot_radius"
        help = "Radius of Fermi potential dots"
        arg_type = Float64
        default = get(defaults, "fermi_dot_radius", 0.25 * 0.2)
        "--fermi_softness"
        help = "Softness parameter σ used for the Fermi potentials"
        arg_type = Float64
        default = get(defaults, "fermi_softness", 0.2)
        "--shake_pos_dev"
        help = "standard deviation of position displacement for the shaken potentials."
        arg_type = Float64
        default = get(defaults, "shake_pos_dev", 0.0)
        "--shake_v_dev"
        help = "standard deviation of bump height displacement for the shaken potentials."
        arg_type = Float64
        default = get(defaults, "shake_v_dev", 0.0)
    end
end

struct ParsedPotential
    instances::Vector{AbstractPotential}
    params::Dict{String,Any}
    name::String
    function ParsedPotential(instances, params)
        p = params["type"]
        if "degree" ∈ keys(params)
            p *= "_$(params["degree"])"
        end
        if "pos_dev" ∈ keys(params)
            p *= "_$(params["pos_dev"])"
        end
        if "v_dev" ∈ keys(params)
            p *= "_$(params["v_dev"])"
        end
        return new(instances, params, p)
    end
end

function get_potentials_from_parsed_args(parsed_args, width, height)::Vector{ParsedPotential}
    potential_types = split(parsed_args["potentials"], ',')
    potentials = ParsedPotential[]
    v0 = parsed_args["v0"]
    lattice_a = parsed_args["lattice_a"]
    dot_radius = parsed_args["fermi_dot_radius"]
    softness = parsed_args["fermi_softness"]
    num_angles = parsed_args["num_angles"]
    num_sims = parsed_args["num_sims"]
    # Note: This is perhaps not sufficient for triangular (etc.) lattices,
    # but for square lattices it's enough to inspect angles from 0 to π/2
    angles = Vector(LinRange(0, π / 2, num_angles + 1)[1:end-1])
    for pt ∈ potential_types
        params = Dict{String,Any}()
        params["type"] = pt
        params["v0"] = v0
        if pt == "rand"
            # Correlated random potential
            correlation_scale = parsed_args["correlation_scale"]
            instances = Vector{AbstractPotential}(undef, num_sims)
            Threads.@threads for i ∈ 1:num_sims
                seed = i
                instances[i] =
                    correlated_random_potential(width, height, correlation_scale, v0,
                        seed)
            end
            params["correlation_scale"] = correlation_scale
            push!(potentials, ParsedPotential(instances, params))
        elseif pt == "fermi_lattice"
            # Periodic Fermi lattice
            instances =
                [LatticePotential(lattice_a * rotation_matrix(θ),
                    dot_radius, v0; softness=softness) for θ ∈ angles]
            params["lattice_a"] = lattice_a
            params["softness"] = softness
            params["angles"] = angles
            params["dot_radius"] = dot_radius
            push!(potentials, ParsedPotential(instances, params))
        elseif pt == "fermi_rand"
            instances = [random_fermi_potential(width, height, lattice_a, dot_radius, v0)
                         for _ ∈ 1:num_sims]
            params["lattice_a"] = lattice_a
            params["softness"] = softness
            params["dot_radius"] = dot_radius
            push!(potentials, ParsedPotential(instances, params))
        elseif pt == "cos_series"
            max_degree = parsed_args["cos_max_degree"]
            params["lattice_a"] = lattice_a
            params["angles"] = angles
            params["dot_radius"] = dot_radius
            params["softness"] = softness
            for degree ∈ 1:max_degree
                params["degree"] = degree
                cos_pot = fermi_dot_lattice_cos_series(
                    degree, lattice_a, dot_radius, v0, softness=softness)
                instances = [
                    RotatedPotential(θ, cos_pot) for θ ∈ angles
                ]
                push!(potentials, ParsedPotential(instances, copy(params)))
            end
        elseif pt == "cint"
            params["lattice_a"] = lattice_a
            params["angles"] = angles
            params["dot_radius"] = dot_radius
            params["softness"] = softness
            complex_int_degree = 8
            params["degree"] = complex_int_degree
            csep = complex_separable_potential(complex_int_degree,
                lattice_a, dot_radius, v0; softness=softness)
            instances = [
                RotatedPotential(θ, csep) for θ ∈ angles
            ]
            push!(potentials, ParsedPotential(instances, params))
        elseif pt == "fermi_lattice_shaken"
            pos_dev = parsed_args["shake_pos_dev"]
            v_dev = parsed_args["shake_v_dev"]

            pot = shaken_fermi_lattice_potential(
                lattice_a * I, dot_radius,
                v0;
                pos_dev=pos_dev, v_dev=v_dev
            )
            instances = [
                RotatedPotential(θ, pot) for θ ∈ angles
            ]
            params["lattice_a"] = lattice_a
            params["softness"] = softness
            params["angles"] = angles
            params["dot_radius"] = dot_radius
            params["pos_dev"] = pos_dev
            params["v_dev"] = v_dev
            push!(potentials, ParsedPotential(instances, params))
        else
            error("Unknown potential type $pt specified in --potentials")
        end
    end
    return potentials
end


# This function is placed here to avoid repetition in scripts. Not sure if this
# is the right place.
"""
    potential_label_from_h5_data(data :: Dict{String, Any})

When storing data in HDF5 files, we store metadata about the potential used to
generate said data in a group "potential/". This function parses that data
and returns a LaTeXString that can be used in plot titles and labels.

Here `data` is assumed to be a dictionary loaded from a HDF5 file by doing
data = load("fname.h5").
"""
function potential_label_from_h5_data(data::Dict{String,Any},
    params::AbstractVector{<:AbstractString}=String[
        "type",
        "correlation_scale",
        "lattice_a",
        "pos_dev",
        "v_dev"
    ]
)::LaTeXString
    type = data["potential/type"]
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
    elseif type == "fermi_lattice_shaken"
        label = "Shaken periodic Fermi lattice"
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
    if "softness" ∈ params && "potential/softness" ∈ keys(data)
        a = data["potential/softness"]
        label *= L", $\sigma=%$a$"
    end
    if "v0" ∈ params && "potential/v0" ∈ keys(data)
        v0 = data["potential/v0"]
        label *= L", $v_0=%$v0$"
    end
    if "v_dev" ∈ params && "potential/v_dev" ∈ keys(data)
        v_dev = data["potential/v_dev"]
        if v_dev > 0
            label *= L", $\sigma_v=%$v_dev$"
        end
    end
    if "pos_dev" ∈ params && "potential/pos_dev" ∈ keys(data)
        pos_dev = data["potential/pos_dev"]
        if pos_dev > 0
            label *= L", $\sigma_r=%$pos_dev$"
        end
    end
    if "dt" ∈ params
        dt = data["dt"]
        label *= L", $dt=%$dt$"
    end
    return LaTeXString(label)
end