using BranchedFlowSim
using ProgressMeter
using LinearAlgebra
using ArgParse
using HDF5
using Statistics
using Printf

s = ArgParseSettings()
@add_arg_table s begin
    "--num_rays", "-n"
    help = "number of rays"
    arg_type = Int
    default = 40000
    "--time", "-T"
    help = "end time"
    arg_type = Float64
    default = 20.0
    "--b", "-b"
    help = "Smoothing parameter b"
    arg_type = Float64
    default = 0.003
    "--yres"
    help = "Resolution in the y axis"
    arg_type = Int64
    default = 1024
    "--xres"
    help = "Resolution in x axis per one length unit. Total resolution is xres*time"
    arg_type = Int64
    default = 20
end

add_potential_args(s;
    default_potentials="fermi_lattice,fermi_rand,rand,cos_series,cint")
parsed_args = parse_args(ARGS, s)

num_rays::Int = parsed_args["num_rays"]
sim_height::Float64 = 1
sim_width::Float64 = parsed_args["time"]
v0::Float64 = parsed_args["v0"]

dir = @sprintf "intensity_%i_%.1f_%.2f" num_rays sim_width v0
path_prefix = "outputs/quasi2d/$(dir)/"
mkpath(path_prefix)
println("Saving data to $path_prefix")

dt = 0.01

# y values for sampling the intensity
yres = parsed_args["yres"]
ys = sample_midpoints(0, 1, yres)
# smoothing parameter
smoothing_b = parsed_args["b"]
# Time steps to count branches.
xres = parsed_args["xres"]
ts = LinRange(0, sim_width, round(Int, xres * sim_width))

function maximum_intensity(int)
    return vec(maximum(int, dims=1))
end

function parallel_compute_intensity(pots, progress_text="")
    println("Start $progress_text")
    int_all = zeros(length(ys), length(ts), length(pots))
    start_bounds = zeros(2, length(pots))
    p = Progress(length(pots); desc=progress_text, enabled=false)
    dy = ys[2] - ys[1]
    Threads.@threads for j ∈ 1:length(pots)
        int,(rmin,rmax) = quasi2d_intensity(
            num_rays, dt, ts, ys, pots[j], b=smoothing_b)
        int_all[:, :, j] = int
        start_bounds[:, j] = [rmin, rmax]
        next!(p)
    end
    end_p = vec(dy * sum(int_all[:, end, :], dims=1))
    println("$progress_text total_p=$(mean(end_p))" *
            " ($(minimum(end_p)) - $(maximum(end_p)))")
    return int_all, start_bounds
end

@time("load potentials",
    potentials = get_potentials_from_parsed_args(parsed_args, sim_width, 3))
    
## Compute intensities

for pot ∈ potentials
    int, start_bounds = parallel_compute_intensity(pot.instances, pot.params["type"])
    fname = path_prefix * "intensity_$(pot.name).h5"
    h5open(fname, "w") do f
        f["dt"] = Float64(dt)
        f["num_rays"] = Int64(num_rays)
        f["ts"] = Vector{Float64}(ts)
        f["ys"] = Vector{Float64}(ys)
        f["b"] = Float64(smoothing_b)
        f["intensity"] = int
        f["start_bounds"] = start_bounds
        pg = create_group(f, "potential")
        for (k, v) ∈ pairs(pot.params)
            pg[k] = v
        end
    end
end