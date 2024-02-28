using DrWatson
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using CodecZlib


function compute_nmb_br_θ(d)
    @unpack V, num_angles, dt, ts, num_rays = d
    angles = range(0, π / 2, length = num_angles + 1)
    angles = angles[1:end-1]
    p = Progress(num_angles, "Potential calc")
    nb_arr = zeros(length(ts), num_angles)
    Threads.@threads for i ∈ 1:num_angles
        potential = V(angles[i])
        nb_arr[:, i] =
            quasi2d_num_branches(num_rays, dt, ts, potential)
        next!(p)
    end
    mean_nbr = vec(mean(nb_arr, dims  = 2))
    return @strdict(mean_nbr) 
end

function get_datas(V, num_rays,  num_angles, ts, dt, prefix = "pot"; T = 1)
    d = @dict(V, num_rays, num_angles, ts, T, dt) # parametros
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_nmb_br_θ, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end

# Comon parameters 
num_rays = 10000; sim_height = 1.
sim_width = 4.; v0 = 0.04
dt = 0.01

# Time steps to count branches.
ts = LinRange(0, sim_width, round(Int, 50 * sim_width))

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; num_angles = 50
V(θ) = LatticePotential(lattice_a*rotation_matrix(θ), dot_radius, v0; softness=softness) 
data_fermi = get_datas(V, num_rays, num_angles, ts, dt, "fermi_lattice"; T = sim_width)

# Random correlated
correlation_scale = 0.1; 
Vr(x) = correlated_random_potential(sim_width, sim_height, 
    correlation_scale, v0, round(Int, 1000*x))
data_rand = get_datas(Vr, num_rays, num_angles, ts, dt, "rand_correlated"; T = sim_width)

# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; num_angles = 50
# degrees = collect(1:max_degree)
degrees = [1, 6]
data_cos = Vector{typeof(data_rand)}()
for degree ∈ degrees
    cos_pot(θ) = RotatedPotential(θ,              
        fermi_dot_lattice_cos_series(degree,  
        lattice_a, dot_radius, v0; softness))
    s = savename("cos", @dict(degree))
    data = get_datas(cos_pot, num_rays, num_angles, ts, dt, s; T = sim_width)
    push!(data_cos, data)
end


# quick plot:


fig = Figure(resolution=(1200, 800))
ax1= Axis(fig[1, 1], xlabel = L"x,t", ylabel = L"N_{branches}", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30,
yscale = Makie.pseudolog10)

@unpack mean_nbr = data_fermi
lines!(ax1, ts, mean_nbr, color = :blue, label = L"Fermi Lattice")
@unpack mean_nbr = data_rand
lines!(ax1, ts, mean_nbr, color = :red, label = L"Rand Correlated")
@unpack mean_nbr = data_cos[1]
lines!(ax1, ts, mean_nbr, color = :black, label = L"Integrable Cos Pot")
@unpack mean_nbr = data_cos[2]
lines!(ax1, ts, mean_nbr, color = :green, label = L"Cos Pot deg = 6")


using JLD2
@load "stretch_factor_Nt=400_num_angles=50.jld2"
c_r = (mr .+ sr/4)./(0.1*v0_range.^(-2/3))
exp1 = 1 .*exp.(c_r[3].*ts)
lines!(ax1, ts, exp1, color = :red, linestyle = :dash, label = L"exp fit rand")

c_f = (mf .+ sf/4)./(0.2*v0_range.^(-2/3))
exp1 = 2.8*exp.(c_f[3].*ts)
lines!(ax1, ts, exp1, color = :blue, linestyle = :dash, label = L"exp fit rand")


c_f = (mc1 .+ sc1/4)./(0.2*v0_range.^(-2/3))
exp1 = 1.2*exp.(c_f[3].*ts)
lines!(ax1, ts, exp1, color = :black, linestyle = :dash, label = L"exp fit rand")
s = "Branch_number_comparison.png"
axislegend(ax1);
save(plotsdir(s),fig)
