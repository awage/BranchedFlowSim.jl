using DrWatson 
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using ChaosTools


function quasi2d_map!(du,u,p,t)
        x,y,py = u; potential, dt = p
        # kick
        du[3] = py + dt * force_y(potential, x, y)
        # drift
        du[2] = y + dt .* du[3]
        du[1] = x + dt
        return nothing
end


function _get_lyap_1D(d) 
    @unpack V, dt, a, T, res = d
    df = DeterministicIteratedMap(quasi2d_map!, [0., 0.4, 0.2], [V, dt])
    
    yrange = range(0, a, res)
    py = 0.
    λ = [lyapunov(df, T; u0 = [0., y, py]) for y in yrange]
    return @strdict(λ, yrange, d)
end


# # print max lyap as a function of y over a range of initial conditions 
# function print_fig_lyap(r; res = 500,  a = 1, v0 = 1., dt = 0.01, T = 1000)
#     d = @dict(res, r, a, v0,  T, dt) # parametros
#     data, file = produce_or_load(
#         datadir("./storage"), # path
#         d, # container for parameter
#         _get_lyap_1D, # function
#         prefix = "periodic_bf_lyap_1D", # prefix for savename
#         force = false, # true for forcing sims
#         wsave_kwargs = (;compress = true)
#     )

#     @unpack yrange, λ = data
#     s = savename("plot_lyap_1D", d, "png")
#     fig = Figure(resolution=(800, 600))
#     ax1= Axis(fig[1, 1], title = string("r = ", r) , xlabel = L"y", ylabel = L"\lambda_{max}", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40) 
#     hm = scatter!(ax1, yrange, λ)
#     lines!(ax1,[-0.5, 0.5], [0.001, 0.001]; color = :red) 
#     save(plotsdir(s),fig)
# end

function get_lyap_index(V, threshold; res = 500, a = 1, v0 = 1., dt = 0.01, T = 10000, prefix = "lyap")
    d = @dict(res,  a, v0,  T, dt, V) # parametros
    data, file = produce_or_load(
        datadir("./storage"), # path
        d, # container for parameter
        _get_lyap_1D, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    @unpack λ = data
    ind = findall(λ .> threshold)
    l_index = length(ind)/length(λ) 
    return l_index
end

# Compute max lyap exp for a range of parameters
# res = 500;  a = 1; v0 = 1.; dt = 0.01; T = 10000; threshold = 0.001
# rrange = range(0,0.5, length = 50)
# ll = Float64[]
# for r in rrange
#     V = CosMixedPotential(r,a,v0)
#     lidx = get_lyap_index(r, V, 0.001; res, a, v0, dt, T)
#     push!(ll, lidx)
# end

# d = @dict(res, a, v0,  T, dt) # parametros
# s = savename("lyap_index",d, "png")
# fig = Figure(resolution=(800, 600))
# ax1= Axis(fig[1, 1],  xlabel = L"r", ylabel = "lyap index", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40,  titlesize = 40) 
# lines!(ax1, rrange, ll, color = :blue)
# save(plotsdir(s),fig)

# print_fig_lyap(0.0; a, v0, dt, T)
# print_fig_lyap(0.12; a, v0, dt, T)
# print_fig_lyap(0.25; a, v0, dt, T)
# print_fig_lyap(0.5; a, v0, dt, T)
#

# Comon parameters
num_rays = 1000; sim_height = 1.
sim_width = 10.; v0 = 0.04
dt = 0.01; T = 40; 
res = 1000; threshold = 0.001; 
num_avg = 10

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; I = rotation_matrix(0)
V = LatticePotential(lattice_a*I, dot_radius, v0; softness=softness)
l = get_lyap_index(V, threshold; res = 500, a = lattice_a, v0 = v0, dt = dt, T = 10000, prefix = "lyap_fermi")
@show l


# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
degrees = 1:max_degree
for degree ∈ degrees
    cos_pot = RotatedPotential(0,              
        fermi_dot_lattice_cos_series(degree,  
        lattice_a, dot_radius, v0; softness))
    s = savename("lyap_cos_lattice", @dict(degree))
    l = get_lyap_index(cos_pot, threshold; res = 500, a = lattice_a, v0 = v0, dt = dt, T = 10000, prefix = s)
    @show l
end

