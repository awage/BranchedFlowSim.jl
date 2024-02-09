using DrWatson 
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
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

# Poincaré sections for the discrete system, 
# we plot y against py for every x = k * a/dt
# with "a" the cell size and dt the time step. 
# Variable y is wrapped around mod "a"

function plot_curves(V; a = 1, v0 = 1., dt = 0.01, T = 100000, x0 = 0, prefix = "poincare")
    Ttr = 1
    df = DeterministicIteratedMap(quasi2d_map!, [0., 0.4, 0.2], [V, dt])
    yrange = range(0, a, length = 40); py = 0.

    s = savename(prefix, @dict(a,v0,dt),  "png")
    fig = Figure(resolution = (800, 600))
    ax = Axis(fig[1,1], ylabel = L"p_y", xlabel = L"y", yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40, title = string("poincare sec"), titlesize = 40)
    for (j,y) in enumerate(yrange)
        u,t = trajectory(df, T, [x0, y, py])
        # We sample the trajectory for x = k * Npoincare with Npoincare = a/dt
        Npoincare = round(Int, a/dt)
        ind = range(Ttr, T, step = Npoincare)
        x_ind = rem.(u[ind,2], a, RoundUp) 
        scatter!(ax, x_ind, u[ind,3], markersize = 1.7, color = Cycled(j), rasterize = 1)
    end
    save(plotsdir(s), fig)
end

# Comon parameters
num_rays = 1000; sim_height = 1.
sim_width = 10.; v0 = 0.4
dt = 0.01; T = 40; 
res = 1000; threshold = 1.5; 
num_avg = 10

# Fermi lattice
lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; I = rotation_matrix(0)
V = LatticePotential(lattice_a*I, dot_radius, v0; softness=softness)
plot_curves(V; a = lattice_a, v0 = v0, dt = dt, T = 10000, prefix = "poincare_fermi")

# Cosine sum 
max_degree = 6; lattice_a = 0.2; dot_radius = 0.2*0.25
softness = 0.2; 
degrees = 1:max_degree
for degree ∈ degrees
    cos_pot = RotatedPotential(0,              
        fermi_dot_lattice_cos_series(degree,  
        lattice_a, dot_radius, v0; softness))
    s = savename("poincare_cos_lattice", @dict(degree))
    plot_curves(cos_pot; a = lattice_a, v0 = v0, dt = dt, T = 10000, prefix = s)
end
