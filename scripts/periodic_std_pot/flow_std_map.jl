using DrWatson
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using CodecZlib
using ChaosTools
using Interpolations

function quasi2d_map!(du,u,p,t)
    y,py = u; potential, dt = p
    # kick
    du[2] = py + dt * force_y(potential, t, y)
    # drift
    du[1] = y + dt * du[2]
    return nothing
end

function _get_orbit(d) 
    @unpack dt, pot, T, a, ntraj = d
    df = DeterministicIteratedMap(quasi2d_map!, [0.4, 0.2], [pot, dt])
    yrange = range(0, a, ntraj)
    py = 0.
    tr_v = [trajectory(df, T,[y, py])[1] for y in yrange]
    return @strdict(tr_v, yrange, d)
end

# This function uses the interpolated function itp_y to 
# trace the rays between two time points n and n+1
function get_density_mat(Nx, Ny, d, itp_y)
    @unpack a, v0, dt, T, ntraj = d
    ymin = -round(2*T*v0/2π); ymax = round(2*T*v0/2π)
    xrange = range(0,T, length = Nx)
    yrange = range(ymin,ymax, length = Ny)
    image = zeros(Ny, Nx) 
    dy = step(yrange)
    for (j,t) in enumerate(xrange)
        for y in itp_y
            yi = 1 + round(Int, (y(t)-ymin)/dy  )
            if yi >= 1 && yi <= Ny
                image[yi, j] += 1
            end
        end
    end
# rescale image
    image = log.(image .+ 1e-2)
    maxd=maximum(image)
    mind=minimum(image)
    image = min.((image .- mind)./(maxd-mind),1)
    return yrange, xrange, image
end

function interp_x(itp_x,t) 
    xit = [ x(t) for x in itp_x]
    return xit
end

function _produce_animation(T, itp_y, itp_p, xr, yr, image)
# Animation begins
    time = Observable(0.)
    # the time series xs_1 and ys_1 are parametrized by the 
    # $time. We use the interpolation function itp_ to get 
    # intermediate time steps
    xs_1 = @lift(mod.(interp_x(itp_y,$time),1))
    ys_1 = @lift(interp_x(itp_p,$time))
    fig = Figure(size = (2048,768))
    gb = fig[1,1] = GridLayout()
    ax1 = Axis(gb[1,1], ylabel = L"y", xlabel = L"p_y", title = @lift("t = $(round($time, digits = 1))"),)
# Scatter plot for phase space.
    scatter!(ax1, xs_1, ys_1, color = :blue, linewidth = 4)
    ax2 = Axis(gb[1,2], ylabel = L"x", xlabel = L"y")
# Density plot to track the trajectory
    heatmap!(ax2, yr, xr, image)
    lines!(ax2,[yr[1]; yr[end]],@lift([$time; $time]); color = :red)

    framerate = 10
    timestamps = range(0, T, step=0.01)
    record(fig, "time_animation.mp4", timestamps;
            framerate = framerate) do t
        time[] = t
    end
end

# Compute max lyap exp for a range of parameters
ntraj = 10000;  a = 1; v0 = 3; dt = 1; T = 10; 
pot = StdMapPotential(a, v0)
d = @dict(ntraj, pot, a, v0, T, dt) # parametros
dat = _get_orbit(d)
@unpack tr_v, yrange = dat
Ny = 1000
Nx = T*20 # Interpolation factor
itp_y = [linear_interpolation(range(0,T,step = 1), x[:,1]) for x in tr_v]
itp_p = [linear_interpolation(range(0,T,step = 1), x[:,2]) for x in tr_v]
yr, xr, image = get_density_mat(Nx, Ny, d, itp_y)
fig = heatmap(yr, xr, image)
s = savename("density_std",d, "png")
save(plotsdir(s),fig)

# _produce_animation(T, itp_y, itp_p, xr, yr, image)
