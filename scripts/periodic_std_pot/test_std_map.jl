using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using DrWatson
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
    @unpack a, v0, dt, T, ntraj = d
    pot = StdMapPotential(a, v0)
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
    ymin = -round(T*v0/2π); ymax = round(T*v0/2π)
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

# Compute max lyap exp for a range of parameters
ntraj = 5000;  a = 1; v0 = 3.7; dt = 1; T = 10; 
d = @dict(ntraj, a, v0,  T, dt) # parametros
dat = _get_orbit(d)
@unpack tr_v, yrange = dat
Ny = 1000
Nx = T*20 # Interpolation factor
itp_y = [linear_interpolation(range(0,T,step = dt), x[:,1]) for x in tr_v]
yr, xr, image = get_density_mat(Nx, Ny, d, itp_y)
fig = heatmap(yr, xr, image)
save("density_1.png",fig)
