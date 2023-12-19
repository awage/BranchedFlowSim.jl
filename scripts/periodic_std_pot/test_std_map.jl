using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using DrWatson
using CodecZlib
using ChaosTools

# # Display and compute histograms :
# function compute_hist(a, v0, dt, T; res = 1000, num_rays = 20000, θ = 0.,  x0 = 0)
#     yg = range(-a/2, a/2, length = res)
#     xg = range(x0, T+x0, step = dt)
#     dy = yg[2] - yg[1]
#     if θ != 0.
#         pot = RotatedPotential(θ, StdMapPotential(a, v0))
#     else
#         pot = StdMapPotential(a, v0)
#     end
#     image = quasi2d_histogram_intensity(num_rays, xg, yg, pot; periodic_bnd = true)
#     return xg, yg, image
# end

# T = 40; res = 1000; num_rays = 100000;  a = 2π; v0 = -.2; dt = 1;
# x,y,image = compute_hist(a, v0, dt, T; res, num_rays)
# heatmap(y,x,log.(image))

function quasi2d_map!(du,u,p,t)
        x,y,py = u; potential, dt = p
        # kick
        du[3] = py + dt * force_y(potential, x, y)
        # drift
        du[2] = y + dt .* du[3]
        du[1] = x + dt
        # while du[2] >= π; du[2] -= 2π; end
        # while du[2] < -π; du[2] += 2π; end
        # while du[3] >= π; du[3] -= 2π; end
        # while du[3] < π; du[3] += 2π; end        
        return nothing
end


function _get_orbit(d) 
    @unpack a, v0, dt, T, res = d
    # yg = range(-a/10, a/10, length = res)
    pot = StdMapPotential(a, v0)
    df = DeterministicIteratedMap(quasi2d_map!, [0., 0.4, 0.2], [pot, dt])
    yrange = range(-a/10, a/10, res)
    py = 0.
    tr_v = [trajectory(df, T,[0., y, py]) for y in yrange]
    return @strdict(tr_v, yrange, d)
end


# Compute max lyap exp for a range of parameters
res = 5000;  a = 1; v0 = 20/(2π); dt = 0.1; T = 300; 
d = @dict(res, a, v0,  T, dt) # parametros
dat = _get_orbit(d)

image = zeros(500, T) 
@unpack tr_v, yrange = dat
# dy = step(range(-pi,pi,500))
dy = 1
for ts in tr_v
    ys = ts[1][:,2]
    for (k,y) in enumerate(ys)
        yi = 1 + 250 + round(Int, (y - (-a/2)) / dy)
        if yi >= 1 && yi <= 500 && k ≤ T
            image[yi, k] += 1
        end
    end
end

heatmap(log.(image))

