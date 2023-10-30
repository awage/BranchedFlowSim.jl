using DrWatson 
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Peaks


function _get_area(d)
    @unpack v0, res, num_rays, a, dt, T, threshold = d # unpack parameters
    yg = range(0, 1, length = res)  
    dy = yg[2]-yg[1]
    pot2 = correlated_random_potential(60*a,10*a, a, v0, rand(UInt))
    xg, area, max_I, rmax = quasi2d_get_stats(num_rays, dt, T, yg, pot2; b = 4*dy, threshold = threshold)
    return @strdict(xg, yg, area, rmax, max_I)
end

res = 1000; num_rays = 100000;  a = 0.1; v0 = 0.1; threshold = 1.4; 
T = 10.; dt = 0.01
d = @dict(res,num_rays, a, v0, threshold, T, dt) # parametros

data, file = produce_or_load(
    datadir("./storage"), # path
    d, # container for parameter
    _get_area, # function
    prefix = "random_area", # prefix for savename
    force = false, # true for forcing sims
    wsave_kwargs = (;compress = true)
)

@unpack area,xg,yg, max_I= data
plot(xg, area)
