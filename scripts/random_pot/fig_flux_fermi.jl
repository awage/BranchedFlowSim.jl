using DrWatson 
@quickactivate
using BranchedFlowSim
using ProgressMeter
using ChaosTools
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StatsBase
using CodecZlib
using KernelDensity
using LsqFit
using Peaks


include(srcdir("utils.jl"))


function count_branches(y_ray, dy, ys)
    dy_ray = diff(y_ray)/dy
    hst = zeros(length(ys))
    a = ys[1]; b = ys[end]; dys = ys[2] - ys[1]
    ind = findall(abs.(dy_ray) .< 1.)
    str = find_consecutive_streams(ind) 
    br = 0; 
    for s in str 
        if length(s) ≥ 3
            if all(x -> (a ≤ x ≤ b) , y_ray[s])
                br += 1
            end
            for y in y_ray[s]
                yi = 1 + round(Int, (y - a) / dys)
                if yi >= 1 && yi <= length(ys)
                    hst[yi] = 1
                end
            end
        end
    end
    return br, hst
end
hst

function manifold_track(num_rays, xs, ys, potential; b = 0.0003,  ind_i = nothing, rmin = nothing, rmax = nothing, ray_y = nothing)

    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    
    if isnothing(rmax) || isnothing(rmin)
        rmin,rmax = BranchedFlowSim.quasi2d_compute_front_length(1024, dt, xs, ys, potential)
    end
    if isnothing(ray_y)        
        ray_y = collect(LinRange(rmin, rmax, num_rays))
    else
        rmin = ray_y[1]; rmax = ray_y[end]
    end
    width = length(xs)
    height = length(ys)
    image = zeros(height, width)
    hst = zeros(height, width)
    dyr = ray_y[2]-ray_y[1]
    ray_py = zeros(num_rays)
    sim_h =  dyr * num_rays
    h = length(ys) * dy
    boundary = (
        ys[1] - dy - 4*b,
        ys[end] + dy + 4*b,
    )
    xi = 1; x = xs[1]
    while x <= xs[end] + dt
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        x += dt
        while xi <= length(xs) &&  xs[xi] <= x 
            if !isnothing(ind_i)
                ind = ind_i
            else
                ind = 1:num_rays
            end
            density = kde(ray_y[ind], bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity = pdf(density, ys)*(sim_h / h)
            image[:,xi] = intensity
            nb, hst[:,xi] = count_branches(ray_y,dyr,ys)
            xi += 1
        end
    end
    return image, hst
end


v0 = 0.1; dt = 0.01; T = 10000
num_rays = 100000; θ = 0.; lyap_threshold = 2e-3
a = 0.2; dot_radius = 0.2*0.25; softness = 0.2; 
xs = range(0,10., step = dt)
y_init = range(-20*a, 20*a, length = num_rays)
V = LatticePotential(a*rotation_matrix(θ), dot_radius, v0; softness=softness)

d = @dict(V,v0, a ,y_init, T, lyap_threshold, dt, xs, num_rays, θ)  
# λ1 = _get_lyap_1D(d) 
ind = findall(λ1 .> lyap_threshold)

f = Figure(size=(600, 600))
args = (xlabel = L"y", ylabel = L"x", yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 30, xlabelsize = 30)

ys = range(-0.5*a, a*1.5, length = length(xs))
img, hst = manifold_track(length(y_init), xs, ys, V; b = 0.0003)
ax = Axis(f[1,1]; args...)
heatmap!(ax,ys,xs,log.(hst .+ 1e-5))

ind = findall(λ1 .> lyap_threshold)
img,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)
ax = Axis(f[1,2]; args...)
heatmap!(ax,ys,xs,log.(img .+ 1e-5))

ind = 1:length(λ1)
img,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)
ax = Axis(f[2,1]; args...)
heatmap!(ax,ys,xs,log.(img .+ 1e-5))

ind = findall(λ1 .< lyap_threshold)
img,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)
ax = Axis(f[2,2]; args...)
heatmap!(ax,ys,xs,log.(img .+ 1e-5))

Label(f[1, 1, TopLeft()], "(a)", padding = (0,10,10,0))
Label(f[1, 2, TopLeft()], "(b)", padding = (0,10,10,0))
Label(f[2, 1, TopLeft()], "(c)", padding = (0,10,10,0))
Label(f[2, 2, TopLeft()], "(d)", padding = (0,10,10,0))
s = "fermi_flow.png"
save(plotsdir(s),f)

