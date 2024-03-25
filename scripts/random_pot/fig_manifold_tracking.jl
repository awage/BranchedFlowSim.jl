using DrWatson 
@quickactivate
using BranchedFlowSim
using ProgressMeter
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StatsBase
using CodecZlib
using KernelDensity


include(srcdir("utils.jl"))


function count_branches(y_ray, dy, ys)
    dy_ray = diff(y_ray)/dy
    hst = zeros(length(ys))
    a = ys[1]; b = ys[end]; dys = ys[2] - ys[1]
    ind_br = findall(abs.(dy_ray) .< 1.)
    br = 0; cnt = 0
    for k in 1:length(ind_br)-1
        if ind_br[k] == ind_br[k+1] - 1  
                cnt += 1 
        else 
            # the manifold stops, let see if we have long enough stretch
            if cnt ≥ 3
                if all(x -> (a ≤ x ≤ b) , y_ray[ind_br[k-cnt:k]])
                    br += 1
                end
                for y in y_ray[ind_br[k-cnt:k]]
                    yi = 1 + round(Int, (y - a) / dys)
                    if yi >= 1 && yi <= length(ys)
                        hst[yi] = 1
                    end
                end
            end
            cnt = 0
        end
    end
    return br, hst
end



function manifold_track(num_rays, xs, ys, potential; b = 0.0003, ray_y = nothing)

    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    
    rmin = ray_y[1]; rmax = ray_y[end]

    width = length(xs)
    height = length(ys)
    image = zeros(height, width)
    flow = zeros(height, width)
    nb_br = zeros(width)
    indi = findall(ys[1] .≤  ray_y  .≤ ys[end])

    dyr = (ray_y[2]-ray_y[1])
    ray_py = zeros(num_rays)
    sim_h =  dyr * num_rays
    h = length(ys) * dy
    bkg = 1/(dy*length(ys))
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
            density = kde(ray_y, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity = pdf(density, ys)*(sim_h / h)
            nb_br[xi], hst = count_branches(ray_y, dyr, ys)
            image[:,xi] = hst
            flow[:,xi] = intensity
            xi += 1
        end
    end
    return image, flow, ray_y[indi],  nb_br
end



v0 = 0.1; dt = 0.01;  num_rays = 400000;  ξ = 0.1
xs = range(0,.5, step = dt)
y_init = range(-20*a, 20*a, length = num_rays)
V = correlated_random_potential(20*a,10*a, ξ, v0, rand(1:1000))
ys = y_init[findall(0 .≤ y_init .≤ 0.5)]

img_all,flow_all, ray_y,  nb_br_all = manifold_track(num_rays, xs, ys, V; b = 0.0003, ray_y = collect(y_init))

pargs = (yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40) 

fig = Figure(size=(1200, 1200))
ax1= Axis(fig[1, 1];  xlabel = L"y(0)", ylabel = L"y(t)", pargs...) 
ax2= Axis(fig[1, 2];  xlabel = L"\delta y(0)", ylabel = L"\delta y(t)", pargs...) 
ax3= Axis(fig[2, 1];  xlabel = L"y(t)", ylabel = L"x(t)", pargs...) 
ax4= Axis(fig[2, 2];  xlabel = L"y(t)", ylabel = L"x(t)", pargs...) 

lines!(ax1, ys, ray_y)
dy = step(y_init)
lines!(ax2, ys[1:end-1], diff(ray_y)/dy)
heatmap!(ax4,ys,xs,img_all)
heatmap!(ax3,ys,xs,log.(flow_all .+ 1e-5))
# save(plotsdir(string("plot_fit_random_pot=", r, "_thr=", threshold, ".png")),fig)
save("test.png", fig)
