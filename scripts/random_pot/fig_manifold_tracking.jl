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

function find_crossings(x,y,ray) 
    v = Vector{Float64}()
    vr = Vector{Float64}()
    for k in 1:length(y)-1
        if sign(y[k]) ≠ sign(y[k+1])
            push!(v,x[k])
            push!(vr,ray[k])
        end
    end
    return v,vr
end


function find_branches(x,y,ray) 
   v = []
   vr = []
   ind = findall(abs.(y) .< 1) 
   str = find_consecutive_streams(ind) 
   for s in str 
        if length(s) ≥ 3
            push!(v, [x[s], y[s]])
            push!(vr, [x[s], ray[s]])
        end
    end
    return v,vr
end

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



a = 0.1; v0 = 0.1; dt = 0.01;  num_rays = 200000;  ξ = 0.1
y_init = range(-20*a, 20*a, length = num_rays)
V = correlated_random_potential(20*a,10*a, ξ, v0, 201)
ys = y_init[findall(0 .≤ y_init .≤ 0.5)]

pargs = (yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40) 
fig = Figure(size=(1200, 1200))
ax1= Axis(fig[1, 1];  xlabel = L"y(0)", ylabel = L"y(t)", pargs...) 
ax2= Axis(fig[1, 2];  ylabel = L"\delta y(t)/\delta y(0)", xlabel = L"y(0)", pargs...) 
ax3= Axis(fig[2, 1];  xlabel = L"y", ylabel = L"x", pargs...) 
ax4= Axis(fig[2, 2];  xlabel = L"y", ylabel = L"x", pargs...) 

xs = range(0,.4, step = dt)
img_all,flow_all, ray_y,  nb_br_all = manifold_track(num_rays, xs, ys, V; b = 0.0003, ray_y = collect(y_init))
lines!(ax1, ys, ray_y)
dy = ys[2] - ys[1]
dry = diff(ray_y)/dy; 
xcr, ycr = find_crossings(ys[1:end-1], dry, ray_y)
vbr, vbr2 = find_branches(ys[1:end-1], dry, ray_y) 


lines!(ax2, ys[1:end-1], dry)
scatter!(ax2, xcr, zeros(length(xcr)), color = :red)
scatter!(ax1, xcr, ycr, color = :red)
for br in vbr 
    lines!(ax2, br[1], br[2], color = :red)
end
for br in vbr2 
    lines!(ax1, br[1], br[2], color = :red)
end
poly!(ax2, Point2f[(0, 1), (0.5, 1), (0.5, -1), (0, -1.)], color = (:blue,0.1), strokewidth = 0)


xs = range(0,1., step = dt)
img_all,flow_all, ray_y,  nb_br_all = manifold_track(num_rays, xs, ys, V; b = 0.0003, ray_y = collect(y_init))
heatmap!(ax4,ys,xs,img_all; colormap = :grays)
lines!(ax4,[ys[1]; ys[end]], [0.4; 0.4]; color = :red)
heatmap!(ax3,ys,xs,log.(flow_all .+ 1e-5))
lines!(ax3,[ys[1]; ys[end]], [0.4; 0.4]; color = :red)
# save(plotsdir(string("plot_fit_random_pot=", r, "_thr=", threshold, ".png")),fig)
Label(fig[1, 1, TopLeft()], "(a)", padding = (0,15,15,0), fontsize = 30)
Label(fig[1, 2, TopLeft()], "(b)", padding = (0,15,15,0), fontsize = 30)
Label(fig[2, 1, TopLeft()], "(c)", padding = (0,15,15,0), fontsize = 30)
Label(fig[2, 2, TopLeft()], "(d)", padding = (0,15,15,0), fontsize = 30)
s = "branch_detection.png"
save(plotsdir(s),fig)
