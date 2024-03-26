using DrWatson 
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StatsBase
using CodecZlib
using KernelDensity
using LsqFit
using Peaks
# include(srcdir("utils.jl"))
function fit_lyap(xg, yg)
    model(x, p) = p[1] .+ p[2] *x
    p0 = [yg[1], 0.2]
    fit = curve_fit(model, xg, yg, p0)
    return fit.param, model
end

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



function manifold_track(num_rays, xs, ys, potential; b = 0.0003, threshold = 3)
    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    
    rmin,rmax = BranchedFlowSim.quasi2d_compute_front_length(1024, dt, xs, ys, potential)
    width = length(xs)
    height = length(ys)
    image = zeros(height, width)
    m_d = zeros(width)
    v_d = zeros(width)
    nb_br = zeros(width)
    nb_pks = zeros(width)
    area = zeros(width)
    max_I = zeros(width)
    ray_y = collect(LinRange(rmin, rmax, num_rays))
    dyr = ray_y[2] - ray_y[1]
    ray_py = zeros(num_rays)
    sim_h = dyr * num_rays
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
            dy_ray = diff(ray_y)/dyr
            m_d[xi] = mean(log.(abs.(dy_ray)))
            v_d[xi] = var(log.(abs.(dy_ray)))
            ind = findall(abs.(dy_ray) .< 1.)
            density = kde(ray_y[ind], bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity = pdf(density, ys)*(sim_h / h)
            image[:,xi] = intensity
            nb_br[xi], hst = count_branches(ray_y, dyr, ys)


            density = kde(ray_y, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity = pdf(density, ys)*(sim_h / h)
            t = findmaxima(intensity)
            ind = findall(t[2] .> threshold*bkg) 
            nb_pks[xi] = length(ind)
            ind = findall(intensity .> threshold*bkg) 
            area[xi] = length(ind)/length(intensity)  
            max_I[xi] = maximum(intensity)  
            xi += 1
        end
    end
    return image, ray_y, (rmin, rmax), m_d, v_d, nb_br, nb_pks, area, max_I
end

res = 2000; v0 = 0.1; dt = 0.1; 
num_rays = 300000
nr = 60
ξ = 0.1; sim_width = 35; sim_height = 30.;  
xs = range(0,20., step = dt)
ys = range(0,20*ξ, length = res)
ys = range(0, a*8, length = length(xs))
mean_nb_br_all = zeros(nr, length(xs))

for j in 1:nr
    V = correlated_random_potential(sim_width, sim_height, ξ, v0, rand(1:5000))
    println("start sim")
    img, ray_y, rm, m_d, v_d, nb_br, nb_pks, area, max_I = manifold_track(num_rays, xs, ys, V; b = 0.0003)
    mean_nb_br_all[j,:] = nb_br
end
# pks = zeros(10,length(xs))
# bra = zeros(10,length(xs))
# arr = zeros(10,length(xs))
# mdd = zeros(10,length(xs))
# vdd = zeros(10,length(xs))

# for k = 1:10
#     V = correlated_random_potential(sim_width, sim_height, correlation_scale, v0, k)
#     img, ray_y, rm, m_d, v_d, nb_br, nb_pks, area, max_I = manifold_track(num_rays, xs, ys, V)
#     pks[k,:] = nb_pks
#     bra[k,:] = nb_br
#     arr[k,:] = area
#     mdd[k,:] = m_d
#     vdd[k,:] = v_d
# end


# dy = step(ys); b = 0.001
# boundary = (
#     ys[1] - dy - 4*b,
#     ys[end] + dy + 4*b,
# )
# dy_ray = diff(ray_y)/dy
# ind = findall(abs.(dy_ray) .< 1.)
# density = kde(ray_y[ind], bandwidth= b, npoints=16 * 1024, boundary=boundary)
# intensity = pdf(density, ys)


# a,m = fit_lyap(xs[200:300],m_d[200:300])
# b,m = fit_lyap(xs[200:400],v_d[200:400])
# t0 = ξ*(v0^(-2/3))
# α = a[2]*(v0^(-2/3))*ξ 
# β = b[2]*2*(v0^(-2/3))*ξ
# @show α^2/β/t0
# gamma(x,y) = x - y/2*(sqrt(1+4*x/y) -1)
# c_f = 2*gamma.(α,β)./(ξ*v0^(-2/3))
pargs = (yticklabelsize = 40, xticklabelsize = 40, ylabelsize = 40, xlabelsize = 40) 
fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1];  xlabel = L"x", ylabel = L"n_b(t)", pargs...) 
nb_br = vec(mean(mean_nb_br_all, dims = 1))
lines!(ax1, xs, nb_br)
save(plotsdir(string("plot_random_v0=", v0, "_n=", num_rays, ".png")),fig)
using JLD2
@save "series_random_v0=0.1_n=30000.jld2" v0 num_rays mean_nb_br_all
