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

function compute_lyap(d)
    @unpack v0, T, dt, V, y_init, lyap_threshold, xs, a = d # parametros
    λ1 = _get_lyap_1D(d) 
    return @strdict(λ1)
end

function compute_branches(d)
    @unpack v0, T, dt, V, y_init, lyap_threshold, xs, a, λ1 = d # parametros

    ys = range(0, a*8, length = length(xs))
    ind = findall(λ1 .> lyap_threshold)
    img_pos,_,_, m_d_pos, v_d_pos, nb_br_pos, nb_pks_pos, area_pos,_,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)

    ind = findall(λ1 .≤ lyap_threshold)
    img_z,_,_, m_d_z, v_d_z, nb_br_z, nb_pks_z, area_z,_,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)

    ind = 1:length(y_init)
    img_all,_,_, m_d_all, v_d_all, nb_br_all, nb_pks_all, area_all,_,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)
    return @strdict(nb_pks_pos, nb_pks_z, nb_pks_all, m_d_pos, m_d_z, m_d_all, v_d_all, v_d_pos, v_d_z, nb_br_all, nb_br_z, nb_br_pos, area_all, area_pos, area_z)

end

function save_data_decay(v0, V, y_init, T, a, lyap_threshold, dt, xs, num_rays, θ; prefix = "fermi_dec") 
    d = @dict(V,v0,a , y_init, T, lyap_threshold, dt, xs, num_rays, θ)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_lyap, # function
        prefix = "lyap_fermi_dec", # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )

    @unpack λ1 = data
    d = @dict(V,v0,a , y_init, T, lyap_threshold, dt, xs, num_rays, θ, λ1)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_branches, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end 


function count_branches(y_ray, dy, ys; δ = 0.03)
    dy_ray = diff(y_ray)/dy
    hst = zeros(length(ys))
    a = ys[1]; b = ys[end]; dys = step(ys)
    ind = findall(abs.(dy_ray) .< 1.)
    ind2 = findall( a .< y_ray[ind] .< b)
    ray_br = y_ray[ind[ind2]]
    ind_br = ind[ind2]
    br = 0; cnt = 0
    for k in 2:length(ind_br)-1
        if ind_br[k] == ind_br[k+1] - 1  
                cnt += 1 
             # end
        else 
            # the manifold stops, let see if we have long enough stretch
            if cnt ≥ 3
                br += 1
                for y in ray_br[k-cnt:k]
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



function manifold_track(num_rays, xs, ys, potential; b = 0.0003, threshold = 3, ind_i = nothing, rmin = nothing, rmax = nothing, ray_y = nothing)

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
    m_d = zeros(width)
    v_d = zeros(width)
    nb_br = zeros(width)
    nb_pks = zeros(width)
    area = zeros(width)
    max_I = zeros(width)
    dyr = (ray_y[2]-ray_y[1])
    ray_py = zeros(num_rays)
    mean_dy_ray = zeros(num_rays - 1)
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
            dy_ray = diff(ray_y)/dyr 
            mean_dy_ray .= mean_dy_ray .+ abs.(dy_ray)
            if !isnothing(ind_i)
                m_d[xi] = mean(log.(abs.(dy_ray[ind_i[1:end-1]])))
                v_d[xi] = var(log.(abs.(dy_ray[ind_i[1:end-1]])))
                ind = ind_i
            else
                m_d[xi] = mean(log.(abs.(dy_ray)))
                v_d[xi] = var(log.(abs.(dy_ray)))
                ind = findall(abs.(dy_ray) .< 1.)
                # ind = 1:num_rays
            end
            density = kde(ray_y[ind], bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity = pdf(density, ys)*(sim_h / h)
            nb_br[xi], hst = count_branches(ray_y[ind], dyr, ys; δ = 0.001)
            image[:,xi] = hst


            # density = kde(ray_y, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            # intensity = pdf(density, ys)*(sim_h / h)
            t = findmaxima(intensity)
            ind = findall(t[2] .> threshold*bkg) 
            nb_pks[xi] = length(ind)
            ind = findall(intensity .> threshold*bkg) 
            # area[xi] = length(ind)/length(intensity)  
            area[xi] = sum(hst)/length(hst)  
            max_I[xi] = maximum(intensity)  
            xi += 1
        end
    end
    return image, ray_y, (rmin, rmax), m_d, v_d, nb_br, nb_pks, area, max_I, mean_dy_ray
end



v0 = 0.05; dt = 0.01; T = 10000; num_rays = 100000; 
a = 0.2; dot_radius = 0.2*0.25; softness = 0.2; 
lyap_threshold = 2e-3; θ = 0.; 
xs = range(0,20., step = dt)
θ_range = range(0,π/4, length = 40)
y_init = range(-20*a, 20*a, length = num_rays)
V = LatticePotential(a*rotation_matrix(θ), dot_radius, v0; softness=softness)
ys = range(0, a*8, length = length(xs))

# Threads.@threads for k in 1:length(θ_range)
for k in 1:length(θ_range)
    data = save_data_decay(v0, V, y_init, T, a, lyap_threshold, dt, xs, num_rays, θ_range[k]; prefix = "fermi_dec") 
    @unpack nb_pks_pos, nb_pks_z, nb_pks_all, m_d_pos, m_d_z, m_d_all, v_d_all, v_d_pos, v_d_z, nb_br_all, nb_br_z, nb_br_pos, area_all, area_pos, area_z = data
end


