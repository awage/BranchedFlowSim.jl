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
    img_pos,_,_, m_d_pos, v_d_pos, nb_br_pos, nb_pks_pos, area_pos,max_pos,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)

    ind = findall(λ1 .≤ lyap_threshold)
    img_z,_,_, m_d_z, v_d_z, nb_br_z, nb_pks_z, area_z,max_z,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)

    ind = 1:length(y_init)
    img_all,_,_, m_d_all, v_d_all, nb_br_all, nb_pks_all, area_all, max_all,_ = manifold_track(length(y_init), xs, ys, V; b = 0.0003, ray_y = collect(y_init), ind_i = ind)
    return @strdict(nb_pks_pos, nb_pks_z, nb_pks_all, m_d_pos, m_d_z, m_d_all, v_d_all, v_d_pos, v_d_z, nb_br_all, nb_br_z, nb_br_pos, area_all, area_pos, area_z, max_pos, max_z, max_all)

end

function save_data_decay(v0, V, y_init, T, a, lyap_threshold, dt, xs, num_rays, θ; prefix = "cos_dec") 
    d = @dict(V,v0,a , y_init, T, lyap_threshold, dt, xs, num_rays, θ)  
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_lyap, # function
        prefix = "lyap_cos_dec", # prefix for savename
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



function get_histogram(ray_y, ys, threshold)
dy = step(ys)
bkg = 1/(dy*length(ys))
hst = zeros(length(ys))
    for y ∈ ray_y
        yi = 1 + round(Int, (y - ys[1]) / dy)
        if yi >= 1 && yi <= length(ys)
            hst[yi] += 1
        end
    end
    ind = findall(hst .> dy*bkg*threshold*sum(hst))
    return length(ind)/length(hst)
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
            nb_br[xi], hst = count_branches(ray_y[ind], dyr, ys)
            image[:,xi] = hst
            # density = kde(ray_y, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            # intensity = pdf(density, ys)*(sim_h / h)
            t = findmaxima(intensity)
            indt = findall(t[2] .> threshold*bkg) 
            nb_pks[xi] = length(indt)
            # ind = findall(intensity .> threshold*bkg) 
            # area[xi] = length(ind)/length(intensity)  
            
            area[xi] = sum(hst)/length(hst)  
            max_I[xi] = get_histogram(ray_y[ind], ys, threshold)
            xi += 1
        end
    end
    return image, ray_y, (rmin, rmax), m_d, v_d, nb_br, nb_pks, area, max_I, mean_dy_ray
end



v0 = 0.1; dt = 0.01; T = 10000; num_rays = 200000; 
a = 0.2; dot_radius = 0.2*0.25; softness = 0.2; 
lyap_threshold = 2e-3; θ = 0.; nθ = 60
xs = range(0,20., step = dt)
θ_range = range(0,π/4, length = nθ)
y_init = range(-20*a, 20*a, length = num_rays)
ys = range(0, a*8, length = length(xs))

mean_nb_br_z = zeros(nθ, length(xs))
mean_nb_br_all = zeros(nθ, length(xs))
mean_nb_br_pos = zeros(nθ, length(xs))

mean_area_z = zeros(nθ, length(xs))
mean_area_all = zeros(nθ, length(xs))
mean_area_pos = zeros(nθ, length(xs))

mean_area2_z = zeros(nθ, length(xs))
mean_area2_all = zeros(nθ, length(xs))
mean_area2_pos = zeros(nθ, length(xs))

mean_nb_pks_z = zeros(nθ, length(xs))
mean_nb_pks_all = zeros(nθ, length(xs))
mean_nb_pks_pos = zeros(nθ, length(xs))

mean_m_z = zeros(nθ, length(xs))
mean_m_all = zeros(nθ, length(xs))
mean_m_pos = zeros(nθ, length(xs))
mean_v_z = zeros(nθ, length(xs))
mean_v_all = zeros(nθ, length(xs))
mean_v_pos = zeros(nθ, length(xs))

# Threads.@threads for k in 1:length(θ_range)
for k in 1:length(θ_range)
    V = RotatedPotential(θ_range[k],
        fermi_dot_lattice_cos_series(1,
        a, dot_radius, v0; softness))
    data = save_data_decay(v0, V, y_init, T, a, lyap_threshold, dt, xs, num_rays, θ_range[k]; prefix = "cos_dec") 
    @unpack nb_pks_pos, nb_pks_z, nb_pks_all, m_d_pos, m_d_z, m_d_all, v_d_all, v_d_pos, v_d_z, nb_br_all, nb_br_z, nb_br_pos, area_all, area_pos, area_z, max_pos, max_z, max_all = data
        mean_nb_br_pos[k,:] = nb_br_pos
        mean_nb_br_z[k,:] = nb_br_z
        mean_nb_br_all[k,:] = nb_br_all

        mean_nb_pks_pos[k,:] = nb_pks_pos
        mean_nb_pks_z[k,:] = nb_pks_z
        mean_nb_pks_all[k,:] = nb_pks_all

        mean_area_pos[k,:] = area_pos
        mean_area_z[k,:] = area_z
        mean_area_all[k,:] = area_all

        mean_area2_pos[k,:] = max_pos
        mean_area2_z[k,:] = max_z
        mean_area2_all[k,:] = max_all

        mean_m_pos[k,:] = m_d_pos
        mean_m_z[k,:] = m_d_z
        mean_m_all[k,:] = m_d_all

        mean_v_pos[k,:] = v_d_pos
        mean_v_z[k,:] = v_d_z
        mean_v_all[k,:] = v_d_all
end

    # mm = vec(mean(mean_m_pos, dims = 1))
    # v = vec(mean(mean_v_pos, dims = 1))
    # br = vec(mean(mean_nb_br_pos, dims = 1))
    # p,m,x = get_fit_lin(xs[10:250], mm[10:250]) 
    # α_vec[jj] = p[2] 
    # p,m,x = get_fit_lin(xs[10:250], v[10:250]) 
    # β_vec[jj] = p[2]*2
    # p,m,x = get_fit_exp(xs, br) 
    # ω_vec[jj] = p[2]


    fig = Figure()
    a1 = Axis(fig[1,1], ylabel = L"n_{b}(t)", xlabel = "t")
    lines!(a1, xs, vec(mean(mean_nb_br_all, dims = 1)), color = :black, label = "all")
    lines!(a1, xs, vec(mean(mean_nb_br_z, dims = 1)), color = :blue, label = L"\lambda \simeq 0")
    lines!(a1, xs, vec(mean(mean_nb_br_pos, dims = 1)), color = :red, label = L"\lambda > 0")
    # p,m,x = get_fit_exp(xs, br) 
    # lines!(a1, x, m(x,p), color = :red, linestyle = :dash, label = L"\lambda > 0, \textrm{ fit}")
   axislegend()
   s = savename("cos_dec_nb",@dict(v0),"png") 
   save(plotsdir(s), fig)

    fig = Figure()
    a1 = Axis(fig[1,1], ylabel = L"n_{b}(t)", xlabel = "t")
    lines!(a1, xs, vec(mean(mean_nb_pks_all, dims = 1)), color = :black, label = "all")
    lines!(a1, xs, vec(mean(mean_nb_pks_z, dims = 1)), color = :blue, label = L"\lambda \simeq 0")
    lines!(a1, xs, vec(mean(mean_nb_pks_pos, dims = 1)), color = :red, label = L"\lambda > 0")
   axislegend()
   s = savename("cos_dec_pks",@dict(v0),"png") 
   save(plotsdir(s), fig)

    fig = Figure()
    a1 = Axis(fig[1,1], ylabel = L"m(t)", xlabel = "t")
    lines!(a1, xs, vec(mean(mean_m_all, dims = 1)), color = :black, label = "all")
    lines!(a1, xs, vec(mean(mean_m_z, dims = 1)), color = :blue, label = L"\lambda \simeq 0")
    lines!(a1, xs, vec(mean(mean_m_pos, dims = 1)), color = :red, label = L"\lambda > 0")
    # p,m,x = get_fit_lin(xs[10:300], mm[10:300]) 
    # lines!(a1, x, m(x,p), color = :red, linestyle = :dash, label = L"\lambda > 0, \textrm{ fit}")
   axislegend()
   s = savename("cos_dec_m",@dict(v0),"png") 
   save(plotsdir(s), fig)

    fig = Figure()
    a1 = Axis(fig[1,1], ylabel = L"v(t)", xlabel = "t")
    lines!(a1, xs, vec(mean(mean_v_all, dims = 1)), color = :black, label = "all")
    lines!(a1, xs, vec(mean(mean_v_z, dims = 1)), color = :blue, label = L"\lambda \simeq 0")
    lines!(a1, xs, vec(mean(mean_v_pos, dims = 1)), color = :red, label = L"\lambda > 0")
   axislegend()
   s = savename("cos_dec_v",@dict(v0),"png") 
   save(plotsdir(s), fig)

    fig = Figure()
    a1 = Axis(fig[1,1], ylabel = L"f_{b}(t)", xlabel = "t")
    lines!(a1, xs, vec(mean(mean_area_all, dims = 1)), color = :black, label = "all")
    lines!(a1, xs, vec(mean(mean_area_z, dims = 1)), color = :blue, label = L"\lambda \simeq 0")
    lines!(a1, xs, vec(mean(mean_area_pos, dims = 1)), color = :red, label = L"\lambda > 0")
   axislegend()
   ylims!(a1, 0., 0.2)
   xlims!(a1, 0., 20.)
   s = savename("cos_area",@dict(v0),"png") 
   save(plotsdir(s), fig)

    fig = Figure()
    a1 = Axis(fig[1,1], ylabel = L"f_{b}(t)", xlabel = "t")
    lines!(a1, xs, vec(mean(mean_area2_all, dims = 1)), color = :black, label = "all")
    lines!(a1, xs, vec(mean(mean_area2_z, dims = 1)), color = :blue, label = L"\lambda \simeq 0")
    lines!(a1, xs, vec(mean(mean_area2_pos, dims = 1)), color = :red, label = L"\lambda > 0")
   axislegend()
   ylims!(a1, 0., 0.2)
   xlims!(a1, 0., 20.)
   s = savename("cos_area2",@dict(v0),"png") 
   save(plotsdir(s), fig)
