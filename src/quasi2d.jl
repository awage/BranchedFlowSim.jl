using Statistics

export quasi2d_num_branches 
export quasi2d_smoothed_intensity
export quasi2d_histogram_intensity
export quasi2d_get_stats
export sample_midpoints 


function entropy(pdf, dθ)
    _xlogx(x) = (x< 0) ? 0. : x*log(x) 
    H = 0
    for p in pdf
        H += _xlogx(p*dθ)
    end
    return H
end

"""
    sample_midpoints(a,b, n)

Divide the interval [a,b] in n equally sized segments and return a Vector of
midpoints of each segment.
"""
function sample_midpoints(a::Real, b::Real, n::Integer)::LinRange{Float64,Int64}
    @assert n > 0
    dx::Float64 = (b - a) / n
    first = a + dx / 2
    last = b - dx / 2
    return LinRange(first, last, n)
end


function quasi2d_num_branches(num_rays, dt, ts, potential::AbstractPotential;
    return_rays=false, rays_span = (0,1))
    ray_y = Vector(LinRange(rays_span[1], rays_span[2], num_rays + 1)[1:num_rays])
    return quasi2d_num_branches(ray_y, dt, ts, potential; return_rays=return_rays)
end

function quasi2d_num_branches(ray_y::AbstractVector{<:Real}, dt::Real,
    ts::AbstractVector{<:Real}, potential::AbstractPotential; return_rays=false)

    # HACK: This uses a lot of memory. Run GC here to try
    # limit memory use before big allocations.
    GC.gc(false)

    ray_y = Vector{Float64}(ray_y)
    num_rays = length(ray_y)
    ray_py = zeros(num_rays)
    num_branches = zeros(length(ts))
    ti = 1
    x = 0.0
    while ti <= length(ts)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        x += dt
        if ts[ti] <= x
            num_branches[ti] = count_branches(ray_y)
            ti += 1
        end
    end
    if return_rays
        return num_branches, ray_y, ray_py
    else
        return num_branches
    end
end

"""
    quasi2d_smoothed_intensity(num_rays, dt, xs, ys, potential; b=0.0030, periodic_bnd = false)

Perform quasi-2D ray tracing simulation and return a matrix of smoothed
intensities.
Returned matrix has size (length(ys), length(xs)).

The situation being modeled is an infinite vertical wavefront starting
at x=0 with p_y=0 and p_x=1 (constant). To cover the area defined by xs and ys,
the actual wavefront being simulated must be taller than ys.
"""
function quasi2d_smoothed_intensity(num_rays::Integer, dt, rs, θs, potential; b=0.0030, x0 = 0, y0 = 0, threshold = 1.5)
    width = length(rs)
    height = length(θs)
    area = zeros(length(rs))
    max_I = zeros(length(rs))
    area = zeros(length(rs))
    image = zeros(height, width)
    ray_θ = collect(LinRange(-pi, pi, num_rays))
    ray_pθ = zeros(num_rays)
    r = rs[1]; ri = 1; dt = step(rs); dθ = step(θs)
    intensity = zeros(length(θs), length(rs))
    boundary = (
        -pi ,
        pi 
    )
    while r < rs[end] 
        # for k in eachindex(ray_θ)
        #     θ = ray_θ[k]; pθ = ray_pθ[k]
        #     x = r*cos(θ); y = r*sin(θ)
        #     F = sum(force(potential, x + x0, y + y0).*[-y, x])
        #     ray_pθ[k] = pθ + dt * F
        #     ray_θ[k]  = θ + dt*ray_pθ[k]/r^2
        # end
        # Kick
        ray_pθ .+=  dt .* force_polar.(Ref(potential), r, ray_θ; x0, y0)
        # drift
        ray_θ .+= dt .* ray_pθ/r^2
        r += dt
        ray_θ .= rem2pi.(ray_θ, RoundNearest) 

        while ri <= length(rs) && rs[ri] <= r
            # Compute intensity
            density = kde(ray_θ, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity[:, ri] = pdf(density, θs)*2π
            ind = findall(intensity[:,ri] .> threshold) 
            area[ri] = length(ind)/length(intensity[:,ri])
            max_I[ri] = maximum(intensity[:,ri])  
            ri += 1
        end
    end
    return intensity, area, max_I
end

"""
    quasi2d_histogram_intensity(ray_y, xs, ys, potential)

Runs a simulation across a grid defined by `xs` and `ys`, returning a
matrix of flow intensity computed as a histogram (unsmoothed).
"""
function quasi2d_histogram_intensity(num_rays, rs, θs, potential; normalized = false, periodic_bnd = false, x0 = 0, y0 =0, threshold = 1.5)
    width = length(rs)
    height = length(θs)
    area = zeros(length(rs))
    image = zeros(height, width)
    ray_θ = collect(LinRange(-pi, pi, num_rays))
    ray_pθ = zeros(num_rays)
    r = rs[1]; ri = 1; dt = step(rs); dθ = step(θs)
    while r < rs[end] 
        for k in eachindex(ray_θ)
            θ = ray_θ[k]; pθ = ray_pθ[k]
            x = r*cos(θ); y = r*sin(θ)
            F = sum(force(potential, x + x0, y + y0).*[-y, x])
            ray_pθ[k] = pθ + dt * F
            ray_θ[k]  = θ + dt*ray_pθ[k]/r^2
        end
        r += dt
        ray_θ .= rem2pi.(ray_θ, RoundNearest) 
        for θ ∈ ray_θ
            θi = 1 + round(Int, (θ - (-pi)) / dθ)
            if θi >= 1 && θi <= height
                image[θi, ri] += 1
            end
        end
        ri += 1
    end
    if normalized 
        for k in 1:length(rs)
            image[:,k] .= image[:,k]/sum(image[:,k])/dθ
            ind = findall(image[:,k]*2*pi .> threshold)
            area[k] = length(ind)/length(image[:,1])    
        end
    end
    # return image * height / num_rays
    return image,area 
end


# This function computes average area and the maximum intensity
function quasi2d_get_stats(num_rays::Integer, dt, rs::AbstractVector, θs::AbstractVector, potential; b=0.003, threshold = 1.5,  x0 = 0, y0 = 0)
    dθ = step(θs)
    width = length(rs)
    height = length(θs)
    entrop = zeros(length(rs))
    max_I = zeros(length(rs))
    ray_θ = collect(LinRange(-pi, pi, num_rays))
    ray_pθ = zeros(num_rays)
    r = rs[1]; ri = 1; dt = step(rs); dθ = step(θs)
    intensity = zeros(length(θs))
    boundary = (
        -pi ,
        pi 
    )
    while r < rs[end] 
        # for k in eachindex(ray_θ)
        #     θ = ray_θ[k]; pθ = ray_pθ[k]
        #     x = r*cos(θ); y = r*sin(θ)
        #     F = sum(force(potential, x + x0, y + y0).*[-y, x])
        #     ray_pθ[k] = pθ + dt * F
        #     ray_θ[k]  = θ + dt*ray_pθ[k]/r^2
        # end
        # F = force_polar.(Ref(potential), Ref(r), ray_θ; x0, y0)
        # Kick
        ray_pθ .+=  dt .* force_polar.(Ref(potential), Ref(r), ray_θ; x0, y0)
        # drift
        ray_θ .+= dt .* ray_pθ/r^2
        r += dt

        ray_θ .= rem2pi.(ray_θ, RoundNearest) 

        while ri <= length(rs) && rs[ri] <= r
            # Compute intensity
            density = kde(ray_θ, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity = pdf(density, θs)*2π
            ind = findall(intensity .> threshold) 
            entrop[ri] = entropy(intensity, dθ)
            # area[ri] = length(ind)/length(intensity)
            max_I[ri] = maximum(intensity)  
            ri += 1
        end
    end
    return entrop, max_I
end

