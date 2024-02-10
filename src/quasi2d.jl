using Statistics
export quasi2d_num_branches 
export quasi2d_smoothed_intensity
export quasi2d_histogram_intensity
export quasi2d_get_stats
export sample_midpoints 

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
function quasi2d_smoothed_intensity(num_rays::Integer, dt, xs, ys, potential; b=0.0030, periodic_bnd = false, x0 , y0)
    h = length(ys) * (ys[2] - ys[1])
    τ = 0 
    if periodic_bnd == true
        rmin = ys[1]; rmax = ys[end]; τ = ys[end] - ys[1]
    else
        rmin,rmax = quasi2d_compute_front_length(1024, dt, xs, ys, potential)
    end
    ray_y = LinRange(rmin, rmax, num_rays)
    sim_h = (ray_y[2]-ray_y[1]) * length(ray_y)
    ints = quasi2d_smoothed_intensity(ray_y, dt, xs, ys, potential, b; periodic_bnd, τ, x0, y0) 
    return ints, (rmin, rmax)
end

function quasi2d_smoothed_intensity(
    ray_y::AbstractVector{<:Real},
    dt::Real,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    potential,
    b; 
    periodic_bnd = false, 
    τ = 0,
    x0, y0 
)
    dy = ys[2] - ys[1]
    rmin = ys[1]; rmax = ys[end]
    y_end = ys[1] + length(ys) * (ys[2] - ys[1])
    ray_y = Vector{Float64}(ray_y)
    num_rays = length(ray_y)
    ray_py = zeros(num_rays)
    xi = 1; x = dt
    intensity = zeros(length(ys), length(xs))
    boundary = (
        -pi ,
        pi 
    )
    while xi <= length(xs)
        # kick and drift polar
        for k in eachindex(ray_y)
            y = ray_y[k]
            F = force_x(potential, x*cos(y) + x0, x*sin(y) + y0)*(-x*sin(y)) 
                + force_y(potential, x*cos(y) + x0, y*sin(y) + y0)*x*cos(y)
            ray_py[k] += dt * F
            # drift
            ray_y[k] += dt*ray_py[k]/x^2
        end
        x += dt
        ray_y .= rem2pi.(ray_y, RoundNearest) 
        # # kick
        # ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # # drift
        # ray_y .+= dt .* ray_py
        # x += dt
        # if periodic_bnd == true
            # for k in eachindex(ray_y)
            #     while ray_y[k] < rmin; ray_y[k] += τ; end
            #     while ray_y[k] > rmax; ray_y[k] -= τ; end
            # end
        # end
        while xi <= length(xs) && xs[xi] <= x
            # Compute intensity
            density = kde(ray_y, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity[:, xi] = pdf(density, ys)*2π
            xi += 1
        end
    end
    return intensity
end

# First shoot some rays to figure out how wide they spread in the given time.
# Then use this to decide how tall to make the initial wavefront.
function quasi2d_compute_front_length(num_canary_rays, dt, xs::AbstractVector{<:Real}, ys, potential)
    T = xs[end] - xs[1]
    return quasi2d_compute_front_length(num_canary_rays, dt, T, ys, potential)
end

function quasi2d_compute_front_length(num_canary_rays, dt, T::Real, ys, potential)
    canary_ray_y = Vector(sample_midpoints(ys[1], ys[end], num_canary_rays))
    ray_y = copy(canary_ray_y)
    ray_py = zero(ray_y)
    for x ∈ 0:dt:T
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        ray_y .+= dt .* ray_py
    end
    max_travel = maximum(abs.(ray_y - canary_ray_y))
    # Front coordinates:
    rmin = ys[1] - max_travel
    rmax = ys[end] + max_travel
    return rmin, rmax
end
            
     

"""
    quasi2d_histogram_intensity(ray_y, xs, ys, potential)

Runs a simulation across a grid defined by `xs` and `ys`, returning a
matrix of flow intensity computed as a histogram (unsmoothed).
"""
function quasi2d_histogram_intensity(num_rays, xs, ys, potential; normalized = true, periodic_bnd = false, x0 = 0, y0 =0)
    dt = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    
    # Compute the spread beforehand
    # if periodic_bnd == true
    #     rmin = ys[1]; rmax = ys[end]; T = ys[end] - ys[1]
    # else
    #     rmin,rmax = quasi2d_compute_front_length(1024, dt, xs, ys, potential)
    # end

    width = length(xs)
    height = length(ys)
    area = zeros(length(xs))
    image = zeros(height, width)
    ray_y = collect(LinRange(-pi, pi, num_rays))
    ray_py = zeros(num_rays)
    for (xi, x) ∈ enumerate(xs)
        for k in eachindex(ray_y)
            y = ray_y[k]
            F = force_x(potential, x*cos(y) + x0, x*sin(y) + y0)*(-x*sin(y)) 
                + force_y(potential, x*cos(y) + x0, y*sin(y) + y0)*x*cos(y)
            ray_py[k] += dt * F
            # drift
            ray_y[k] += dt*ray_py[k]/x^2
        end
        x += dt
        # # kick
        ray_y .= rem2pi.(ray_y, RoundNearest) 
        # ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # # drift
        # ray_y .+= dt .* ray_py
        # # Collect
        for y ∈ ray_y
            # if periodic_bnd == true
            #     while y < rmin; y += T; end
            #     while y > rmax; y -= T; end
            # end
            yi = 1 + round(Int, (y - (-pi)) / dy)
            if yi >= 1 && yi <= height
                image[yi, xi] += 1
            end
        end
    end
    if normalized 
        for k in 1:length(xs)
            image[:,k] .= image[:,k]/sum(image[:,k])/dy
            ind = findall(image[:,k]*2*pi .> 1.5)
            area[k] = length(ind)/length(image[:,1])    
        end
    end
    # return image * height / num_rays
    return image,area 
end


# This function computes average area and the maximum intensity
function quasi2d_get_stats(num_rays::Integer, dt, xs::AbstractVector, ys::AbstractVector, potential; b=0.003, threshold = 1.5, periodic_bnd = false, x0 = 0, y0 = 0)
    τ = 0. 
    # if periodic_bnd == true
        # rmin = ys[1]; rmax = ys[end]; τ = ys[end] - ys[1]
    # else
        # rmin,rmax = quasi2d_compute_front_length(1024, dt, xs[end], ys, potential)
    # end
    ray_y = LinRange(ys[1], ys[end], num_rays)
    area, max_I = quasi2d_smoothed_intensity_stats(ray_y, dt, xs, ys, potential, b, threshold, periodic_bnd, τ, x0, y0) 
    return area, max_I
end

# In this version you provide dt, T and x0. 
function quasi2d_get_stats(num_rays::Integer, dt, T::Real, ys::AbstractVector, potential; b=0.003, threshold = 1.5, x0 = 0, y0 = 0,  periodic_bnd = false)
    τ = 0. 
    # if periodic_bnd == true
    #     rmin = ys[1]; rmax = ys[end]; τ = ys[end] - ys[1]
    # else
    #     rmin,rmax = quasi2d_compute_front_length(2*1024, dt, T, ys, potential)
    # end
    # ray_y = LinRange(rmin, rmax, num_rays)
    ray_y = LinRange(ys[1], ys[end], num_rays)

    xs = range(0, T, step = dt) 
    area, max_I = quasi2d_smoothed_intensity_stats(ray_y, dt, xs, ys, potential, b, threshold, periodic_bnd, τ, x0, y0) 
    return xs, area, max_I, (rmin, rmax)
end

function quasi2d_smoothed_intensity_stats(
    ray_y::AbstractVector{<:Real},
    dt::Real,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    potential::AbstractPotential,
    b::Real, 
    threshold::Real, 
    periodic_bnd::Bool,
    τ::Real,
    x0, y0 
)
    dy = ys[2] - ys[1]
    rmin = ys[1]; rmax = ys[end];
    h = length(ys) * (ys[2] - ys[1])
    num_rays = length(ray_y)
    sim_h = (ray_y[2]-ray_y[1]) * num_rays
    ray_y = Vector{Float64}(ray_y)
    ray_py = zeros(num_rays)
    area = zeros(length(xs))
    max_I = zeros(length(xs))
    intensity = zeros(length(ys))
    background = (num_rays/length(ys));  bckgnd_density = background/(dy*num_rays)
    xi = 1; x = dt
    # boundary = (
    #     ys[1] - dy - 4*b,
    #     ys[end] + dy + 4*b,
    # )
    boundary = (-2*pi, 2*pi)
    while x <= xs[end] + dt

        
        # kick and drift polar
        for k in eachindex(ray_y)
            r = x; θ = ray_y[k]; 
            xr = r*cos(θ); yr = r*sin(θ)
            F = sum(force(potential, xr + x0, yr + y0).*[-yr ,  xr])
            ray_py[k] += dt * F
            # drift
            ray_y[k] += dt*ray_py[k]/x^2
        end
        x += dt

        ray_y .= rem2pi.(ray_y, RoundNearest) 

        # if periodic_bnd == true
        #     for k in eachindex(ray_y)
        #         while ray_y[k] < rmin; ray_y[k] += τ; end
        #         while ray_y[k] > rmax; ray_y[k] -= τ; end
        #     end
        # end

        while xi <= length(xs) &&  xs[xi] <= x 
            density = kde(ray_y, bandwidth=b, npoints=16 * 1024, boundary=boundary)
            intensity = pdf(density, ys)*2π
            ind = findall(intensity .> threshold*bckgnd_density) 
            # area[xi] = abs(mean(exp.(im*ray_y)))  
            area[xi] = length(ind)/length(intensity)
            max_I[xi] = maximum(intensity)  
            xi += 1
        end
    end
    return area, max_I
end

