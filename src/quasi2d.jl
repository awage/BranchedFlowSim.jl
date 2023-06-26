using HDF5
using Statistics

export quasi2d_num_branches, quasi2d_visualize_rays
export quasi2d_compute_and_save_num_branches
export quasi2d_num_branches_parallel
export quasi2d_intensity

function quasi2d_num_branches(num_rays, dt, ts, potential::AbstractPotential;
    return_rays=false)
    ray_y = Vector(LinRange(0, 1, num_rays + 1)[1:num_rays])
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

function quasi2d_intensity(num_rays, dt, xs, ys, potential; b=0.0030)
    h = length(ys) * (ys[2] - ys[1])
    T = xs[end] - xs[1]
    # Make rays start from a region 3 times ys, so that we can measure
    # intensity in the middle.
    extra = T * h / 2
    ray_y = LinRange(ys[1] - extra, ys[end] + extra, num_rays)
    return quasi2d_intensity(ray_y, dt, xs, ys, potential, b) * (1 + 2extra / h)
end

function quasi2d_intensity(
    ray_y::AbstractVector{<:Real},
    dt::Real,
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    potential,
    b=0.0030
)
    y_start = ys[1]
    y_end = ys[1] + length(ys) * (ys[2] - ys[1])
    ray_y = Vector{Float64}(ray_y)
    num_rays = length(ray_y)
    ray_py = zeros(num_rays)
    xi = 1
    x = 0.0
    intensity = zeros(length(ys), length(xs))
    while xi <= length(xs)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        x += dt
        if xs[xi] <= x
            # Compute intensity
            density = kde(ray_y, bandwidth=b, npoints=16 * 1024, boundary=(y_start, y_end))
            intensity[:, xi] = pdf(density, ys)
            xi += 1
        end
    end
    return intensity
end



"""
    quasi2d_visualize_rays(path, xs, ys, potential;
        triple_y=false,
        height=(if triple_y 3*length(ys) else length(ys) end)
     )

TBW
"""
function quasi2d_visualize_rays(path, num_rays, sim_width, potential;
    triple_y=false
)
    pixel_scale = 600
    sim_height = 1
    height = round(Int,
        if triple_y
            3 * pixel_scale
        else
            pixel_scale
        end
    )
    width = round(Int, sim_width * pixel_scale)
    xs = LinRange(0, sim_width, width)
    dt = xs[2] - xs[1]
    image = zeros(height, width)
    # Height of one pixel in simulation length units
    pixel_h = 1 / pixel_scale
    ray_y = Vector(LinRange(0, 1, num_rays + 1)[1:end-1])
    ypixels = LinRange(0, ray_y[end], height)
    ray_py = zeros(num_rays)
    for (xi, x) ∈ enumerate(xs)
        # kick
        ray_py .+= dt .* force_y.(Ref(potential), x, ray_y)
        # drift
        ray_y .+= dt .* ray_py
        # Collect
        for y ∈ ray_y
            yi = 1 + round(Int, y / pixel_h)
            if triple_y
                yi += height ÷ 3
            end
            if yi >= 1 && yi <= height
                image[yi, xi] += 1
            end
        end
    end
    scene = Scene(camera=campixel!, resolution=size(image'))
    if triple_y
        ypixels = LinRange(ypixels[1] - sim_height, ypixels[end] + sim_height, height)
    end
    pot_values = [
        potential(x, y) for x ∈ xs, y ∈ ypixels
    ]
    heatmap!(scene, pot_values; colormap=:Greens_3)
    heatmap!(scene, image'; colormap=[
            RGBA(0, 0, 0, 0), RGBA(0, 0, 0, 1),
        ],
        colorrange=(0, 15 * num_rays / pixel_scale)
    )
    save(path, scene)
end

function quasi2d_num_branches_parallel(
    num_rays, dt, ts, potentials, progress_text="")

    p = Progress(length(potentials), progress_text)
    nb_arr = zeros(length(ts), length(potentials))
    Threads.@threads for i ∈ 1:length(potentials)
        potential = potentials[i]
        nb_arr[:, i] =
            quasi2d_num_branches(num_rays, dt, ts, potential)
        next!(p)
    end
    return nb_arr
end

"""
    quasi2d_compute_and_save_num_branches(
    h5_fname, num_rays, dt, ts, potentials::AbstractArray{<:AbstractPotential},
    pot_params::NamedTuple=())
    h5_fname, num_rays, dt, ts, potentials, pot_params=())

Computes the number of branches for given potentials and stores
the results in a HDF5 file at `h5_fname`. pot_a
"""
function quasi2d_compute_and_save_num_branches(
    h5_fname, num_rays, dt, ts, potentials::AbstractArray{<:AbstractPotential},
    pot_params::NamedTuple=())
    # TODO: Skip if file already exists with given parameters

    nb_all = quasi2d_num_branches_parallel(num_rays, dt, ts, potentials,
        basename(h5_fname))
    nb_mean = vec(mean(nb_all, dims=2))

    h5open(h5_fname, "w") do f
        f["dt"] = Float64(dt)
        f["num_rays"] = Int64(num_rays)
        f["ts"] = Vector{Float64}(ts)
        f["nb_mean"] = nb_mean
        f["nb_all"] = nb_all
        pot = create_group(f, "potential")
        for (k, v) ∈ pairs(pot_params)
            pot[String(k)] = v
        end
    end
end