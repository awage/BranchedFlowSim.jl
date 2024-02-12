
# Display and compute histograms :
function compute_area(V, a, dt, T; yres = 1000, xres = 10, num_rays = 20000, threshold = 1.5, x0 = 0, smoothing_b = 0.003)
    θg = range(-pi, pi, length = yres)
    rg = range(dt, T, length = round(Int,xres*T))
    area, max_I = quasi2d_get_stats(num_rays, dt, rg, θg, V; b = smoothing_b, threshold, x0)
    return rg, θg, area, max_I
end

function compute_average_theta(d)
    @unpack V, xres, yres, num_angles, num_rays, a, dt, T, threshold = d # unpack parameters
    rg = 0; θg = 0;
    x0 = range(0, 1, num_angles)
    p = Progress(num_angles, "Potential calc")
    nb_arr = zeros(T*xres, num_angles)
    mx_arr = zeros(T*xres, num_angles)
    potential = V(rand()*2π)
    Threads.@threads for i ∈ 1:num_angles
        rg, θg, area, max_I = compute_area(potential, a, dt, T; xres, yres,  num_rays, threshold, x0 = x0[i])
        nb_arr[:, i] = area
        mx_arr[:, i] = max_I
        next!(p)
    end
    return @strdict(rg, θg, nb_arr, mx_arr)
end


function get_data_decay(V, a, num_angles, num_rays, T, threshold, dt, xres, yres; prefix = "decay")
    d = @dict(V, a, num_angles, num_rays, T, threshold, dt, xres, yres)
    data, file = produce_or_load(
        datadir("storage"), # path
        d, # container for parameter
        compute_average_theta, # function
        prefix = prefix, # prefix for savename
        force = false, # true for forcing sims
        wsave_kwargs = (;compress = true)
    )
    return data
end

function get_fit(xg, yg)
    model(x, p) = p[1] .+ p[2] * exp.(p[3] * x)
    # model(x, p) = p[1] .+ p[2] * x.^p[3]
    # mx, ind = findmax(yg)
    ind = findall(xg .> 2); ind = ind[1]
    xdata = xg[ind:end]
    ydata = yg[ind:end]
    p0 = [yg[end], 2., 0.01]
    fit = curve_fit(model, xdata, ydata, p0)
    # return fit.param, model, xdata
    return [yg[end], fit.param[2], fit.param[3]], model, xdata
end
