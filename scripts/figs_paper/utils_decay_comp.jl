
# Display and compute histograms :
function compute_area(V, a, dt, T; yres = 1000, xres = 10, num_rays = 20000, threshold = 1.5, x0 = 0, smoothing_b = 0.003)
    yg = sample_midpoints(0, a, yres)
    xg = range(0+x0, T+x0, length = round(Int,xres*T))
    area, max_I, nb_pks, rmax = quasi2d_get_stats(num_rays, dt, xg, yg, V; b = smoothing_b, threshold = threshold, periodic_bnd = false)
    # @show x0, rmax, T
    return xg, yg, area, max_I, nb_pks
end

function compute_average_theta(d)
    @unpack V, xres, yres, num_angles, num_rays, a, dt, T, threshold = d # unpack parameters
    xg = 0; yg = 0;
    angles = range(0, π/4, length = num_angles + 1)
    angles = angles[1:end-1]
    p = Progress(num_angles, "Potential calc")
    nb_arr = zeros(T*xres, num_angles)
    mx_arr = zeros(T*xres, num_angles)
    pks_arr = zeros(T*xres, num_angles)
    Threads.@threads for i ∈ 1:num_angles
        potential = V(angles[i])
        xg, yg, area, max_I, nb_pks = compute_area(potential, a, dt, T; xres, yres,  num_rays, threshold)
        nb_arr[:, i] = area
        mx_arr[:, i] = max_I
        pks_arr[:, i] = nb_pks
        next!(p)
    end
    return @strdict(xg, yg, nb_arr, mx_arr, pks_arr)
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
    mx, ind = findmax(yg)
    xdata = xg[ind:end]
    ydata = yg[ind:end]
    p0 = [yg[end], 0.5*xdata[1], -0.2]
    fit = curve_fit(model, xdata, ydata, p0)
    return fit.param, model, xdata
end

