using JLD2 

# Display and compute histograms :
function compute_area(V, a, dt, T; yres = 1000, xres = 10, num_rays = 20000, threshold = 1.5, x0 = 0, smoothing_b = 0.0003)
    yg = sample_midpoints(0, a, yres)
    xg = range(0+x0, T+x0, length = round(Int,xres*T))
    area, max_I, nb_pks, rmax = quasi2d_get_stats(num_rays, dt, xg, yg, V; b = smoothing_b, threshold = threshold, periodic_bnd = false)
    return xg, yg, area, max_I, nb_pks
end

function compute_average_theta(d)
    @unpack V, xres, yres, num_angles, num_rays, a, dt, T, threshold = d # unpack parameters
    xg = 0; yg = 0;
    angles = range(0, π/4, length = num_angles + 1)
    angles = angles[1:end-1]
    p = Progress(num_angles, "Potential calc")
    nb_arr = zeros(round(Int,T*xres), num_angles)
    mx_arr = zeros(round(Int,T*xres), num_angles)
    pks_arr = zeros(round(Int,T*xres), num_angles)
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

function get_fit(xg, yg; Tf = 0, Ti = 0)
    model(x, p) = p[1] .+ p[2] * exp.(p[3] * x)
    mx, ind = findmax(yg)
    if Ti > 0 && Ti < xg[end]
        indi = findfirst(xg .> Ti)
        xdata = xg[indi:end]
        ydata = yg[indi:end]
    else  
        mx, indi = findmax(yg)
        xdata = xg[indi:end]
        ydata = yg[indi:end]
    end
    if Tf > 0 && Tf < xg[end]
        indf = findfirst(xg .> Tf)
        xdata = xg[indi:indf]
        ydata = yg[indi:indf]
    else
        xdata = xg[indi:end]
        ydata = yg[indi:end]
    end
    lb = [0., 0., -1.]
    ub = [100., 200., 0.]   
    p0 = [yg[end], ydata[1], -0.2]
    fit = curve_fit(model, xdata, ydata, p0; lower = lb, upper = ub)
    σ = stderror(fit)
    return fit.param, model, xdata, σ
end


function print_f(x,y, xp, yp,s) 
    fig = Figure(size=(900, 600))
    ax1= Axis(fig[1, 1], xlabel = L"xg", ylabel = L"Area", yticklabelsize = 30, xticklabelsize = 30, ylabelsize = 30, xlabelsize = 30,  titlesize = 30, yscale = Makie.pseudolog10)
    lines!(ax1, x, y, color = :blue, linestyle=:dash)
    lines!(ax1, xp, yp, color = :orange)
    save(plotsdir(string(s,".png")),fig)
end

