using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using StaticArrays
using Peaks




# Compute average over all angles
function get_branch_nb(r; num_rays = 10000, dt = 1/512, ts = LinRange(0,6,200), sim_height = 1)
        potential = CosMixedPotential(r, 2π)
        nb = quasi2d_num_branches(num_rays, dt, ts, potential; rays_span = (-pi, pi)) / sim_height
    return nb
end

function plt_branch_number(r)
    sim_width = 200
    ts = LinRange(0,sim_width,200)

    @time " sim" nb_branch = get_branch_nb(r; num_rays = 20000,  ts = ts )

    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"N_b",
        title=LaTeXString("Number of branches"), limits=((0, sim_width), (0, nothing)))

    lines!(ax, ts, nb_branch, label=L"Cos mixed r")
    axislegend(ax, position=:lt)
    display(fig)
end


function count_area(hst, threshold) 
    brchs = zeros(size(hst)[1])
    num_rays = sum(hst[:,1])
    for (k,h) in enumerate(eachcol(hst))
        t = findall(h .> threshold) 
        brchs[k] = length(t)/length(h)  
    end
    return brchs
end

function count_peaks(hst, threshold)
    pks = zeros(size(hst)[1])
    for (k,h) in enumerate(eachcol(hst))
        t = findmaxima(h) 
        ind = findall(t[2] .> threshold)
        pks[k] = length(ind)  
    end
    return pks
end

function count_heights(hst, threshold)
    pks = zeros(size(hst)[1])
    for (k,h) in enumerate(eachcol(hst))
        t = findmaxima(h) 
        ind = findall(t[2] .> threshold)
        pks[k] = sum(t[2][ind])  
    end
    return pks
end

# b = Vector{Vector{Float64}}()
# for tr in 10 .^range(-1,1,length=10)
#     push!(b,count_area(I, tr))
# end
# plot(vcat(b...))


function _get_histograms(d)
    @unpack v0, res, num_rays, a = d # unpack parameters
    xg = range(0, 60*a, length = res) 
    yg = range(0, 1, length = res)  
    dt = xg[2] - xg[1]
    pot2 = correlated_random_potential(60*a,10*a, a, v0, rand(UInt))
    I = quasi2d_histogram_intensity(num_rays, xg, yg, pot2, (0,1); normalized = true)
    Is, rmax = quasi2d_intensity(num_rays, dt,  xg, yg, pot2)
    return @strdict(xg, yg, I, Is, rmax)
end

function get_stats(I, threshold)
    b = count_area(I, threshold)
    c = count_peaks(I, threshold)
    d = count_heights(I, threshold)
    return b,c,d
end


res = 1000; num_rays = 500000; a = 0.1; v0 = 0.1
d = @dict(res,num_rays, v0, a) # parametros

data, file = produce_or_load(
    datadir("./storage"), # path
    d, # container for parameter
    _get_histograms, # function
    prefix = "random_bf", # prefix for savename
    force = false, # true for forcing sims
    wsave_kwargs = (;compress = true)
)

@unpack xg,yg,I,Is

# threshold = 8.
for threshold in 1:0.1:2
    b = count_area(I, threshold)
    c = count_peaks(I, threshold)
    d = count_heights(I, threshold)

    fig = Figure(resolution=(800, 600))
    ax1= Axis(fig[1, 1])#, yscale = log) 
    ax2= Axis(fig[1, 1])# yscale = log) 
    ax3= Axis(fig[1, 1])#, yscale = log) 
    lines!(ax1, xg, b, color = :blue)
    lines!(ax2, xg, c, color = :black)
    lines!(ax3, xg, d, color = :red)
    save(string("../outputs/plot_hist_correlated_",threshold, ".png"),fig)
end

# display(fig)

for threshold in 1:0.1:2
# threshold = 2.
    b = count_area(Is, threshold)
    c = count_peaks(Is, threshold)
    d = count_heights(Is, threshold)

    fig = Figure(resolution=(800, 600))
    ax1= Axis(fig[1, 1])#, yscale = log) 
    ax2= Axis(fig[1, 1])# yscale = log) 
    ax3= Axis(fig[1, 1])#, yscale = log) 
    lines!(ax1, xg, b, color = :blue)
    lines!(ax2, xg, c, color = :black)
    lines!(ax3, xg, d, color = :red)
# display(fig)
    save(string("../outputs/plot_smoothed_hist_correlated", threshold, ".png"),fig)
end
# heatmap(yg, xg, log.(I))
# plot_heatmap(0.3, 0.1)


