


# quick plot:
fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"x,t", ylabel = "Area meas", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

@unpack xg, nb_arr, mx_arr = data_fermi
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, xg, m, color = :blue, label = "Fermi Lattice")
p, model, xdata = get_fit(xg,m) 
lines!(ax1, xdata, model(xdata,p), color = :blue, linestyle = :dash, label = "Fermi Lattice")

@unpack xg, nb_arr, mx_arr = data_cos[1]
m = vec(mean(nb_arr;dims = 2))
lines!(ax1, xg, m, color = :black, label = "Integrable Cos Pot")
p, model, xdata = get_fit(xg,m) 
lines!(ax1, xdata, model(xdata,p), linestyle = :dash, color = :black, label = " Fit cos n = 1")

# @unpack xg, nb_arr = data_cos[2]
# m = vec(mean(nb_arr;dims = 2))
# lines!(ax1, xg, m, color = :green, label = L"Cos Pot deg = 6")
# p, model, xdata = get_fit(xg,m) 
# lines!(ax1, xdata, model(xdata,p), color = :green, label = L" Fit cos n = 6")

s = "quick_comparison_decay_area.png"
axislegend(ax1);
save(plotsdir(s),fig)



# quick plot:
fig = Figure(size=(800, 600))
ax1= Axis(fig[1, 1], xlabel = L"x,t", ylabel = L"I_{max}", yticklabelsize = 30, xticklabelsize = 40, ylabelsize = 30, xlabelsize = 40,  titlesize = 30, yscale = Makie.pseudolog10)

@unpack xg, nb_arr, mx_arr = data_fermi
m = vec(mean(nb_arr;dims = 2))
mx = vec(mean(mx_arr;dims = 2))
lines!(ax1, xg, mx, color = :blue, label = "Fermi Lattice")
p, model, xdata = get_fit(xg,mx) 
lines!(ax1, xdata, model(xdata,p), color = :blue, linestyle = :dash, label = "Fermi Lattice fit")

@unpack xg, nb_arr, mx_arr = data_cos[1]
m = vec(mean(nb_arr;dims = 2))
mx = vec(mean(mx_arr;dims = 2))
lines!(ax1, xg, mx, color = :black, label = "Integrable Cos Pot")
p, model, xdata = get_fit(xg,mx) 
lines!(ax1, xdata, model(xdata,p), color = :black, linestyle = :dash, label = " Fit cos n = 1")

# @unpack xg, nb_arr = data_cos[2]
# m = vec(mean(nb_arr;dims = 2))
# lines!(ax1, xg, m, color = :green, label = L"Cos Pot deg = 6")
# p, model, xdata = get_fit(xg,m) 
# lines!(ax1, xdata, model(xdata,p), color = :green, label = L" Fit cos n = 6")

s = "quick_comparison_decay_area.png"
axislegend(ax1);
save(plotsdir(s),fig)

