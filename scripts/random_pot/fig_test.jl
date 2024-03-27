using CairoMakie

fig = Figure()

fig[1:2, 1:2] = [Axis(fig) for _ in 1:4]

supertitle = Label(fig[0, :], "Six plots", fontsize = 30)

sideinfo = Label(fig[1, 1, TopLeft()], "(a)", padding = (0,5,5,0))
sideinfo = Label(fig[1, 2, TopLeft()], "(b)")
sideinfo = Label(fig[2, 1, TopLeft()], "(c)")
sideinfo = Label(fig[2, 2, TopLeft()], "(d)")

fig
