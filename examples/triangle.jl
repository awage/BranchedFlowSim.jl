using BranchedFlowSim

L = 20
N = 500
xgrid = LinRange(-L/2, L/2, N)
potential = triangle_potential(xgrid, 2, 100)
