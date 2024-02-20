using DrWatson
@quickactivate
using BranchedFlowSim
using CairoMakie
using LaTeXStrings
using CodecZlib
using StatsBase
using LsqFit
using ProgressMeter

include("utils_decay_comp.jl")

# Comon parameters
num_rays = 1500000; 
dt = 0.01; T = 100; xres = 20
yres = 1024; threshold = 2.; 
num_angles = 50

v0_range = range(0.04, 0.4, step = 0.04)

include("plot_v0_vs_expfit_Imax.jl")
include("plot_v0_vs_expfit_area.jl")
include("plot_v0_vs_expfit_pks.jl")

