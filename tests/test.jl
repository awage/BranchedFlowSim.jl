using BranchedFlowSim
using Test
using CairoMakie
using LinearAlgebra

path_prefix = "tests/figures/"

mkpath(path_prefix)

"""
    compare_force_with_diff(p)

Debugging function for comparing `force` implementation with `force_diff`
"""
function compare_force_with_diff(p)
    xs = LinRange(-0.2, 1.2, 99)
    ys = LinRange(-0.2, 1.2, 124)
    return [
        norm(BranchedFlowSim.force_diff(p, x, y) - force(p, x, y)) for y ∈ ys, x ∈ xs
    ]
end

function save_heatmap(path, fd)
    fig, ax, plt = heatmap(fd)
    Colorbar(fig[1, 2], plt)
    save(path, fig)
end

@testset "fermi_dot_force" begin
    v0 = 0.33
    radius = 0.25
    potential = FermiDotPotential(radius, 0.2, v0)
    # At the edge the force should be pointing outwards
    @test force(potential, 0, radius)[2] > 0
    @test force(potential, 0, -radius)[2] < 0
    @test force(potential, radius, 0)[1] > 0
    @test force(potential,-radius, 0)[1] < 0

    ss = compare_force_with_diff(potential)
    save_heatmap(path_prefix * "fermi_dot_force_diff_compare.png", ss)
    xs = LinRange(-1, 1, 200)
    ys = LinRange(-1, 1, 200)
    save_heatmap(path_prefix * "fermi_dot.png", grid_eval(xs, ys, potential))
    @test maximum(ss) ≈ 0 atol=1e-5
end

@testset "lattice_potential_force" begin
    a = 0.2
    radius = 0.2 * 0.25
    potential = LatticePotential(rotation_matrix(0) * a, radius, 0.04)
    ss = compare_force_with_diff(potential)
    # At the edge the force should be pointing outwards
    @test force(potential, 0, radius)[2] > 0
    @test force(potential, 0, -radius)[2] < 0
    @test force(potential, radius, 0)[1] > 0
    @test force(potential,-radius, 0)[1] < 0
    save_heatmap(path_prefix * "lattice_force_diff_compare.png", ss)
    @test maximum(ss) ≈ 0 atol=1e-5
end

@testset "repeated_potential_vs_lattice" begin
    # Produce exact same potential using LatticePotential and RepeatedPotential
    a = 0.2
    radius = 0.25 * a
    dot = FermiDotPotential(radius, 0.1)
    A = rotation_matrix(0) * a
    lattice = LatticePotential(A, radius, 0.1)
    lattice_points = lattice_points_in_a_box(A, 3)
    repeated = RepeatedPotential(lattice_points, dot, a)
    
    xs = LinRange(-0.1, 1.1, 200)
    ys = LinRange(-0.1, 1.1, 200)
    
    lg = grid_eval(xs, ys, lattice)
    rg = grid_eval(xs, ys, repeated)
    save_heatmap(path_prefix * "repeated_lattice_compare.png", (lg .- rg))
    save_heatmap(path_prefix * "repeated_lattice.png", rg)
    save_heatmap(path_prefix * "lattice.png", lg)

    @test maximum(abs.(lg-rg)) ≈ 0 atol=1e-5
end

@testset "lattice_potential_branches" begin
    num_rays = 512
    sim_width = 3
    ts = LinRange(0, sim_width, 50)

    potential = LatticePotential(rotation_matrix(0) * 0.2, 0.2 * 0.25, 0.04)
    @time nb = quasi2d_num_branches(num_rays,0.01, ts, potential)[end]
    @time potential(1.2, 0.2)
    # Known answer, should not change (unless maybe if num_rays is increased)
    @test nb == 57
    println("nb=$nb")
end

@testset "repeated_potential_branches" begin
    num_rays = 512
    sim_width = 3
    ts = LinRange(0, sim_width, 50)
    A = rotation_matrix(0)*0.2
    lattice_points = lattice_points_in_a_box(A, -1, sim_width + 1, -1, 2)
    dot = FermiDotPotential(0.2 * 0.25, 0.04)
    potential = RepeatedPotential(lattice_points, dot, 0.2)

    @time nb = quasi2d_num_branches(num_rays,0.01, ts, potential)[end]
    @time potential(1.2, 0.2)
    # Known answer, should not change (unless maybe if num_rays is increased)
    @test nb == 57
    println("nb=$nb")
end

@testset "function_potential_branches" begin
    # Test using a plain function as the potential. That should get converted
    # nicely.
    num_rays = 512
    sim_width = 4
    ts = LinRange(0, sim_width, 50)
    k = 0.2 * 2pi
    function potential(x,y)
        0.08*(cos(k*x)+cos(k*y))
    end
    @time nb = quasi2d_num_branches(num_rays,0.01, ts, FunctionPotential(potential))[end]
    # Known answer, should not change (unless maybe if num_rays is increased)
    @test nb == 0.0
    println("nb=$nb")
end


@testset "poincare_mapper" begin
    # Just test that the code doesn't crash. This is not deeply checking for correctness.
    a = 0.2
    radius = 0.2 * 0.25
    potential = LatticePotential(rotation_matrix(0) * a, radius, 0.04)
    
end
