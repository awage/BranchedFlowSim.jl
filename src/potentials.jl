export fermi_dot, add_fermi_dot!
export absorbing_potential, lattice_points_in_a_box, lattice_potential, triangle_potential
export make_angled_grid_potential

"""
    fermi_step(x, α)

Smooth step function, i.e. goes from 0 to 1 as x goes from negative to
positive. α is a term controlling how smooth the transition from 0 to 1 is.
Equal to Fermi-Dirac distribution where α=kT, so large values of α correspond
to smooth transition and small values to an abrupt one.
"""
function fermi_step(x, α)
    return 1 / (1 + exp(-x / α))
end

"""
    lattice_points_in_a_box(A, xmin, xmax, ymin, ymax)

Returns all lattice points in a box [xmin,xmax]×[ymin,ymax]. A is a 2×2 matrix
with generating vectors of the lattice as the columns. Returns a matrix with 2
rows and one column per lattice point.
"""
function lattice_points_in_a_box(A, xmin, xmax, ymin, ymax)
    # Start with a vector of 2d vectors
    points = Vector{Vector{Float64}}()
    avec = A[:, 1]
    bvec = A[:, 2]
    # Solve the maximum integer multiple of each vector by mapping each corner
    # of the box to the lattice basis.
    Na = 0
    Nb = 0
    for corner ∈ [
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]]
        ca, cb = A \ corner
        Na = max(Na, ceil(Int, abs(ca)))
        Nb = max(Nb, ceil(Int, abs(cb)))
    end
    for ka ∈ -Na:Na
        for kb ∈ -Nb:Nb
            v = ka * avec + kb * bvec
            if v[1] >= xmin && v[1] <= xmax && v[2] >= ymin && v[2] <= ymax
                push!(points, v)
            end
        end
    end
    # Convert to a N-by-2 matrix
    return reduce(hcat, points)
end
"""
    lattice_points_in_a_box(A, L)

Returns all lattice points in L-by-L sized box (2D) centered at (0,0). A is a
2×2 matrix with generating vectors of the lattice as the columns. Returns a
matrix with 2 rows and one column per lattice point.
"""
function lattice_points_in_a_box(A, L)
    return lattice_points_in_a_box(A, -L / 2, L / 2, -L / 2, L / 2)
end

"""
    fermi_dot(xgrid::AbstractVector{Float64},
    pos, radius, softness=1)

Return a potential function matrix of size length(xgrid)×length(xgrid) with
value tending to 1 inside a circle of given position and radius.
"""
function fermi_dot(xgrid::AbstractVector{Float64},
    pos, radius, softness=1)
    # N x N matrix of distance from `pos`
    dist = sqrt.(((xgrid .- pos[2]) .^ 2) .+ transpose(((xgrid .- pos[1]) .^ 2)))
    α = softness * radius / 5
    return fermi_step.(radius .- dist, α)
end

function fermi_dot(xgrid::AbstractVector{Float64},
    ygrid::AbstractVector{Float64},
    pos, radius, softness=1)
    # N x N matrix of distance from `pos`
    dist = sqrt.(((ygrid .- pos[2]) .^ 2) .+ transpose(((xgrid .- pos[1]) .^ 2)))
    α = softness * radius / 5
    return fermi_step.(radius .- dist, α)
end

function add_fermi_dot!(arr, xgrid, ygrid, pos, radius, softness=1)
    xmin = pos[1] - 3 * radius
    xmax = pos[1] + 3 * radius
    ymin = pos[2] - 3 * radius
    ymax = pos[2] + 3 * radius
    # First solve indices 
    dx = xgrid[2] - xgrid[1]
    dy = ygrid[2] - ygrid[1]

    # 
    ix0 = 1 + round(Int, (xmin - xgrid[1]) / dx)
    ix1 = 1 + round(Int, (xmax - xgrid[1]) / dx)
    iy0 = 1 + round(Int, (ymin - ygrid[1]) / dy)
    iy1 = 1 + round(Int, (ymax - ygrid[1]) / dy)
    ix0 = clamp(ix0, 1:length(xgrid))
    ix1 = clamp(ix1, 1:length(xgrid))
    iy0 = clamp(iy0, 1:length(ygrid))
    iy1 = clamp(iy1, 1:length(ygrid))

    fd = fermi_dot(
        xgrid[ix0:ix1],
        ygrid[iy0:iy1],
        pos, radius, softness
    )
    # add to arr
    arr[iy0:iy1, ix0:ix1] .+= fd
end

"""
    lattice_potential(xgrid, A, dot_height, dot_radius, offset=[0, 0],
    dot_softness=1)

Returns a potential matrix.
"""
function lattice_potential(xgrid, A, dot_height, dot_radius;
    offset=[0, 0], dot_softness=1)
    N = length(xgrid)
    L = (xgrid[end] - xgrid[1]) + 10 * dot_radius + abs(maximum(offset))
    V = zeros(N, N)
    for point ∈ eachcol(lattice_points_in_a_box(A, L))
        V .+= fermi_dot(xgrid, point + offset, dot_radius, dot_softness)
    end
    return V * dot_height
end


"""
    triangle_potential(xgrid, a, dot_height, dot_radius=0.1 * a, dot_softness=1)

Convenience function for creating a lattice potential with a triangle lattice.
"""
function triangle_potential(xgrid, a, dot_height, dot_radius=0.2 * a, dot_softness=1)
    # Lattice matrix
    A = a * [
        1 cos(π / 3)
        0 sin(π / 3)
    ]
    # Add offset such that 0,0 is in the center of a triangle.
    # Computed by taking the average of three lattice points,
    # one of which is (0,0).
    offset = (A[:, 1] + A[:, 2]) / 3
    return lattice_potential(xgrid, A, dot_height, dot_radius,
        offset=offset,
        dot_softness=dot_softness)
end

"""
    absorbing_potential(xgrid, strength, width)

Returns a length(xgrid)×length(xgrid) sized matrix representing an absorbing
potential. `strength` is the maximum absolute value of the potential and
`width` is the width of the absorbing wall in same units as `xgrid`.
"""
function absorbing_potential(xgrid, strength, width)
    N = length(xgrid)
    L = xgrid[end] - xgrid[1]

    # Width in grid points
    width_n = round(Int, N * width / L)
    pot = zeros((N, N))
    pot[1:width_n, :] .+= LinRange(1, 0, width_n) .^ 2
    pot[:, 1:width_n] .+= (LinRange(1, 0, width_n) .^ 2)'
    pot[N-width_n+1:N, :] .+= LinRange(0, 1, width_n) .^ 2
    pot[:, N-width_n+1:N] .+= (LinRange(0, 1, width_n) .^ 2)'
    return -1im * strength * pot
end

"""
    make_angled_grid_potential(xs, ys, int_cot::Integer)

Returns a potential which is angled with cot(θ)=int_cot. This strange way to
specify the angle is required so that the returned potential is periodic along
the y axis.
"""
function make_angled_grid_potential(xs, ys, int_cot::Integer)
    dy = ys[2] - ys[1]
    H = length(ys) * dy
    pot = zeros(length(ys), length(xs))
    onedot = zeros(length(ys), length(ys))
    θ = atan(1 / int_cot)
    print("make_angled_grid_potential: θ=$(rad2deg(θ))°\n")
    lattice_constant = H / sqrt(1 + int_cot^2)
    lattice_matrix = lattice_constant * [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]
    radius = lattice_constant / 4
    points = lattice_points_in_a_box(
        lattice_matrix,
        xs[1]-3 * radius, xs[end] + 3 * radius,
        ys[1] -3 * radius, ys[end] + 3 * radius
    )
    for p ∈ eachcol(points)
        add_fermi_dot!(pot, xs, ys, p, radius)
    end
    return pot
end