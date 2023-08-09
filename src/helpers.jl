"""
Set of small helpers, not directly belonging to any other file.
"""

export sample_midpoints
export line_integer_cells
export correlation_dimension

"""
    sample_midpoints(a,b, n)

Divide the interval [a,b] in n equally sized segments and return a Vector of
midpoints of each segment.
"""
function sample_midpoints(a::Real, b::Real, n::Integer)::LinRange{Float64,Int64}
    @assert n > 0
    dx::Float64 = (b - a) / n
    first = a + dx / 2
    last = b - dx / 2
    return LinRange(first, last, n)
end

"""
    iterate_line_integer_cells(f, x0, y0, x1, y1)

A line drawing algorithm which visits all integer cells touched by line from
(possibly fractional) coordinates (x0,y0) to (x1,y1). Integer cells are
centered at integer coordinates (x,y) and extend to (x-0.5,y-0.5) and (x+0.5,
y+0.5).

f is a function that is called with two Int parameters, x and y

+----+----+----+----+
|####|    |    |    |
|####|    |    |    |
|0x##|    |    |    |
+--xx+----+----+----+
|####xx###|####|    |
|####|#xx#|####|    |
|####|###xx####|    |
+----+----+xx--+----+
|    |    |##xx|####|
|    |    |####xx###|
|    |    |####|#1##|
+----+----+----+----+
"""
function line_integer_cells(f, x0, y0, x1, y1)
    # Always draw in positive x direction
    if x0 > x1
        line_integer_cells(-x0, y0, -x1, y1) do x, y
            @inline f(-x, y)
        end
        return nothing
        # x0, x1 = x1, x0
        # y0, y1 = y1, y0
    end
    ix::Int = round(Int, x0)
    iy::Int = round(Int, y0)
    # End pixel coordinate
    ixe::Int = round(Int, x1)
    iye::Int = round(Int, y1)

    if abs(x0 - x1) < 1e-9
        for y ∈ iy:iye
            @inline f(ix, y)
        end
        return nothing
    end

    # Form line equation y = mx + b
    m::Float64 = (y1 - y0) / (x1 - x0)
    # b :: Float64 = y0 - m * x0
    # ny ::Float64  = b + m * (ix + 0.5) - iy
    ny::Float64 = y0 + m * (ix - x0 + 0.5) - iy
    if y0 < y1
        # Moving upwards
        while true
            @inline f(ix, iy)
            if ix == ixe && iy == iye
                break
            end
            # We move either up or to the right
            if ny > 0.5
                iy += 1
                ny -= 1
            else
                ix += 1
                ny += m
            end
        end
    else
        # Moving downwards
        while true
            @inline f(ix, iy)
            if ix == ixe && iy == iye
                break
            end
            # We move either down or to the right
            if ny < -0.5
                iy -= 1
                ny += 1
            else
                ix += 1
                ny += m
            end
        end
    end
    return nothing
end

function loop_close_distances_squared(f,
    points::AbstractMatrix{Float64},
    r0::Float64)
    @assert size(points)[1] == 2 "Only 2D points are supported"
    N = size(points)[2]
    # Sort by x coordinate
    idx_x = sortperm(@view points[1, :])
    miny, maxy = extrema(@view points[2, :])

    Vec2 = SVector{2,Float64}

    # Create an array of linked lists sorted in descending order of the
    # x coordinate.
    H = ceil(Int, (maxy - miny) / r0)
    row_prev::Vector{Int64} = fill(0, N)
    row_last::Vector{Int64} = fill(0, H)

    r2 = r0^2
    for i ∈ idx_x
        p = Vec2(
            points[1, i],
            points[2, i]
        )
        row = 1 + floor(Int, (p[2] - miny) / r0)
        for j ∈ (row-1):(row+1)
            if 1 <= j <= H
                k = row_last[j]
                while k != 0
                    p2 = Vec2(
                        points[1, k],
                        points[2, k]
                    )
                    if p2[1] <= p[1] - r0
                        break
                    else
                        d2 = (p[1] - p2[1])^2 + (p[2] - p2[2])^2
                        if d2 < r2
                            # println("i=$i j=$j k=$k")
                            @inline f(d2)
                        end
                    end
                    k = row_prev[k]
                end
            end
        end
        # Maintain linked list
        row_prev[i] = row_last[row]
        row_last[row] = i
    end
    return nothing
end

function close_distances(points, r0)
    dists = Float64[]
    loop_close_distances_squared(points, r0) do d2
        push!(dists, sqrt(d2))
    end
    return dists
end

function close_distances_brute(points, r0)
    N = size(points)[2]
    dists = Float64[]
    r2 = r0^2
    for i ∈ 1:N
        for j ∈ (i+1):N
            a = @view points[:, i]
            b = @view points[:, j]
            d2 = (a[1] - b[1])^2 + (a[2] - b[2])^2
            if d2 < r2
                push!(dists, sqrt(d2))
            end
        end
    end
    return dists
end

"""
    pick_r0(points::AbstractMatrix{Float64})

Given a set of 2D points (one per column), returns a reasonable
value for r0 used for dimension estimation.

For a set of N points distributed uniformly randomly in a rectangular area,
this should result in approximately N close pairs.
"""
function pick_r0(points::AbstractMatrix{Float64})
    @assert size(points)[1] == 2 "Only 2D points are supported"
    minx, maxx = extrema(@view points[1, :])
    miny, maxy = extrema(@view points[2, :])
    N = size(points)[2]
    A = (maxx - minx) * (maxy - miny)
    return sqrt(2 * A / (pi * N))
end

"""
    correlation_dimension(points :: AbstractMatrix{Float64},
     r0::Float64)

TBW
"""
function correlation_dimension(points::AbstractMatrix{Float64},
    r0::Float64 = pick_r0(points))
    @assert size(points)[1] == 2 "Only 2D points are supported"

    tot = Ref(0.0)
    num = Ref(0.0)
    log_r0 :: Float64 = log(r0)
    loop_close_distances_squared(points, r0) do d2
        tot[] += log(d2)/2 - log_r0
        num[] += 1.0
    end
    return -1.0 / (tot[] / num[])
end
