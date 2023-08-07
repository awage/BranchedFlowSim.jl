"""
Set of small helpers, not directly belonging to any other file.
"""

using DataStructures

export sample_midpoints
export line_integer_cells
export correlation_dimension

"""
    sample_midpoints(a,b, n)

Divide the interval [a,b] in n equally sized segments and return a Vector of
midpoints of each segment.
"""
function sample_midpoints(a::Real, b::Real, n::Integer)::LinRange{Float64, Int64}
    @assert n > 0
    dx::Float64 = (b-a) / n
    first = a + dx/2
    last = b - dx/2
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
        line_integer_cells(-x0, y0, -x1, y1) do x,y
            @inline f(-x, y)
        end
        return nothing
        # x0, x1 = x1, x0
        # y0, y1 = y1, y0
    end
    ix :: Int = round(Int, x0)
    iy :: Int = round(Int, y0)
    # End pixel coordinate
    ixe :: Int = round(Int, x1)
    iye :: Int = round(Int, y1)

    if abs(x0 - x1) < 1e-9
        for y ∈ iy:iye
            @inline f(ix, y)
        end
        return nothing
    end

    # Form line equation y = mx + b
    m :: Float64 = (y1 - y0) / (x1 - x0)
    # b :: Float64 = y0 - m * x0
    # ny ::Float64  = b + m * (ix + 0.5) - iy
    ny ::Float64  = y0 + m * (ix -x0 + 0.5) - iy
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

function loop_close_pairs(f,
    points::AbstractMatrix{Float64},
    r0 :: Float64)
    @assert size(points)[1] == 2 "Only 2D points are supported"
    N = size(points)[2]
    # Sort by x coordinate
    idx_x = sortperm(@view points[1,:])

    Vec2 = SVector{2, Float64}
    active_points = SortedMultiDict{Float64, Vec2}()
    pastend = pastendsemitoken(active_points)
    # Map of inspected points in insertion (x) order.
    old_tokens = SMDSemiToken[pastend for _ ∈ 1:N]
    old = 1
    r2 = r0^2
    for j ∈ 1:N
        p1 = Vec2(@view points[:, idx_x[j]])
        # Look for points for potential pairs
        it = searchsortedfirst(active_points,
            p1[2] - r0)

        @assert pastend == pastendsemitoken(active_points)
        while it != pastend
            p0 = active_points[it]
            @assert p0[1] - p1[1] <= r0
            @assert p1[2] - p0[2] <= r0
            if p0[2] - p1[2] >= r0
                break
            end
            dist2 = (p1[1]-p0[1])^2 + (p1[2]-p0[2])^2
            if dist2 < r2
                f(p0, p1)
            end
            it = advance((active_points, it))
        end
        # Add the point 
        t = insert!(active_points, p1[2], p1)
        old_tokens[j] = t
        while true
            # Potentially remove a past point
            @assert old_tokens[old] != pastend
            p_old = active_points[old_tokens[old]]
            if p1[1] - p_old[1] >= r0
                # println("del")
                delete!((active_points, old_tokens[old]))
                old += 1
            else
                break
            end
        end
        # TODO: loop to remove more than one point if necessary
    end
end

function close_distances(points, r0)
    dists = Float64[]
    loop_close_pairs(points, r0) do a,b
        push!(dists, norm(a-b))
    end
    return dists
end

function close_distances2(points, r0)
    N = size(points)[2]
    dists = Float64[]
    r2 = r0^2
    for i ∈ 1:N
        for j ∈ (i+1):N
            a = @view points[:, i]
            b = @view points[:, j]
            d2 = (a[1]-b[1])^2 + (a[2]-b[2])^2
            if d2 < r2
                push!(dists, sqrt(d2))
            end
        end
    end
    return dists
end

function correlation_dimension(points :: AbstractMatrix{Float64},
     r0::Float64)
    @assert size(points)[1] == 2 "Only 2D points are supported"

    tot = 0.0
    num = 0.0
    loop_close_pairs(points, r0) do a,b
        d = norm(a-b)
        tot += log(d / r0)
        num += 1.0
    end
    return -1.0 / (tot/num)
end
