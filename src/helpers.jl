"""
Set of small helpers, not directly belonging to any other file.
"""

export sample_midpoints
export line_integer_cells

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