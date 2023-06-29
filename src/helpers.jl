"""
Set of small helpers, not directly belonging to any other file.
"""

export sample_midpoints

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