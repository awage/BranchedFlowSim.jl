using CairoMakie
using ColorTypes
using ImageIO
using Printf
using FileIO
using Random
using FixedPointNumbers

function low_line(f, x0, y0, x1, y1)
    # Assumptions here:
    #  x0 < x1
    #  y0 < y1
    #  y1 - y0 < x1 - x0
    
    dx = x1 - x0
    dy = y1 - y0

end

function iter_pixels(f, x0, y0, x1, y1)
    # Always draw in positive x direction
    if x0 > x1
        iter_pixels(-x0, y0, -x1, y1) do x,y
            f(-x, y)
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
    b :: Float64 = y0 - m * x0
    ny ::Float64  = b + m * (ix + 0.5) - iy
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

function draw_line!(img, x0, y0, x1, y1, color)
    @inline iter_pixels(x0, y0, x1, y1) do x,y
        img[y,x] = color
    end
end

function test_line_image()
    W = 100
    PixelType = RGB{N0f8}
    img = zeros(PixelType, W, W)
    draw_line!(img, 10.2, 10, 80.5, 20, PixelType(1,0,0))
    draw_line!(img, 10, 10, 20.4, 80, PixelType(1,1,0))
    # horizontal line
    draw_line!(img, 10, 50, 90, 50, PixelType(0,1,0))
    # vertical line
    # draw_line!(img, 10, 50, 90, 50, PixelType(0,1,0))

    # backwards 
    draw_line!(img, 80, 50, 10, 90, PixelType(0,1,1))
    draw_line!(img, 90, 90, 21, 19, PixelType(1,0,1))
    return img
end

function draw_lines(img, lines, colors)
    for (i, (x0,y0, x1, y1)) ∈ enumerate(eachcol(lines))
        c = colors[i]
        draw_line!(img, x0, y0, x1, y1, c)
    end
end

function benchmark()
    W = 100
    num_lines = 100000
    PixelType = RGB{N0f8}
    rng = Xoshiro(0)
    lines = 1 .+ (W-1) * rand(rng, 4, num_lines)
    colors = [
        PixelType(r,g,b) for (r,g,b) ∈ eachcol(rand(rng, 3, num_lines))
    ]

    # Call twice to not count compilation
    img = zeros(PixelType, W, W)
    draw_lines(img, lines, colors)
    img = zeros(PixelType, W, W)
    times = @timed draw_lines(img, lines, colors)
    if times.bytes > 0
        println("ALLOCATED $(times.bytes) bytes")
    end
    t = times.time
    total_length = sum(
        hypot(x1-x0, y1-y0) for (x0,y0,x1,y1) ∈ eachcol(lines)
    )
    @printf "time: %.8fs, %.0f pixels/s\n" t (total_length/t)
    # display(img)
end

img = test_line_image()

benchmark()

const golden_path::String = "outputs/linedraw_golden.png"
golden = load(golden_path)
println("golden: $(if img == golden "MATCH" else "DIFFER" end)")
save("outputs/linedraw_last.png", img)

display(img)
