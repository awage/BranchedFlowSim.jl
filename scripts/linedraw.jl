using CairoMakie
using ColorTypes
using ImageIO
using FileIO
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
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    end
    if abs(x0 - x1) < 1e-9
        # TODO: Special code for vertical line
        error("Vertical line not yet supported")
        return nothing
    end
    ix :: Int32 = round(Int32, x0)
    iy :: Int32 = round(Int32, y0)
    # Form line equation y = mx + b
    m :: Float64 = (y1 - y0) / (x1 - x0)
    b :: Float64 = y0 - m * x0
    # End pixel coordinate
    ixe :: Int32 = round(Int32, x1)
    iye :: Int32 = round(Int32, y1)

    while true
        @inline f(ix, iy)
        if ix == ixe && iy == iye
            break
        end
        # TODO: get rid of this multiplication, should be easy
        ny = b + m * (ix + 0.5)
        # We can go either up, down, or right
        if ny > iy + 0.5
            iy += 1
        elseif ny < iy - 0.5
            iy -= 1
        else
            ix += 1
        end
    end
end

function draw_line!(img, x0, y0, x1, y1, color)
    iter_pixels(x0, y0, x1, y1) do x,y
        img[y,x] = color
    end
end

function test_line_image()
    W = 100
    img = zeros(RGB, W, W)
    draw_line!(img, 10.2, 10, 80.5, 20, RGB(1,0,0))
    draw_line!(img, 10, 10, 20.4, 80, RGB(1,1,0))
    # horizontal line
    draw_line!(img, 10, 50, 90, 50, RGB(0,1,0))
    # vertical line
    draw_line!(img, 10, 50, 90, 50, RGB(0,1,0))
    return img
end

img = test_line_image()

const golden_path::String = "outputs/linedraw_golden.png"
golden = load(golden_path)
println("golden: $(if img == golden "MATCH" else "DIFFER" end)")

display(img)
