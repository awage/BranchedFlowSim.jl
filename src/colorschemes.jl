using ColorTypes
using ColorSchemes
using Interpolations

daza_data = [
    1.0000 0.2937 0
    0.9765 0.2868 0
    0.9531 0.2799 0
    0.9296 0.2730 0
    0.9062 0.2661 0
    0.8827 0.2592 0
    0.8592 0.2524 0
    0.8358 0.2455 0
    0.8123 0.2386 0
    0.7889 0.2317 0
    0.7654 0.2248 0
    0.7420 0.2179 0
    0.7185 0.2110 0
    0.6950 0.2041 0
    0.6716 0.1972 0
    0.6481 0.1903 0
    0.6247 0.1834 0
    0.6012 0.1766 0
    0.5777 0.1697 0
    0.5543 0.1628 0
    0.5308 0.1559 0
    0.5074 0.1490 0
    0.4839 0.1421 0
    0.4476 0.1314 0
    0.4113 0.1208 0
    0.3750 0.1101 0
    0.3387 0.0995 0
    0.3024 0.0888 0
    0.1512 0.0444 0
    0 0 0
    0 0 0
    0 0 0
    0 0 0
    0 0.0355 0.0677
    0 0.0710 0.1354
    0 0.1136 0.2166
    0 0.1561 0.2979
    0 0.1751 0.3340
    0 0.1940 0.3701
    0 0.2129 0.4062
    0 0.2259 0.4309
    0 0.2388 0.4557
    0 0.2518 0.4804
    0 0.2648 0.5052
    0 0.2777 0.5299
    0 0.2907 0.5547
    0 0.3036 0.5794
    0 0.3166 0.6041
    0 0.3296 0.6289
    0 0.3425 0.6536
    0 0.3555 0.6784
    0 0.3684 0.7031
    0 0.3814 0.7278
    0 0.3944 0.7526
    0 0.4073 0.7773
    0 0.4203 0.8021
    0 0.4333 0.8268
    0 0.4462 0.8515
    0 0.4592 0.8763
    0 0.4721 0.9010
    0 0.4851 0.9258
    0 0.4981 0.9505
    0 0.5110 0.9753
    0 0.5240 1.0000
]
function make_colorscheme_daza()
    colors = [RGB{Float64}(c[1], c[2], c[3]) for c ∈ eachrow(daza_data)]
    return ColorScheme(colors)
end

"""
Colorscheme similar to one used by Alvar Daza in
https://www.pnas.org/doi/10.1073/pnas.2110285118.
"""
const colorscheme_daza :: ColorScheme = make_colorscheme_daza();

function make_sigmoid_berlin(c=10)
    xs = LinRange(-1, 1, 1024)
    sigmoid = 1 ./ (1 .+ exp.(- c * xs))
    colors = get(ColorSchemes.berlin, sigmoid)
    return ColorScheme(colors)
end

const sigmoid_berlin = make_sigmoid_berlin()

function hsv_color(z)
    r = abs(z)
    arg = angle(z)
    return RGB{Float64}(HSV{Float64}(rad2deg(arg), 1, r))
end

abstract type ComplexColorScheme end

struct ComplexHSV <: ComplexColorScheme
end

struct RealPartColor <: ComplexColorScheme
    cs :: ColorScheme
end

struct AbsColor <: ComplexColorScheme
    cs :: ColorScheme
end

# 
const complex_hsv = ComplexHSV()
const complex_berlin = RealPartColor(ColorSchemes.berlin)

# Provide identical syntax
function Base.get(c :: ComplexHSV, color_or_arr)
    return hsv_color.(color_or_arr)
end

function Base.get(c :: RealPartColor, color_or_arr)
    # Convert from -1:1 to 0:1
    r = (real(color_or_arr) .+ 1) / 2
    return get(c.cs, r)
end

function Base.get(c :: AbsColor, color_or_arr)
    return get(c.cs, abs.(color_or_arr))
end