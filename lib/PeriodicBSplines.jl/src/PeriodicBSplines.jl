"""
    PeriodicBSplines

Periodic cubic B-splines on a uniform grid of the circle [0, 2π): the basis functions, their values
and derivatives at the points of a finer uniform grid (collocation matrices), their exact Gram
matrices, interpolation, and the Fourier symbols that measure how well they resolve a harmonic.

The spline `m = 1, …, Nc` is centred at `2π (m - 1)/Nc`, with spacing `Δ = 2π/Nc`:

    φ_m(s) = B((s - 2π (m - 1)/Nc) / Δ),   periodised,

where `B` is the cardinal cubic B-spline, supported on [-2, 2]. See the documentation for the
theory and figures.
"""
module PeriodicBSplines

using LinearAlgebra
using SparseArrays

export bspline, dbspline, d2bspline, collocation, gram, derivative_gram, interpolate,
       evaluate, mass_symbol, derivative_symbol, modified_wavenumber

"""
    bspline(u) -> B(u)

The cardinal cubic B-spline: piecewise cubic, twice continuously differentiable, supported on
[-2, 2], with B(0) = 2/3, B(±1) = 1/6 and ∫B = 1.
"""
function bspline(u::Float64)
    v = abs(u)
    v < 1 && return (4 - 6v^2 + 3v^3) / 6
    v < 2 && return (2 - v)^3 / 6
    return 0.0
end

"""
    dbspline(u) -> B′(u)

The derivative of the cardinal cubic B-spline.
"""
function dbspline(u::Float64)
    v = abs(u)
    v < 1 && return -2u + 1.5u * v
    v < 2 && return -sign(u) * (2 - v)^2 / 2
    return 0.0
end

"""
    d2bspline(u) -> B″(u)

The second derivative of the cardinal cubic B-spline, piecewise linear.
"""
function d2bspline(u::Float64)
    v = abs(u)
    v < 1 && return -2 + 3v
    v < 2 && return 2 - v
    return 0.0
end

# the offset of the point s from the centre of spline m, in units of Δ, in [-Nc/2, Nc/2)
_offset(s::Float64, m::Int, Nc::Int) = mod(s * Nc / 2π - (m - 1) + Nc / 2, Nc) - Nc / 2

"""
    collocation(Ns, Nc) -> (Φ, Φ′)

Values and derivatives (with respect to s) of the `Nc` periodic splines at the `Ns` points
`s_l = 2π (l - 1)/Ns`: sparse `Ns × Nc` matrices with `Φ[l, m] = φ_m(s_l)` and
`Φ′[l, m] = φ_m′(s_l)`. Each row has at most four nonzeros. `Nc ≥ 4` is needed for the four splines
that overlap a point to be distinct.
"""
function collocation(Ns::Int, Nc::Int)
    Nc >= 4 || throw(ArgumentError("need at least 4 splines"))
    Δ = 2π / Nc
    I_, J_, V_, D_ = Int[], Int[], Float64[], Float64[]
    for l in 1:Ns
        s  = 2π * (l - 1) / Ns
        m₀ = floor(Int, s / Δ)
        for m in m₀ - 1:m₀ + 2
            u = (s - Δ * m) / Δ
            push!(I_, l); push!(J_, mod(m, Nc) + 1)
            push!(V_, bspline(u)); push!(D_, dbspline(u) / Δ)
        end
    end
    return sparse(I_, J_, V_, Ns, Nc), sparse(I_, J_, D_, Ns, Nc)
end

# Gauss–Legendre, 4 points on [0, 1]: exact for polynomials of degree 7
const _GX = (1 .+ [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]) ./ 2
const _GW = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538] ./ 2

# ∫ f(u) g(u - j) du over the real line, exactly for piecewise polynomials with integer knots
function _correlate(f, g, j::Int)
    acc = 0.0
    for a in -2:1, (x, w) in zip(_GX, _GW)
        u = a + x
        acc += w * f(u) * g(u - j)
    end
    return acc
end

function _circulant(c::Vector{Float64}, Nc::Int)
    h = length(c) ÷ 2
    I_, J_, V_ = Int[], Int[], Float64[]
    for m in 1:Nc, j in -h:h
        push!(I_, m); push!(J_, mod(m - 1 + j, Nc) + 1); push!(V_, c[j + h + 1])
    end
    return sparse(I_, J_, V_, Nc, Nc)
end

"""
    gram(Nc) -> M

The exact Gram (mass) matrix of the `Nc` periodic splines, `M[m, m′] = ∫₀^{2π} φ_m φ_m′ ds`:
circulant and banded, with the stencil `Δ (1, 120, 1191, 2416, 1191, 120, 1)/5040`.
"""
function gram(Nc::Int)
    Nc >= 7 || throw(ArgumentError("need at least 7 splines for the stencils not to overlap"))
    Δ = 2π / Nc
    return _circulant([Δ * _correlate(bspline, bspline, j) for j in -3:3], Nc)
end

"""
    derivative_gram(Nc) -> D

The exact matrix `D[m, m′] = ∫₀^{2π} φ_m φ_m′′ ds` (the prime on the second index): circulant,
banded and antisymmetric. `M⁻¹ D` is the Galerkin derivative on the splines.
"""
function derivative_gram(Nc::Int)
    Nc >= 7 || throw(ArgumentError("need at least 7 splines for the stencils not to overlap"))
    return _circulant([_correlate(bspline, dbspline, j) for j in -3:3], Nc)
end

"""
    interpolate(f) -> c

The coefficients `c` of the periodic spline `Σ c_m φ_m` that takes the values `f[m]` at the
centres `2π (m - 1)/Nc`, `Nc = length(f)`. The interpolation conditions are the circulant system
`(c_{m-1} + 4 c_m + c_{m+1})/6 = f_m`, solved by the FFT of its stencil, which never vanishes.
"""
function interpolate(f::Vector{Float64})
    Nc = length(f)
    θ  = 2π .* (0:Nc - 1) ./ Nc
    σ  = (4 .+ 2 .* cos.(θ)) ./ 6                        # symbol of the stencil (1, 4, 1)/6
    return _real_ifft(_dft(f) ./ σ)
end

# a plain DFT and its inverse: Nc is small, and no FFT package is needed
_dft(f::Vector{Float64}) = [sum(f[m] * cis(-2π * (k - 1) * (m - 1) / length(f)) for m in eachindex(f)) for k in eachindex(f)]
_real_ifft(F::Vector{ComplexF64}) = [real(sum(F[k] * cis(2π * (k - 1) * (m - 1) / length(F)) for k in eachindex(F))) / length(F) for m in eachindex(F)]

"""
    evaluate(c, s) -> Σ c_m φ_m(s)

The value at `s` of the periodic spline with coefficients `c`.
"""
function evaluate(c::Vector{Float64}, s::Float64)
    Nc = length(c)
    acc = 0.0
    for m in 1:Nc
        u = _offset(s, m, Nc)
        abs(u) < 2 && (acc += c[m] * bspline(u))
    end
    return acc
end

"""
    mass_symbol(θ) -> M̂(θ)/Δ

The symbol of the Gram matrix on the coefficients `c_m = e^{iθm}`, divided by `Δ`:
`(2416 + 2·1191 cos θ + 2·120 cos 2θ + 2 cos 3θ)/5040`, real and positive.
"""
mass_symbol(θ::Float64) = (2416 + 2382cos(θ) + 240cos(2θ) + 2cos(3θ)) / 5040

"""
    derivative_symbol(θ) -> D̂(θ)

The symbol of `derivative_gram` on `c_m = e^{iθm}`, purely imaginary.
"""
derivative_symbol(θ::Float64) = sum(_correlate(bspline, dbspline, j) * cis(j * θ) for j in -3:3)

"""
    modified_wavenumber(θ) -> θ*

The Galerkin derivative on the splines applied to the harmonic `c_m = e^{iθm}` gives
`i θ*/Δ` times it: `θ* = Im D̂(θ)/M̂(θ)`, against the exact `θ`. Its departure from `θ` measures
how well the splines differentiate a harmonic with `2π/θ` points per period.
"""
modified_wavenumber(θ::Float64) = imag(derivative_symbol(θ)) / mass_symbol(θ)

end
