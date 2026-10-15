# Orbit: the unknowns of the search, modal coefficients plus the scalar parameters, as one element
# of a real vector space. Both methods work on it directly: L-BFGS and the Arnoldi iteration need
# broadcasting, dot and copies.

"""
    Orbit(a, p)

Unknowns of the search. These are the space-time field `a`, in any discretisation, and the
parameter vector `p = [log ω, c₁, c₂, …]`, the logarithm of the frequency `ω = 2π/T` followed by
one drift speed per drift direction of the [`System`](@ref). The frequency is searched through its
logarithm, which keeps it positive and makes a step a relative change of `ω`.
[`frequency`](@ref) returns `ω`. Orbits broadcast like vectors, and `dot` adds the inner product
of the fields to that of the parameters. The parameters are stored in the real type of the field,
`Float32` for a single-precision field.

```julia
x = Orbit(a, [log(2π/T), c])
```
"""
struct Orbit{X, T<:Real}
    a::X         # space-time field
    p::Vector{T} # log ω, then the drift speeds; a vector so that broadcasts write in place

    Orbit(a::X, p::AbstractVector{<:Real}) where {X} =
        new{X, real(eltype(a))}(a, collect(real(eltype(a)), p))
end

# ---- parameters: p[1] is log ω, p[2:end] the drift speeds ----
"""
    frequency(x::Orbit) -> ω

The frequency `ω = 2π/T` of the orbit, from its logarithm `x.p[1]`.
"""
frequency(x::Orbit) = exp(x.p[1])
ndrift(x::Orbit)    = length(x.p) - 1

# the i-th drift speed; i = 0 would silently return log ω, so the index is checked
function drift(x::Orbit, i::Integer)
    1 <= i <= ndrift(x) || throw(BoundsError(x.p[2:end], i))
    return x.p[1 + i]
end


# ---------------------------------------------------------------------------- #
# vector space                                                                 #
# ---------------------------------------------------------------------------- #
Base.similar(x::Orbit) = Orbit(similar(x.a), similar(x.p))
Base.copy(x::Orbit)    = Orbit(copy(x.a), copy(x.p))
Base.zero(x::Orbit)    = Orbit(zero(x.a), zero(x.p))

LinearAlgebra.dot(x::Orbit, y::Orbit) = dot(x.a, y.a) + dot(x.p, y.p)
LinearAlgebra.norm(x::Orbit)          = sqrt(dot(x, x))


# ---------------------------------------------------------------------------- #
# broadcasting: coefficients and parameters separately                         #
# ---------------------------------------------------------------------------- #
# Orbits are not arrays: the style bypasses axes and instantiation, and a broadcast is
# evaluated in place once on the coefficients and once on the parameters.
struct OrbitStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:Orbit})                          = OrbitStyle()
Base.BroadcastStyle(::OrbitStyle, ::Broadcast.BroadcastStyle) = OrbitStyle()
Base.broadcastable(x::Orbit)                                  = x

Broadcast.materialize(bc::Broadcast.Broadcasted{OrbitStyle})   = copyto!(similar(_find_orbit(bc)), bc)
Broadcast.materialize!(dest::Orbit, bc::Broadcast.Broadcasted) = copyto!(dest, bc)

function Base.copyto!(dest::Orbit, bc::Broadcast.Broadcasted)
    bcf = Broadcast.flatten(bc)

    # ---- the same expression on the coefficients and on the parameters ----
    broadcast!(bcf.f, dest.a, map(_a, bcf.args)...)
    broadcast!(bcf.f, dest.p, map(_p, bcf.args)...)

    return dest
end

_a(x::Orbit) = x.a
_a(x)        = x
_p(x::Orbit) = x.p
_p(x)        = x

# first Orbit among the arguments of a broadcast
_find_orbit(bc::Broadcast.Broadcasted) = _find_orbit(bc.args)
_find_orbit(args::Tuple)               = _find_orbit(_find_orbit(args[1]), Base.tail(args))
_find_orbit(x)                         = x
_find_orbit(::Tuple{})                 = nothing
_find_orbit(x::Orbit, rest)            = x
_find_orbit(::Any, rest)               = _find_orbit(rest)
