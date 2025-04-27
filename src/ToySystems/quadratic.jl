# Simple 1-dimensional dynamical system with a single fixed point.


# ~~~ state definition ~~~
mutable struct QuadraticState <: AbstractVector{Float64}
    x::Float64
end


# ~~~ array interface ~~~
Base.IndexStyle(::Type{<:QuadraticState})           = Base.IndexLinear()
Base.eltype(::QuadraticState)                       = Float64
Base.size(::QuadraticState)                         = (1,)
Base.similar(::QuadraticState, ::Type{T}) where {T} = QuadraticState(0.0)
Base.getindex(x::QuadraticState, i::Int)            = 0 < i < 2 ? x.x : throw(BoundsError(x, i))
Base.setindex!(x::QuadraticState, v, i::Int)        = 0 < i < 2 ? x.x = v : throw(BoundsError(x, i))


# ~~~ linear algebra ~~~
dot(x::QuadraticState, y::QuadraticState) = x.x*y.x
norm(x::QuadraticState)                   = sqrt(dot(x, x))


# ~~~ operators ~~~
dds!(dxds::Q, ::Q) where {Q<:QuadraticState}     = (dxds.x = 0; dxds)
rhs!(F::Q, x::Q) where {Q<:QuadraticState}       = (F.x = x.x^2; F)
adj!(G::Q, x::Q, r::Q) where {Q<:QuadraticState} = (G.x = -2*x.x*r.x; G)
