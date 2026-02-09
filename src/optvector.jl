# Wrapper object for generic type and period to interface with optimisation
# packages seemlessly

# This type is mutable to allow the period to be altered over the duration of
# the optimisation. The only objects that will actually be of this type are
# the actual optimisation variable, the gradient, and any internal copies
# made by the optimisation package. Every other case will treat them as
# separate objects. This is also a good way to add in dependence on continuous
# symmetries similar to MVector in NKSearch.jl.

# ~~~ Wrapper vector struct for optimisation ~~~
mutable struct OptVector{X, N, TX} <: AbstractVector{Float64}
    const x::X
          T::TX
    
    OptVector(x::X, T::Real) where {X} = new{X, length(x) + 1, real(eltype(x))}(x, convert(real(eltype(x)), T))
end


# ~~~ interface methods ~~~
Base.IndexStyle(::Type{<:OptVector})                      = IndexLinear()
Base.size(::OptVector{X, N}) where {X, N}                 = (N,)
Base.eltype(x::OptVector)                                 = eltype(x.x)
Base.similar(x::OptVector, ::Type{T}=eltype(x)) where {T} = OptVector(similar(x.x, T), real(T)(0.0))
Base.copy(x::OptVector)                                   = OptVector(copy(x.x), x.T)

@inline function Base.getindex(x::OptVector{X, N}, i::Int) where {X, N}
    if i < N
        @boundscheck checkbounds(x.x, i)
        @inbounds val = x.x[i]
    elseif i == N
        val = x.T
    else
        throw(BoundsError(x, i))
    end
    return val
end
function Base.setindex!(x::OptVector{X, N}, val, i::Int) where {X, N}
    if i < N
        @boundscheck checkbounds(x.x, i)
        @inbounds x.x[i] = val
    elseif i == N
        x.T = val
    else
        throw(BoundsError(x, i))
    end
end


# ~~~ broadcasting ~~~
const OptVectorStyle = Base.Broadcast.ArrayStyle{OptVector}
Base.BroadcastStyle(::Type{<:OptVector}) = OptVectorStyle()

@inline Base.similar(bc::Base.Broadcast.Broadcasted{OptVectorStyle}, ::Type{T}) where {T} = similar(find_opvec(bc), T)

find_opvec(bc::Base.Broadcast.Broadcasted) = find_opvec(bc.args)
find_opvec(args::Tuple)                    = find_opvec(find_opvec(args[1]), Base.tail(args))
find_opvec(x)                              = x
find_opvec(::Tuple{})                      = nothing
find_opvec(x::OptVector, rest)             = x
find_opvec(::Any, rest)                    = find_opvec(rest)

@inline function Base.copyto!(dest::OptVector, bc::Base.Broadcast.Broadcasted{OptVectorStyle})
    # flatten nested broadcasting representation
    bcf = Base.Broadcast.flatten(bc)

    # broadcast function over all the main vector arguments
    Base.Broadcast.broadcast!(bcf.f, dest.x, map(_get_x, bcf.args)...)

    # broadcast function ovar all the period arguments
    dest.T = Base.Broadcast.broadcast(bcf.f, map(_get_T, bcf.args)...)

    return dest
end

_get_x(x::OptVector)   = x.x
_get_x(x)              = x
_get_T(x::OptVector)   = x.T
_get_T(x)              = x
