# Wrapper object for generic type and period to interface with optimisation
# packages seemlessly

# This type is mutable to allow the period to be altered over the duration of
# the optimisation. The only objects that will actually be of this type are
# the actual optimisation variable, the gradient, and any internal copies
# made by the optimisation package. Every other case will treat them as
# separate objects. This is also a good way to add in dependence on continuous
# symmetries similar to MVector in NKSearch.jl.

# ~~~ Wrapper vector struct for optimisation ~~~
mutable struct OptimVector{X, N} <: AbstractVector{Float64}
    const x::X
          T::Float64
    
    OptimVector(x::X, T::Real) where {X} = new{X, length(x) + 1}(x, convert(Float64, T))
end


# ~~~ interface methods ~~~
Base.IndexStyle(::Type{<:OptimVector})            = IndexLinear()
Base.size(::OptimVector{X, N}) where {X, N}       = (N,)
Base.eltype(x::OptimVector)                       = eltype(x.x)
Base.similar(x::OptimVector, ::Type{T}) where {T} = OptimVector(similar(x.x, T), 0.0)
Base.copy(x::OptimVector)                         = OptimVector(copy(x.x), x.T)

# storing "N" as parametric type allows compiler to optimise away the if-else blocks through constant propagation
@inline function Base.getindex(x::OptimVector{X, N}, i::Int) where {X, N}
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


# ~~~ broadcasting ~~~
const OptimVectorStyle = Base.Broadcast.ArrayStyle{OptimVector}
Base.BroadcastStyle(::Type{<:OptimVector}) = OptimVectorStyle()

@inline Base.similar(bc::Base.Broadcast.Broadcasted{OptimVectorStyle}, ::Type{T}) where {T} = similar(find_opvec(bc), T)

find_opvec(bc::Base.Broadcast.Broadcasted) = find_opvec(bc.args)
find_opvec(args::Tuple)                    = find_opvec(find_opvec(args[1]), Base.tail(args))
find_opvec(x)                              = x
find_opvec(::Tuple{})                      = nothing
find_opvec(x::OptimVector, rest)           = x
find_opvec(::Any, rest)                    = find_opvec(rest)

@inline function Base.copyto!(dest::OptimVector, bc::Base.Broadcast.Broadcasted{OptimVectorStyle})
    # flatten nested broadcasting representation
    bcf = Base.Broadcast.flatten(bc)

    # broadcast function over all the main vector arguments
    Base.Broadcast.broadcast!(bcf.f, dest.x, map(_get_x, bcf.args)...)

    # broadcast function ovar all the period arguments
    dest.T = Base.Broadcast.broadcast(bcf.f, map(_get_T, bcf.args)...)

    return dest
end

_get_x(x::OptimVector) = x.x
_get_x(x)              = x
_get_T(x::OptimVector) = x.T
_get_T(x)              = x
