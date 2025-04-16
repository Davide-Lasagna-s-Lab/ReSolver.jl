# Definition of the interface to define the object that calculates the residuals
# for a given generic input.


# ~~~ Assumed interface for Residual ~~~
#  - similar(::X) -> X
#  - dot(::X, ::X) -> Number
#  - norm(::X) -> Non-negative Real
#  - broadcasting between variables of type X
#  - get_period(::X) -> Real
#  - set_period!(::X, Real) -> X

# ! the base profile should be handled inside the navier-stokes operator call
# ! any norm scaling should be handled inside the norm method and adjoint operator calls

"""
Some type
"""
struct Residual{X, DT, NS, ADJ}
            cache::NTuple{5, X}
             ddt!::DT
    rhs_operator!::NS
    adj_operator!::ADJ

    Residual(x::X, 
             ddt!::DT,
             rhs_operator!::NS,
             adj_operator!::ADJ) where {X, DT, NS, ADJ} = new{X, DT, NS, ADJ}(ntuple(i->similar(x), 5),
                                                                                     ddt!,
                                                                                     rhs_operator!,
                                                                                     adj_operator!)
end

"""
A call
"""
function (f::Residual{X})(a::X) where {X}
    # aliases
    dadt = f.cache[1]
    N_a  = f.cache[2]
    s    = f.cache[3]

    # compute local residual
    s .= ddt!(dadt, a) .- f.rhs_operator!(N_a, a)

    return norm(s)^2/2
end

"""
A second call
"""
function (f::Residual{X})(dRdx::X, a::X) where {X}
    # aliases
    s     = f.cache[3]
    dsdt  = f.cache[4]
    M_a_s = f.cache[5]

    # compute residual
    R = f(a)

    # compute field gradient
    dRdx .= f.adj_operator!(M_a_s, a, s) .- ddt!(dsdt, s)

    # compute frequency gradient
    dRdT = dot(dadt, s)/get_period(a)
    set_period!(dRdx, dRdT)

    return dRdx, R
end



# TODO: move optimisation stuff into this package so this stuff isn't necessary
function (f::Residual{X})(R, G, a::X) where {X}
    if G === nothing
        return f(a)
    else
        return f(G, a)
    end
end

function (f::Residual)(R, G, x::Vector{Float64})
    a = vectorToField!(f.projectedCache[3], x)
    dRda = f.projectedCache[4]
    if G === nothing
        R = f(a)[1]
    else
        R, dRdω = f(dRda, a)
        fieldToVector!(G, dRda, dRdω)
    end
    return R
end
