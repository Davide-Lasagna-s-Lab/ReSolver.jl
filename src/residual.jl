# Definition of the interface to define the object that calculates the residuals
# for a given generic input.

export Residual

# ! the base profile should be handled inside the navier-stokes operator call
# ! any norm scaling should be handled inside the norm method and adjoint operator calls

"""
Some type
"""
struct Residual{X, DT, NS, ADJ, F}
    cache::NTuple{5, X}
     dds!::DT
     rhs!::NS
     adj!::ADJ
     grad_scale!::F

    Residual(x::X, 
             dds!::DT,
             rhs!::NS,
             adj!::ADJ,
             grad_scale!::F=dRdx->dRdx) where {X, DT, NS, ADJ, F} = new{X, DT, NS, ADJ, F}(ntuple(i->similar(x), 5),
                                                                                           dds!,
                                                                                           rhs!,
                                                                                           adj!,
                                                                                           grad_scale!)
end

"""
A call
"""
function (f::Residual{X})(x::X, T::Real) where {X}
    # aliases
    dxds = f.cache[1]
    N_x  = f.cache[2]
    r    = f.cache[3]

    # compute fundamental frequency
    ω = 2π/T

    # compute local residual
    r .= ω.*f.dds!(dxds, x) .- f.rhs!(N_x, x)

    return norm(r)^2/2
end

"""
A second call
"""
function (f::Residual{X})(dRdx::X, x::X, T::Real) where {X}
    # aliases
    dxds  = f.cache[1]
    r     = f.cache[3]
    drds  = f.cache[4]
    M_x_r = f.cache[5]

    # compute fundamental frequency
    ω = 2π/T

    # compute residual
    R = f(x, T)

    # compute field gradient
    dRdx .= .-ω.*f.dds!(drds, r) .- f.adj!(M_x_r, x, r)

    # scaling of gradient required for real FFTs
    f.grad_scale!(dRdx)

    # compute frequency gradient
    dRdT = -(ω/T)*dot(dxds, r)

    return R, dRdx, dRdT
end
