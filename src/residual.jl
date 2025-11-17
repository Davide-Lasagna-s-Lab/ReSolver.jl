# Definition of the interface to define the object that calculates the residuals
# for a given generic input.


# -------------- #
# norm weighting #
# -------------- #
struct UniformWeight end
LinearAlgebra.mul!(x, ::UniformWeight) = x
LinearAlgebra.norm(x, ::UniformWeight) = norm(x)

# TODO: add outer constructor for single operator pass

# ------------------- #
# residual functional #
# ------------------- #
"""
Some type
"""
struct Residual{X, DT, NS, ADJ, F, A}
    cache::NTuple{5, X}
     dds!::DT
     rhs!::NS
     adj!::ADJ
     grad_scale!::F
     norm_weight::A

    Residual(x::X, 
          dds!::DT,
          rhs!::NS,
          adj!::ADJ;
    grad_scale::F=dRdx->dRdx,
   norm_weight::A=UniformWeight()) where {X, DT, NS, ADJ, F, A} = new{X, DT, NS, ADJ, F, A}(ntuple(i->similar(x), 5),
                                                                                            dds!,
                                                                                            rhs!,
                                                                                            adj!,
                                                                                            grad_scale,
                                                                                            norm_weight)
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

    # pre-compute residual
    R = norm(r, f.norm_weight)^2/2

    # scale residual according to norm weighting
    mul!(r, f.norm_weight)

    return R
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
