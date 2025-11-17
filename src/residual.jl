# Definition of the interface to define the object that calculates the residuals
# for a given generic input.

# -------------- #
# norm weighting #
# -------------- #
struct UniformWeight end
LinearAlgebra.mul!(x, ::UniformWeight) = x
LinearAlgebra.norm(x, W) = sqrt(dot(x, W, x))
LinearAlgebra.dot(x, ::UniformWeight, y) = dot(x, y)


# ------------------- #
# residual functional #
# ------------------- #
struct Residual{X, DT, NS, ADJ, F, A}
    cache::NTuple{5, X}
     dds!::DT
     rhs!::NS
     adj!::ADJ
     grad_scale!::F
     norm_weight::A

    """
        Residual(x::X,
                 dds!,
                 rhs!,
                 adj!;
                 grad_scale=x->x,
                 norm_weight=UniformWeight()) -> Residual{X}

    Construct a `Residual` object that can be used to compute the global residual
    and gradient for an optimisation problem.

    # Arguments
    - `x::X`: base optimisation variable that the residual takes as input
    - `dds!`: time derivative operator, `dds!(::X, ::X)`
    - `rhs!`: nonlinear operator for the dynamical system, `rhs!(::X, ::X)`
    - `adj!`: adjoint linearised operator for the dynamical system,
              `adj!(::X, ::X)`
    - `grad_scale`: scaling operator for the gradient, required if real FFT's
                    are used to account for the hermitian symmetry, must operate
                    in place with the signature `grad_scale(::X)`
    - `norm_weight`: weighting operator for the inner-product space, must be
                     symmetric and positive definite

    # Interface requirements for `X`
    - `dot(::X, ::X)->Real`
    - `similar(::X)->::X`
    - ::X must be broadcastable
    - `LinearAlgebra.mul!(::X, norm_weight)->::X`
    - `LinearAlgebra.dot(::X, norm_weight, ::X)->Real`
    """
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
Same as `Residual` constructor except the single object `op!` is used as both
the nonlinear, `rhs!`, and adjoint linearised, `adj!`, operators.
"""
Residual(x,
         dds!,
         op!;
         grad_scale=dRdx->dRdx,
         norm_weight=UniformWeight()) = Residual(x,
                                                 dds!,
                                                 op!,
                                                 op!,
                                                 grad_scale=grad_scale,
                                                 norm_weight=norm_weight)


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
    # the inner-product doesn't need weighting because the residual has already been modified

    return R, dRdx, dRdT
end
