# Preconditioning: an invertible operator B on orbits, given by the user, that defines the metric
#
#     ⟨p, q⟩_B = ⟨B p, B q⟩ = ⟨p, M q⟩,     M = B⁺ B,
#
# in which both methods work (see the Preconditioning page of the documentation). L-BFGS applies
# M⁻¹ = B⁻¹ B⁻⁺, the hookstep B⁻¹ as a right preconditioner. The search only applies the inverses, through two
# functions that a preconditioner type extends:
#
#     precondition!(out, B, p)            out = B⁻¹ p
#     precondition_adjoint!(out, B, p)    out = B⁻⁺ p, with B⁻⁺ the adjoint of B⁻¹
#     relinearise!(B, x)                  B rebuilt about the orbit x, optional
#     metric!(out, B, p)                  out = M p, the operator whose norm measures the steps of the hookstep
#                                         with metric=:operator, optional
#
# A multiple of the identity, `I` by default, leaves both methods as without preconditioning. The
# hookstep calls relinearise! at the start of every Newton iteration, with the current orbit; L-BFGS
# never does, since its metric must stay fixed for the curvature pairs to make sense.

"""
    precondition!(out::Orbit, B, p::Orbit) -> out

Apply the inverse of the preconditioner `B` to the orbit `p`, `out = B⁻¹ p`. Extend it, together
with [`precondition_adjoint!`](@ref), for each preconditioner type.
"""
precondition!(out::Orbit, B::UniformScaling, p::Orbit) = (out .= p ./ B.λ)

"""
    precondition_adjoint!(out::Orbit, B, p::Orbit) -> out

Apply the adjoint of the inverse of the preconditioner `B` to the orbit `p`, `out = B⁻⁺ p`, in
the inner product of the orbits. For a self-adjoint `B` it is [`precondition!`](@ref).
"""
precondition_adjoint!(out::Orbit, B::UniformScaling, p::Orbit) = (out .= p ./ conj(B.λ))

"""
    metric!(out::Orbit, B, p::Orbit) -> out

Apply to the orbit `p` the operator `M` whose norm measures the steps of
[`NewtonHookstep`](@ref) with `metric=:operator`, `out = M p`: usually the preconditioner applied
forward. Extend it for a preconditioner whose inverse is not one fixed operator, e.g. one that
solves some directions exactly, so that the trust region keeps a fixed metric. For a multiple of
the identity it multiplies.
"""
metric!(out::Orbit, B::UniformScaling, p::Orbit) = (out .= B.λ .* p)

"""
    relinearise!(B, x::Orbit) -> B

Rebuild the preconditioner `B` in place about the orbit `x`, its frequency, drift speeds and
field. [`NewtonHookstep`](@ref) calls it at the start of every Newton iteration with the current
orbit, unless `relinearise=false`. L-BFGS never calls it. The default does nothing, so a
preconditioner that does not depend on the orbit, or the identity, needs no method.
"""
relinearise!(B, x::Orbit) = B

# applications of B⁻¹ and B⁻⁺ through a System, counted, and projected back onto the invariant
# subspace of the System, which a preconditioner that does not commute with the symmetry would leave
function _precondition!(out::Orbit, F, p::Orbit)
    F.evaluations.precondition += 1
    precondition!(out, F.B, p)
    F.project(out.a)

    return out
end

function _precondition_adjoint!(out::Orbit, F, p::Orbit)
    F.evaluations.precondition += 1
    precondition_adjoint!(out, F.B, p)
    F.project(out.a)

    return out
end

# out = M⁻¹ p = B⁻¹ B⁻⁺ p, with tmp as scratch
function _metric_inverse!(out::Orbit, F, p::Orbit, tmp::Orbit)
    _precondition_adjoint!(tmp, F, p)
    _precondition!(out, F, tmp)

    return out
end
