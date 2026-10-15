# The linearised operator of the Jacobian-free System, a finite difference of the nonlinear
# operator about the linearisation point a, forward or central,
#
#     L b ≈ (N(a + ε b) - N(a)) / ε             first order,  δ = eps^(1/2)
#     L b ≈ (N(a + ε b) - N(a - ε b)) / 2ε      second order, δ = eps^(1/3)
#
# with the step ε = δ (1 + ‖a‖) / ‖b‖ and eps the machine precision of the fields (Knoll & Keyes,
# J. Comput. Phys. 193, 2004). Each δ balances the truncation error against the rounding error,
# which leaves a relative error of the action of about 1e-8 (forward) and 5e-11 (central) in double
# precision, 3e-4 and 2e-5 in single. For a quadratic N the central difference is exact up to
# rounding. Either error is far below the tolerance of the Krylov solver, so the Newton steps are
# hardly changed. The residual is not approximated, so an orbit converged without the linearised
# operator satisfies the equations to the same tolerance.
#
# A forward action costs one evaluation of N, with N(a) evaluated once per Newton iteration, when
# the linearisation point is set. A central action costs two. The adjoint operator has no such
# matrix-free approximation, so a Jacobian-free system serves the hookstep only, not L-BFGS.

# The finite-difference linearised operator, callable as lin(out, b) -> out.
struct FiniteDifference{NL, X}
         nl::NL      # nonlinear operator N
          a::X       # linearisation point
        tmp::X       # perturbed field a ± ε b
       work::X       # N(a) for forward differences, N(a - ε b) for central ones
    central::Bool    # central differences, of second order, instead of forward ones
          δ::Float64 # relative step, eps^(1/2) forward or eps^(1/3) central

    FiniteDifference(nl::NL, a::X, central::Bool) where {NL, X} =
        new{NL, X}(nl,
                   similar(a),
                   similar(a),
                   similar(a),
                   central,
                   eps(real(eltype(a)))^(central ? 1 / 3 : 1 / 2))
end

# L b by the finite difference, written into out
function (J::FiniteDifference)(out, b)
    nb = norm(b)
    nb == 0 && return fill!(out, 0)

    # ---- step, in the precision of the fields ----
    ε = convert(real(eltype(b)), J.δ * (1 + norm(J.a)) / nb)

    # ---- N at a + ε b ----
    J.tmp .= J.a .+ ε .* b
    J.nl(out, J.tmp)

    # ---- forward difference, with N(a) from the linearisation point ----
    if !J.central
        out .= (out .- J.work) ./ ε
        return out
    end

    # ---- central difference, with N at a - ε b ----
    J.tmp .= J.a .- ε .* b
    J.nl(J.work, J.tmp)
    out .= (out .- J.work) ./ 2ε

    return out
end

# sets the linearisation point to a, and evaluates N there for forward differences
function _linearise!(J::FiniteDifference, a)
    J.a .= a
    J.central || J.nl(J.work, J.a)
    return J
end
