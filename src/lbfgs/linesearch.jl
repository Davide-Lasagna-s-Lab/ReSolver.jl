# Backtracking Armijo line search.
#
# Starting from a unit step, the step α along the direction p is halved until
#
#     R(x + α p) ≤ R(x) + c₁ α ⟨g, p⟩,
#
# or until it falls below the smallest step or the attempts run out. The trial point is left in
# `xt` even when the condition fails: the caller decides what to do with it.

const ARMIJO      = 1e-4         # sufficient-decrease constant c₁
const CONTRACTION = 0.5          # step reduction after each failed attempt
const MIN_STEP    = eps(Float64) # smallest step tried
const MAX_TRIALS  = 25           # largest number of attempts

function _linesearch!(xt::Orbit,
                       F::System,
                       x::Orbit,
                       p::Orbit,
                       g::Orbit,
                       R::Real)

    # ---- slope along the direction ----
    dR = dot(g, p)
    α  = 1.0

    for _ in 1:MAX_TRIALS
        # ---- trial point and Armijo test ----
        xt .= x .+ α .* p
        Rt  = objective(F, xt)

        isfinite(Rt) && Rt <= R + ARMIJO * α * dR && return α, Rt

        # ---- shorter step ----
        α *= CONTRACTION
        α < MIN_STEP && break
    end

    # ---- no sufficient decrease: return the last trial point ----
    xt .= x .+ α .* p

    return α, objective(F, xt)
end
