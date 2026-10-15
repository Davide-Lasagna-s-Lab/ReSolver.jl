# System: a dynamical system ∂t a = N(a) written in space-time form, and the quantities the
# methods need:
#
#     residual    r = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a)
#     objective   R = ½‖r‖²                                   both methods
#     gradient    ∇R, from the adjoint operator                L-BFGS
#     jacobian    𝒥 δx, from the linearised operator           Newton–Krylov hookstep
#     adjoint     𝒥⁺ w, from the adjoint operator              least-squares Krylov solvers
#
# With x = (a, ρ, c), ρ = log ω the unknown for the frequency, and ∂ᵢ the derivative along the
# i-th drift direction,
#
#     ∇ₐ R       = -ω ∂s r + Σᵢ cᵢ ∂ᵢ r - L⁺ r
#     ∂R/∂ρ      =  ω ⟨∂s a, r⟩,     since ∂ω/∂ρ = ω
#     ∂R/∂cᵢ     = -⟨∂ᵢ a, r⟩
#
# The gradient uses the skew-adjointness of the derivatives in the inner product, ∂⁺ = -∂.
#
# The operators and the derivatives are given by the user and opaque to the search, whatever the
# system and its discretisation. What they must satisfy, with a the field of an Orbit:
#
#     nl(out, a)               nonlinear operator N(a)
#     lin(out, b)              linearised operator L b, about the point set by linearise!
#     adj(out, b)              adjoint operator L⁺ b, about the same point
#     linearise!(op, a)        sets the linearisation point of op, lin or adj, to a
#     dds!(out, a)             derivative along the rescaled time s ∈ [0, 2π)
#     dd!(out, a)              derivative along a drift direction, one function per direction
#     B                        preconditioner, optional (precondition.jl)
#     project(a)               projection on an invariant subspace, in place, optional
#
# All write into `out`, of the type of `a`, and return it; the derivatives must be skew-adjoint in
# `dot`, ⟨∂a, b⟩ = -⟨a, ∂b⟩, as Fourier derivatives are. The fields broadcast and support `dot`
# and `similar`.
#
# Three constructors, by the operators given: N, L and L⁺ for both methods; N and L for the hookstep
# alone, since only L-BFGS applies L⁺; N alone for the hookstep with L approximated by finite
# differences of N (finitedifference.jl).
#
# The Newton system is square: the unknowns ω and cᵢ are matched by phase conditions that remove
# the neutral directions of time and space translations,
#
#     ⟨∂s a, δa⟩ = 0,    ⟨∂ᵢ a, δa⟩ = 0.
#
# Symmetric subspaces. If the system is equivariant under a symmetry S, an isometry with S² = I,
# its fixed fields Fix(S) = {a : S a = a} form a subspace that the operators map into itself: an
# orbit, its residual, its gradient and the Jacobian actions on fields of Fix(S) stay in Fix(S).
# The search then never leaves the subspace, except by rounding errors, which the orthogonal
# projection P = (I + S)/2, given as `project`, removes: it is applied to the initial orbit, to
# every residual, gradient and Jacobian action, and to every application of the preconditioner.

# Number of evaluations made through a System, cumulative over its lifetime.
mutable struct Evaluations
        residual::Int # residuals, one nonlinear operator each
        gradient::Int # gradients, one adjoint operator each
        jacobian::Int # Jacobian actions, one linearised operator each
         adjoint::Int # adjoint Jacobian actions, one adjoint operator each
    precondition::Int # applications of B⁻¹ or B⁻⁺

    Evaluations() = new(0, 0, 0, 0, 0)
end

# a snapshot of the counters, as a named tuple
counts(e::Evaluations) = (residual     = e.residual,
                          gradient     = e.gradient,
                          jacobian     = e.jacobian,
                          adjoint      = e.adjoint,
                          precondition = e.precondition)

"""
    System(nl, lin, adj, dds!, x::Orbit; linearise!, ddi=(), B=I, project=identity)
    System(nl, lin, dds!, x::Orbit; linearise!, ddi=(), B=I, project=identity)
    System(nl, dds!, x::Orbit; difference=:forward, ddi=(), B=I, project=identity)

The system with nonlinear operator `nl` and rescaled-time derivative `dds!`, on orbits shaped like
`x`. The three constructors differ in the operators given.

- With the linearised and adjoint operators `lin` and `adj`, and the function `linearise!(op, a)`
  that sets the linearisation point of `lin` or `adj`, the system serves both
  [`NewtonHookstep`](@ref) and [`LBFGS`](@ref).
- Without the adjoint operator, it serves the hookstep only, since only L-BFGS applies the adjoint.
  L-BFGS throws an `ArgumentError`.
- With the nonlinear operator alone, the system is Jacobian-free. The hookstep applies the
  linearised operator by finite differences of `nl`, with the step `ε = δ (1 + ‖a‖) / ‖b‖`.
  `difference=:forward` takes `L b ≈ (N(a + ε b) - N(a)) / ε`, one evaluation of `nl` per Jacobian
  action, with `δ` the square root of the machine precision of the fields. `difference=:central`
  takes `L b ≈ (N(a + ε b) - N(a - ε b)) / 2ε`, of second order and two evaluations per action,
  with `δ` the cube root. The residual is exact, so the converged orbit is as accurate as with the
  linearised operator. L-BFGS throws an `ArgumentError`.

For the Kuramoto–Sivashinsky example,

```julia
System(KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!, x;
       linearise!, ddi=(ddx!,))                                  # both methods
System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, ddi=(ddx!,))   # hookstep
System(KSNonlinear(g), dds!, x; ddi=(ddx!,))                     # hookstep, Jacobian-free
System(KSNonlinear(g), dds!, x; difference=:central, ddi=(ddx!,)) # second order
```

`ddi` is a tuple of functions `dd!(out, a) -> out`, the derivatives along the directions of
translational symmetry in which the orbit drifts with an unknown speed, one per drift speed in
`x.p`. The default `()` searches for periodic orbits. Leave out the directions in which a symmetry
of the problem forbids drifting.

`B` is the preconditioner, an invertible operator on orbits for which
[`precondition!`](@ref) and [`precondition_adjoint!`](@ref) are defined. It is the identity `I` by
default.

`project(a) -> a` restricts the search to a subspace of fields that the residual and the Jacobian
map into itself. It projects the field `a` in place, orthogonally, onto the subspace. Examples are
the fields fixed by a symmetry of the problem, and the fields of zero mean when the equation of the
mean is void, as for Kuramoto–Sivashinsky. The projection is applied to the initial orbit, and to
every residual, gradient, Jacobian action and application of the preconditioner, so that the
search stays in the subspace despite rounding errors. The default `identity` does nothing.
"""
struct System{NL, LIN, ADJ, DS, DD, LN, PB, PR, X, O<:Orbit}
            nl::NL           # nonlinear operator N(a)
           lin::LIN          # linearised operator L b, about the linearisation point
           adj::ADJ          # adjoint operator L⁺ b, about the same point
          dds!::DS           # derivative in rescaled time s
           ddi::DD           # derivatives along the drift directions
    linearise!::LN           # sets the linearisation point of lin or adj
             B::PB           # preconditioner
       project::PR           # projection on an invariant subspace, in place
         cache::NTuple{2, X} # scratch fields
      residual::O            # residual of the last call to objective
   evaluations::Evaluations  # counters of the evaluations

    function System(        nl,
                           lin,
                           adj,
                          dds!,
                             x::Orbit;
                    linearise!,
                           ddi::Tuple=(),
                             B=I,
                       project=identity)

        # ---- input checks ----
        length(ddi) == ndrift(x) ||
            throw(ArgumentError("one drift derivative per drift speed in x.p"))

        # ---- assemble ----
        cache    = (similar(x.a), similar(x.a))
        residual = similar(x)

        return new{typeof(nl),
                   typeof(lin),
                   typeof(adj),
                   typeof(dds!),
                   typeof(ddi),
                   typeof(linearise!),
                   typeof(B),
                   typeof(project),
                   typeof(x.a),
                   typeof(x)}(nl,
                              lin,
                              adj,
                              dds!,
                              ddi,
                              linearise!,
                              B,
                              project,
                              cache,
                              residual,
                              Evaluations())
    end

    # ---- without the adjoint operator: the hookstep only ----
    System(        nl,
                  lin,
                 dds!,
                    x::Orbit;
           linearise!,
                  ddi::Tuple=(),
                    B=I,
              project=identity) = System(nl,
                                         lin,
                                         nothing,
                                         dds!,
                                         x;
                                         linearise!,
                                         ddi,
                                         B,
                                         project)

    # ---- with the nonlinear operator alone: L by finite differences, the hookstep only ----
    function System(        nl,
                          dds!,
                             x::Orbit;
                    difference::Symbol=:forward,
                           ddi::Tuple=(),
                             B=I,
                       project=identity)

        # ---- input checks ----
        difference in (:forward, :central) ||
            throw(ArgumentError("difference must be :forward or :central"))

        # ---- assemble ----
        lin = FiniteDifference(nl, x.a, difference === :central)

        return System(nl,
                      lin,
                      nothing,
                      dds!,
                      x;
                      linearise! = _linearise!,
                      ddi,
                      B,
                      project)
    end
end


# ---------------------------------------------------------------------------- #
# residual and objective                                                       #
# ---------------------------------------------------------------------------- #
"""
    residual!(r::Orbit, F::System, x::Orbit) -> r

The augmented residual at `x`. Its field is `r.a = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a)`. Its parameters
`r.p` are the phase conditions, which vanish at the current point, so they are zero. Minus `r` is
the right-hand side of the Newton system.
"""
function residual!(r::Orbit, F::System, x::Orbit)
    tmp = F.cache[1]
    F.evaluations.residual += 1

    # ---- time derivative ----
    F.dds!(tmp, x.a)
    r.a .= frequency(x) .* tmp

    # ---- drift ----
    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, x.a)
        r.a .-= drift(x, i) .* tmp
    end

    # ---- nonlinear operator ----
    F.nl(tmp, x.a)
    r.a .-= tmp

    # ---- back onto the invariant subspace, against rounding errors ----
    F.project(r.a)

    # ---- phase conditions ----
    fill!(r.p, 0)

    return r
end

"""
    objective(F::System, x::Orbit) -> R

`R = ½‖r‖²`. The residual is left in `F.residual`.
"""
function objective(F::System, x::Orbit)
    r = residual!(F.residual, F, x)
    return norm(r)^2 / 2
end


# ---------------------------------------------------------------------------- #
# gradient, for L-BFGS                                                         #
# ---------------------------------------------------------------------------- #
"""
    gradient!(g::Orbit, F::System, x::Orbit) -> R

Write the gradient of `R = ½‖r‖²` into `g` and return `R`. Moves the linearisation point of the
operators to `x`.
"""
function gradient!(g::Orbit, F::System, x::Orbit)
    F.evaluations.gradient += 1

    # ---- residual and linearisation point ----
    R = objective(F, x)
    F.linearise!(F.adj, x.a)

    # ---- ∇R = 𝒥⁺ (r, 0): the residual has zero phase conditions ----
    _jacobian_adjoint!(g, F, x, F.residual)

    return R
end


# ---------------------------------------------------------------------------- #
# Jacobian, for Newton–Krylov                                                  #
# ---------------------------------------------------------------------------- #
"""
    jacobian!(out::Orbit, F::System, x::Orbit, δx::Orbit) -> out

The Jacobian of the residual at `x`, with the phase conditions as extra rows, applied to `δx`.
Assumes the linearisation point was set to `x` with `F.linearise!(F.lin, x.a)`.
"""
function jacobian!(out::Orbit, F::System, x::Orbit, δx::Orbit)
    tmp = F.cache[1]
    ∂a  = F.cache[2]
    F.evaluations.jacobian += 1

    # ---- operator on δa: ω ∂s δa - Σᵢ cᵢ ∂ᵢ δa - L δa ----
    F.dds!(tmp, δx.a)
    out.a .= frequency(x) .* tmp

    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, δx.a)
        out.a .-= drift(x, i) .* tmp
    end

    F.lin(tmp, δx.a)
    out.a .-= tmp

    # ---- log-frequency column ω ∂s a δρ and time phase condition ----
    F.dds!(∂a, x.a)
    out.a .+= (frequency(x) * δx.p[1]) .* ∂a
    out.p[1] = dot(∂a, δx.a)

    # ---- drift columns and space phase conditions ----
    for (i, dd!) in enumerate(F.ddi)
        dd!(∂a, x.a)
        out.a .-= drift(δx, i) .* ∂a
        out.p[1 + i] = dot(∂a, δx.a)
    end

    # ---- back onto the invariant subspace, against rounding errors ----
    F.project(out.a)

    return out
end


"""
    jacobian_adjoint!(out::Orbit, F::System, x::Orbit, w::Orbit) -> out

The adjoint of [`jacobian!`](@ref) in the inner product of the orbits, applied to `w`. With
`w = (w_a, [w_s, w₁, …])`, the field and the multipliers of the phase conditions,

    out.a = -ω ∂s w_a + Σᵢ cᵢ ∂ᵢ w_a - L⁺ w_a + w_s ∂s a + Σᵢ wᵢ ∂ᵢ a
    out.p = [ω ⟨∂s a, w_a⟩, -⟨∂₁ a, w_a⟩, …]

so that `⟨w, 𝒥 v⟩ = ⟨𝒥⁺ w, v⟩` for all orbits `v`. Assumes the linearisation point of the adjoint
operator was set to `x` with `F.linearise!(F.adj, x.a)`. Needed by Krylov solvers of the
least-squares problem `min ‖𝒥 δx + r‖`, such as LSQR. The gradient of `R = ½‖r‖²` is
`𝒥⁺ (r, 0)`.
"""
function jacobian_adjoint!(out::Orbit, F::System, x::Orbit, w::Orbit)
    F.evaluations.adjoint += 1
    return _jacobian_adjoint!(out, F, x, w)
end

# 𝒥⁺ w, not counted: shared by jacobian_adjoint! and gradient!
function _jacobian_adjoint!(out::Orbit, F::System, x::Orbit, w::Orbit)
    tmp = F.cache[1]
    ∂a  = F.cache[2]

    # ---- adjoint operator on w_a: -ω ∂s w_a + Σᵢ cᵢ ∂ᵢ w_a - L⁺ w_a ----
    F.dds!(tmp, w.a)
    out.a .= -frequency(x) .* tmp

    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, w.a)
        out.a .+= drift(x, i) .* tmp
    end

    F.adj(tmp, w.a)
    out.a .-= tmp

    # ---- time phase condition, transposed, and log-frequency row ω ⟨∂s a, w_a⟩ ----
    F.dds!(∂a, x.a)
    out.a .+= w.p[1] .* ∂a
    out.p[1] = frequency(x) * dot(∂a, w.a)

    # ---- space phase conditions, transposed, and drift rows -⟨∂ᵢ a, w_a⟩ ----
    for (i, dd!) in enumerate(F.ddi)
        dd!(∂a, x.a)
        out.a .+= w.p[1 + i] .* ∂a
        out.p[1 + i] = -dot(∂a, w.a)
    end

    # ---- back onto the invariant subspace, against rounding errors ----
    F.project(out.a)

    return out
end


# ---------------------------------------------------------------------------- #
# entry point, with one method per search                                      #
# ---------------------------------------------------------------------------- #
"""
    solve!(x::Orbit, F::System, method) -> x

Search for an orbit of the system `F`, starting from `x` and overwriting it with the result.
`method` is an [`LBFGS`](@ref) object, which minimises `½‖r‖²` and is robust far from a solution,
or a [`NewtonHookstep`](@ref) object, which solves `r = 0` and converges fast close to one. Both
hold their own options. A typical search runs L-BFGS first and refines with the hookstep.

```julia
solve!(x, F, LBFGS(maxiter=300))
solve!(x, F, NewtonHookstep(krylov_dim=150))
```
"""
function solve! end
