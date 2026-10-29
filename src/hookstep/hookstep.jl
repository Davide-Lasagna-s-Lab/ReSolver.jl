# Newton–Krylov hookstep (Viswanath 2007, https://arxiv.org/abs/0809.1498), as in GMRES.jl and
# NKSearch.
#
# Newton's method solves r(x) = 0 by repeatedly solving the linear system
#
#     𝒥 δx = -r,                                                                   (1)
#
# with 𝒥 the Jacobian of the residual at x, augmented with the phase conditions so that (1) is
# square (see system.jl). Far from a solution the full Newton step is unreliable, so the step is
# restricted to a trust region ‖δx‖ ≤ Δ, inside which the linear model r + 𝒥 δx is trusted:
#
#     min ‖r + 𝒥 δx‖   subject to   ‖δx‖ ≤ Δ.                                      (2)
#
# Krylov space. 𝒥 is never formed: only its action on a vector, `jacobian!`, is available. The
# Arnoldi iteration started from b = -r builds an orthonormal basis Qₙ of the Krylov space
# span{b, 𝒥 b, …, 𝒥ⁿ⁻¹ b} and the (n+1) × n Hessenberg matrix H with 𝒥 Qₙ = Qₙ₊₁ H. For
# δx = Qₙ y, and since Qₙ₊₁ has orthonormal columns with first column b/β, β = ‖b‖,
#
#     ‖r + 𝒥 δx‖ = ‖Qₙ₊₁ (g - H y)‖ = ‖g - H y‖,     g = (β, 0, …, 0),
#
# so (2) restricted to the Krylov space is a small problem in y, ‖y‖ = ‖δx‖:
#
#     min ‖g - H y‖   subject to   ‖y‖ ≤ Δ.                                        (3)
#
# Hookstep. Problem (3) is solved exactly from the SVD of H (`_hookstep` below). Inside the trust
# region its solution is the GMRES least-squares solution; on the boundary it bends from the
# Newton direction towards steepest descent, hence the name. The Krylov space stops growing when
# the residual of the GMRES solution, the unconstrained minimiser of ‖g - H y‖, falls below
# `krylov_tol` β, or at dimension `krylov_dim`, and (3) is then solved in that space (Viswanath
# 2007). The criterion is on the unconstrained problem on purpose: when the trust region is active,
# the residual of the hookstep cannot fall below a level set by Δ, and a criterion on it would grow
# the Krylov space to `krylov_dim` at every step limited by the trust region.
#
# Preconditioning. With a preconditioner B the Arnoldi iteration runs on 𝒥 B⁻¹ and the step is
# δx = B⁻¹ Qₙ y (right preconditioning): the residual of the model is unchanged, and the trust
# region ‖y‖ ≤ Δ bounds ‖B δx‖, the length of the step in the metric of B (see the Preconditioning
# page of the documentation). A preconditioner applied inexactly, by an inner iterative solver,
# changes from one Arnoldi step to the next, and B⁻¹ Qₙ y is then not the step that the model
# describes. With `flexible` the vectors zⱼ = B⁻¹ qⱼ of the Arnoldi steps are kept, 𝒥 Zₙ = Qₙ₊₁ H
# holds whatever the preconditioner did, and the step is δx = Zₙ y (flexible GMRES, Saad 1993): the
# model is exact again, at the cost of storing Zₙ, and no B⁻¹ is applied to form the steps.
#
# Metric of the trust region (`metric=:operator`). The radius bounds ‖y‖, which is ‖B δx‖ only while B⁻¹ is one fixed
# operator. A preconditioner that solves some directions exactly, as a slow solve on a few directions plus a cheap
# factorisation on the rest, maps a unit y to a long step along the directions on which 𝒥 is small, and the trust
# region no longer bounds them. With metric=:operator the length of a step is ‖M δx‖, M applied by `metric!` of the
# preconditioner (precondition.jl), e.g. the cheap factorisation applied forward: with flexible GMRES and
# M Zₙ = Wₙ R, Wₙ orthonormal and R upper triangular, ‖M δx‖ = ‖R y‖, built one column per Arnoldi step by
# Gram-Schmidt, and (3) keeps its form in u = R y, with H R⁻¹ in place of H.
#
# Trust-region update. The linear model predicts the reduction ½(β² - ‖g - H y‖²) of ½‖r‖²; the
# step is evaluated by the ratio ρ of the actual reduction to the predicted one:
#
#     ρ < 1/4                  poor model: Δ shrinks by 4,
#     ρ > 3/4 on the boundary  good model, step limited by Δ: Δ doubles, up to `Δmax`,
#     ρ > eta                  the step is accepted, otherwise it is solved again from (3) with
#                              the smaller Δ in the same Krylov space, without new Jacobian actions.
#
# Scan of the radius (`trust=:scan`). A trial step costs one residual and one combination of the
# Krylov vectors, against the Jacobian actions of the whole Krylov space, so several radii are
# tried in the same space: Δ p for the factors p of SCAN, 1/8 to 4 in steps of √2, capped by `Δmax`
# and by the length of the GMRES step, which every larger radius gives. Among the trials with
# ρ > eta, the one of smallest residual is accepted, and the next iteration starts from the length
# of its step. If none has ρ > eta, the scan starts again from the smallest radius tried over 8.
#
# The output reports, for every iteration, whether the accepted step is the full Newton step, inside
# the trust region, or a hookstep on its boundary, and how many trial steps were rejected first.
#
# Symmetric subspaces. With a projection in the System (keyword `project`), the System projects the
# initial orbit, the residual, every Jacobian action and every application of B⁻¹, so the Arnoldi
# basis, the step and the iterates stay in the subspace; nothing is projected here.


"""
    NewtonHookstep(; krylov_dim=50, krylov_tol=1e-3, Δ=1e-2, Δmax=1, eta=1e-3, maxiter=20,
                     tol=1e-10, flexible=false, relinearise=true, recycle=0, trust=:ratio,
                     metric=:krylov, verbose=true, io=stdout, callback=info -> false)

Newton–Krylov hookstep solution of `r = 0`, for [`solve!`](@ref). Fast near a solution.

Each Newton iteration builds a Krylov space of the right-preconditioned Jacobian, up to dimension
`krylov_dim` or until the GMRES solution of the Newton system has a residual below `krylov_tol`
times `‖r‖`. The step is the hookstep in that space, the minimiser of the linear model within the
trust region. The radius starts at `Δ` and never exceeds `Δmax`. A step is accepted when the actual
reduction of `½‖r‖²` is at least `eta` times the predicted one.

- `flexible=true` keeps the preconditioned Arnoldi vectors and forms the step from them (flexible
  GMRES). Use it with a preconditioner that changes between Arnoldi steps, e.g. one applied by an
  inner iterative solver. It doubles the memory of the Krylov space.
- `relinearise=true` rebuilds the preconditioner about the current orbit at the start of every
  Newton iteration, by [`relinearise!`](@ref), which does nothing for a preconditioner that defines
  no method.
- The search stops when `‖r‖ < tol`, after `maxiter` iterations, or when `callback(info)` returns
  `true`. The callback is called at the start and after every iteration. `info` is a named tuple
  with the iteration `iter`, the orbit `x`, the residual norm `res`, the cumulative `evaluations`
  of the system, `krylov`, the relative residual of the GMRES solution after each Arnoldi step of
  the iteration, `step`, `"newton"` or `"hook"`, the kind of the accepted step, and `trust`, a
  named tuple with the trust-region radius `radius` with which the accepted step was computed, its
  length `length` in the metric of the preconditioner, the ratio `ratio` of the actual to the
  predicted reduction of `½‖r‖²`, the number `rejected` of trial steps rejected before it, and
  `trials`, the radius and the ratio of each of them. A
  [`Trace`](@ref) records them.
- `recycle=k` carries k directions of the step from one Newton system to the next: the right
  singular vectors of the smallest singular values of the restricted Jacobian of the last system,
  the slow directions that GMRES would otherwise find again. Each system then applies the Jacobian
  to them once, k more actions, and builds its Krylov space orthogonal to their images. The default
  0 recycles nothing.
- `trust=:ratio` updates the radius from the ratio ρ of each trial: by 4 down below 1/4, by 2 up
  above 3/4 on the boundary. `trust=:scan` tries the hookstep at the radii Δ p, p = 2^(j/2) for
  j = -6, …, 4, in the same Krylov space, accepts the trial of smallest residual among those with
  ρ > `eta`, and starts the next iteration from the length of its step. Each trial costs one
  residual. `trials` then holds every trial but the accepted one, and `rejected` those with
  ρ ≤ `eta`.
- `metric=:krylov` measures a step by its coordinates in the Krylov space, ‖B δx‖ for a fixed
  preconditioner. `metric=:operator` measures it by ‖M δx‖, with M applied by
  [`metric!`](@ref) of the preconditioner, for a preconditioner whose inverse is not one fixed
  operator, e.g. one that solves some directions exactly. It needs `flexible=true`, and neither
  recycling nor a free parameter, and keeps the vectors M B⁻¹ qⱼ, one more field per Arnoldi step.
- `parameter=(; derivative!, update!, value)` leaves a parameter σ of the residual free, an unknown of
  the Newton step with no extra equation. `derivative!(out, x)` writes ∂r/∂σ at `x` into the orbit
  `out`, with zero phase conditions, `update!(δσ)` changes the parameter of the system by δσ, and
  `value()` returns it. The Newton system then has one more unknown than equations, and its
  solutions near an orbit form a branch along σ: the hookstep takes the correction of least norm
  within its Krylov space, whose solutions converge to a point of the branch near the guess. The
  column ∂r/∂σ enters the space as a recycled direction would, with no Jacobian action, and the
  trust region measures the change of σ by the change of the residual it produces. `info.parameter`
  holds the value. The default `nothing` keeps the parameter fixed. Not with `recycle > 0`.
- `verbose=true` prints one line per iteration to `io`.
"""
Base.@kwdef struct NewtonHookstep
    krylov_dim::Int      = 50            # largest dimension of the Krylov space
    krylov_tol::Real     = 1e-3          # relative GMRES residual stopping the Krylov space
             Δ::Real     = 1e-2          # initial trust-region radius
          Δmax::Real     = 1             # largest trust-region radius
           eta::Real     = 1e-3          # smallest actual/predicted reduction to accept
       maxiter::Int      = 20            # largest number of Newton iterations
           tol::Real     = 1e-10         # stop when ‖r‖ < tol
      flexible::Bool     = false         # keep B⁻¹ qⱼ: B⁻¹ may change between Arnoldi steps
   relinearise::Bool     = true          # rebuild B about the orbit at every Newton iteration
       recycle::Int      = 0             # directions carried from one Newton system to the next
         trust::Symbol   = :ratio        # radius update: :ratio, or :scan of several radii
        metric::Symbol   = :krylov       # length of a step: :krylov, ‖y‖, or :operator, ‖M δx‖ (metric!)
     parameter::Any      = nothing       # a free parameter: nothing, or (; derivative!, update!, value)
       verbose::Bool     = true          # print one line per iteration
            io::IO       = stdout        # where to print
      # any callable, in a field of abstract type, so that solve! is compiled once for all callbacks
      callback::Any      = info -> false # called every iteration; true stops
end


# ---------------------------------------------------------------------------- #
# Newton iterations                                                            #
# ---------------------------------------------------------------------------- #
# the factors of the radius tried by trust=:scan, 1/8 to 4 in steps of √2
const SCAN = 2.0 .^ (-3:0.5:2)

function solve!(     x::Orbit,
                     F::System,
                method::NewtonHookstep)

    # Δ is the trust-region radius: it starts from the option and changes at every iteration
    (; krylov_dim, krylov_tol, Δ, Δmax, eta, maxiter, tol, flexible, relinearise, recycle, trust,
       metric, parameter, verbose, io, callback) = method

    # ---- input checks ----
    trust in (:ratio, :scan) ||
        throw(ArgumentError("trust must be :ratio or :scan"))
    parameter === nothing || recycle == 0 ||
        throw(ArgumentError("a free parameter cannot be combined with recycled directions"))
    metric in (:krylov, :operator) ||
        throw(ArgumentError("metric must be :krylov or :operator"))
    metric === :krylov || (flexible && recycle == 0 && parameter === nothing) ||
        throw(ArgumentError("metric=:operator needs flexible=true, and neither recycling nor a free parameter"))

    # ---- workspace ----
    b  = similar(x) # right-hand side of (1), (-r, 0): no phase condition on the step
    z  = similar(x) # preconditioned step Qₙ y
    δx = similar(x) # Newton step B⁻¹ Qₙ y
    xn = similar(x) # trial point x + δx
    xb = similar(x) # best trial point of a scan of the radius
    w  = similar(x) # B⁻¹ v inside the Arnoldi operator
    U  = typeof(x)[] # recycled directions of the step, carried from one Newton system to the next
    dp = similar(x) # the column ∂r/∂σ of a free parameter, normalised
    nd = 1.0        # its norm
    ϵ  = 100 * eps(typeof(dot(x.a, x.a))) # with metric=:operator, the relative singular value below which the
                                          # vectors M zⱼ, scaled to unit length, depend on each other to rounding,
                                          # in the precision of x

    # ---- initial residual, on the invariant subspace of the system ----
    F.project(x.a)
    R = objective(F, x)

    verbose && _print_newton_header(io)
    verbose && _print_newton_row(io, 0, R, frequency(x), Δ, NaN, 0, NaN, "", 0)
    stop = callback(_info(0, x, R, F, Float64[], "", (radius=Δ, length=NaN, ratio=NaN, rejected=0,
                                                       trials=Tuple{Float64, Float64}[]);
                          parameter = parameter === nothing ? NaN : parameter.value()))

    for iter in 1:maxiter
        (stop || sqrt(2R) < tol) && break

        # ---- Newton system (1) at x ----
        # the Jacobian acts about x: set the linearisation point of the operators, and of the
        # preconditioner if it follows the orbit
        F.linearise!(F.lin, x.a)
        relinearise && relinearise!(F.B, x)

        residual!(b, F, x)
        b .*= -1
        β = norm(b)

        # ---- recycled space: its images C = 𝒥 U at x, orthonormal, and the part r₀ of b off them; or
        #      the column ∂r/∂σ of a free parameter, normalised, which needs no Jacobian action (U then
        #      stays empty, and the first coordinate of the step is ‖∂r/∂σ‖ δσ) ----
        if parameter === nothing
            C, Rc = _recycle!(U, F, x)
        else
            parameter.derivative!(dp, x)
            nd = norm(dp)
            dp ./= nd
            C  = [dp]
            Rc = Matrix(1.0I, 1, 1)
        end
        k  = length(C)
        cb = [dot(c, b) for c in C]
        r₀ = copy(b)
        for (c, h) in zip(C, cb)
            r₀ .-= h .* c
        end
        β₀ = norm(r₀)

        # ---- Krylov space, one Arnoldi step at a time ----
        # right preconditioning: the Arnoldi operator is 𝒥 B⁻¹, and the step δx = B⁻¹ z; flexible:
        # the operator keeps each B⁻¹ qⱼ in Z, and the step is δx = Z y
        # Arnoldi normalises its starting vector in place, hence the copy of r₀; with a recycled space
        # the operator is followed by the projection off C, whose coefficients E = Cᵀ 𝒥 zⱼ are kept
        Z   = typeof(x)[]
        E   = Vector{Float64}[]
        W   = typeof(x)[]          # with metric=:operator, the vectors M zⱼ orthonormalised
        Gw  = Vector{Float64}[]    # and the columns of R, M Zₙ = Wₙ R, upper triangular
        A!  = flexible ?
              (out, v) -> (push!(Z, _precondition!(similar(x), F, v)); _metric!(W, Gw, F, Z[end], metric);
                           jacobian!(out, F, x, Z[end]); _deflate!(out, C, E)) :
              (out, v) -> (jacobian!(out, F, x, _precondition!(w, F, v)); _deflate!(out, C, E))
        arn = ArnoldiIteration(A!, copy(r₀))
        g   = [β₀]

        H      = arn.H
        krylov = Float64[]                        # relative residual of the GMRES solution
        qr     = (c=Float64[], s=Float64[], γ=[β₀]) # Givens QR of H, updated one column at a time

        for _ in 1:(β₀ > 1e-12 * β ? krylov_dim : 0)
            # one more basis vector, one more column of H, one more entry of g
            _, H = step!(arn)
            push!(g, 0.0)

            # least-squares (GMRES) solution in the current space: stop once it solves the Newton
            # system well enough. The criterion ignores the trust region: on its boundary the
            # residual of the hookstep stalls at a level set by Δ, which no larger space can lower
            push!(krylov, _givens!(qr, H[:, end]) / β)
            krylov[end] < krylov_tol && break

            # breakdown: the new direction lies in the space, which no further step can enlarge
            H[end, end] <= 1e-12 * β && break
        end

        # hookstep (3) in the final space, enlarged by the recycled directions: with W = [C Qₙ₊₁],
        # 𝒥 [U Zₙ] = W G and b = W g̃, so that the problem keeps the form (3) in the coordinates
        # c = (a, y) of the step U a + Zₙ y; the SVD of G is reused by the shorter hooksteps of rejected
        # trials, since the space does not change
        m              = size(H, 2)
        G              = [Rc reduce(hcat, E[1:m]; init=zeros(k, 0)); zeros(m + 1, k) H]
        g̃              = [cb; g]

        # (with metric=:operator, the problem in u = R y, the hookstep back in y, and the length of a step ‖R y‖; with
        #  metric=:krylov R = I)
        Rm, Ri         = _root(Gw, m, metric, ϵ)
        SH             = svd(G * Ri)
        hook(r)        = (h = _hookstep(SH, g̃, r); (Ri * h[1], h[2]))
        len(y)         = norm(Rm * y)
        y, at_boundary = hook(Δ)

        # ---- trust region: accept the step, or shrink Δ and solve (3) again ----
        ρ        = 0.0
        rejected = 0 # trial steps rejected before the accepted one
        trials   = Tuple{Float64, Float64}[] # their radius and ratio ρ
        used     = Δ # the radius of the hookstep that is finally accepted
        while trust === :ratio
            # step in the full space, U a + Zₙ y, and trial point, with a free parameter changed by
            # δσ = y₁/‖∂r/∂σ‖
            _step!(δx, z, y, k, U, Z, arn.Q, F, flexible)
            xn .= x .+ δx
            δσ  = parameter === nothing ? 0.0 : y[1] / nd
            parameter === nothing || parameter.update!(δσ)

            # actual and predicted reductions of ½‖r‖²
            Rn        = objective(F, xn)
            predicted = (norm(g̃)^2 - norm(g̃ - G * y)^2) / 2
            ρ         = (R - Rn) / predicted

            # radius update; a ratio that is not a number, e.g. from a vanishing predicted
            # reduction or a residual that overflows, counts as a poor step
            if !(ρ >= 1/4)
                Δ /= 4
            elseif ρ > 3/4 && at_boundary
                Δ = min(2Δ, Δmax)
            end

            # accepted: move to the trial point
            if ρ > eta
                x .= xn
                R  = Rn
                break
            end

            # rejected: the parameter back, then give up if the trust region has collapsed, otherwise
            # a shorter hookstep
            parameter === nothing || parameter.update!(-δσ)
            if Δ < eps() * norm(x)
                verbose && println(io, "  trust region collapsed: no step in the Krylov space reduces ‖r‖")
                objective(F, x)
                return x
            end
            push!(trials, (used, ρ))
            y, at_boundary = hook(Δ)
            used      = Δ
            rejected += 1
        end

        # ---- or a scan of the radius: the trial of smallest residual among those with ρ > eta ----
        if trust === :scan
            # the GMRES step, which every radius above its length gives (none if it is not finite)
            yN, _ = hook(Inf)
            ℓN    = isfinite(len(yN)) ? len(yN) : Inf
            tried = Tuple{Float64, Float64}[] # radius and ratio ρ of every trial
            best  = 0                         # the trial accepted, 0 for none yet
            Rb    = Inf                       # its residual
            δσb   = 0.0                       # its change of a free parameter

            while best == 0
                for r in unique(min.(Δ .* SCAN, Δmax, ℓN))
                    # the hookstep at radius r, its trial point and its ratio ρ
                    yr, at = hook(r)
                    _step!(δx, z, yr, k, U, Z, arn.Q, F, flexible)
                    xn .= x .+ δx

                    # (a free parameter changed for the trial only)
                    δσr = parameter === nothing ? 0.0 : yr[1] / nd
                    parameter === nothing || parameter.update!(δσr)
                    Rn  = objective(F, xn)
                    parameter === nothing || parameter.update!(-δσr)
                    predicted = (norm(g̃)^2 - norm(g̃ - G * yr)^2) / 2
                    q         = (R - Rn) / predicted
                    push!(tried, (r, q))

                    # the best so far: kept with its point, its step and its kind
                    if q > eta && Rn < Rb
                        best        = length(tried)
                        Rb          = Rn
                        y           = yr
                        at_boundary = at
                        xb         .= xn
                        δσb         = δσr
                    end
                end
                best > 0 && break

                # none accepted: a scan below the smallest radius tried, unless the region has collapsed
                Δ = minimum(first.(tried)) / 8
                if Δ < eps() * norm(x)
                    verbose && println(io, "  trust region collapsed: no step in the Krylov space reduces ‖r‖")
                    objective(F, x)
                    return x
                end
            end

            # ---- the accepted trial, the others, and the radius of the next iteration ----
            x       .= xb
            R        = Rb
            parameter === nothing || parameter.update!(δσb)
            ρ        = tried[best][2]
            used     = tried[best][1]
            trials   = tried[[1:best - 1; best + 1:end]]
            rejected = count(t -> !(t[2] > eta), trials)
            Δ        = len(y)
        end

        # ---- the directions recycled into the next system: the right singular vectors of G of the
        #      smallest singular values, the slow directions of this one ----
        recycle > 0 && (U = _recycled(SH, min(recycle, size(G, 2)), k, U, Z, arn.Q, F, flexible, x))

        # the accepted step: the Newton (GMRES) step inside the trust region, or a hookstep on
        # its boundary
        step = at_boundary ? "hook" : "newton"

        verbose && _print_newton_row(io, iter, R, frequency(x), Δ, ρ, size(H, 2), krylov[end], step, rejected)
        verbose && parameter !== nothing && @printf(io, "        parameter %.10g\n", parameter.value())
        stop = callback(_info(iter, x, R, F, krylov, step, (radius=used, length=len(y), ratio=ρ, rejected, trials);
                              parameter = parameter === nothing ? NaN : parameter.value()))
    end

    # ---- leave the residual of x in the cache ----
    objective(F, x)

    return x
end


# Residual of the least-squares problem min ‖g - H y‖ after a new column h of the upper Hessenberg
# matrix H: the previous Givens rotations are applied to h, a new one zeroes its last entry and
# rotates the right-hand side γ, whose last entry is the residual. O(k) operations per column,
# against O(k³) for solving the problem again.
function _givens!(qr, h::AbstractVector)
    c, s, γ = qr
    k       = length(h) - 1

    # ---- previous rotations ----
    for j in 1:k - 1
        h[j], h[j + 1] = c[j] * h[j] + s[j] * h[j + 1], -s[j] * h[j] + c[j] * h[j + 1]
    end

    # ---- new rotation, and the rotated right-hand side ----
    ρ = hypot(h[k], h[k + 1])
    push!(c, ρ > 0 ? h[k] / ρ : 1.0)
    push!(s, ρ > 0 ? h[k + 1] / ρ : 0.0)
    push!(γ, -s[k] * γ[k])
    γ[k] *= c[k]

    return abs(γ[k + 1])
end


# ---------------------------------------------------------------------------- #
# trust-region step in the Krylov space                                        #
# ---------------------------------------------------------------------------- #
# Solve (3), min ‖g - H y‖ with ‖y‖ ≤ Δ, as in the hookstep of GMRES.jl.
#
# With the SVD H = U D Vᵀ and p = Uᵀg, the problem decouples along the singular directions. Its
# solution is
#
#     y(μ) = V (p d / (d² + μ)),     ‖y(μ)‖ = ‖p d / (d² + μ)‖,
#
# where μ ≥ 0 is the Lagrange multiplier of the constraint: μ = 0 gives the least-squares (GMRES)
# solution V (p / d), and larger μ damp the directions of small singular values, shortening the
# step and turning it towards steepest descent. If the least-squares solution lies inside the trust
# region it is the answer; otherwise μ is the root of ‖y(μ)‖ = Δ, which decreases monotonically in
# μ and is found by bisection on [0, ‖p‖ max(d) / Δ], an interval where ‖y(μ)‖ ≤ ‖p‖ max(d) / μ
# guarantees a sign change. Returns y and whether it lies on the boundary. The SVD of H may be given
# instead of H, to be reused for several radii in the same space.
_hookstep(H::Matrix, g::Vector, Δ::Real) = _hookstep(svd(H), g, Δ)

function _hookstep(SH::SVD, g::Vector, Δ::Real)
    U, d, V = SH
    p       = U' * g

    # ---- ‖y(μ)‖ - Δ, decreasing in μ ----
    fun(μ) = norm(p .* d ./ (d.^2 .+ μ)) - Δ

    # ---- least-squares solution, if inside the trust region ----
    fun(0.0) <= 0 && return V * (p ./ d), false

    # ---- boundary: root of fun by bisection ----
    lo = 0.0
    hi = norm(p) * maximum(d) / Δ
    for _ in 1:100
        μ = (lo + hi) / 2
        fun(μ) > 0 ? (lo = μ) : (hi = μ)
    end

    # hi keeps ‖y‖ ≤ Δ
    return V * (p .* d ./ (d.^2 .+ hi)), true
end


# M z, orthonormalised against the vectors before it, into W, and its coordinates on them, a column of R, into G,
# with metric=:operator; nothing otherwise. Gram-Schmidt is applied twice, so that R keeps the precision of the
# vectors when their lengths spread over orders of magnitude, as they do when a coarse level solves a few slow
# directions exactly: the Gram matrix of the vectors would square that spread, and its small eigenvalues, those of
# the directions of the fine level, would fall below rounding
function _metric!(W::Vector{X}, G::Vector{Vector{Float64}}, F::System, z::X, metric::Symbol) where {X}
    metric === :operator || return W

    # ---- M z, then its components along the vectors before it, removed twice ----
    w = metric!(similar(z), F.B, z)
    c = zeros(length(W) + 1)
    for _ in 1:2
        for (i, wi) in enumerate(W)
            h     = dot(wi, w)
            w   .-= h .* wi
            c[i] += h
        end
    end

    # ---- the length of the rest, and the rest normalised ----
    c[end] = norm(w)
    c[end] > 0 && (w ./= c[end])

    push!(W, w)
    push!(G, c)

    return W
end

# R and R⁻¹ for the first m vectors M zⱼ = Wₙ R eⱼ, from the SVD of R with its columns scaled to unit length, without
# the directions of singular values below ϵ times the largest, which rounding sets: there the vectors depend on each
# other, the hookstep would take long steps for nothing, and R⁻¹ maps the coordinates u of the kept directions back
# to y. Scaling the columns first keeps the short vectors, those of the fine level, beside the long ones, those that
# the coarse level stretches. The identity with metric=:krylov
function _root(G::Vector{Vector{Float64}}, m::Int, metric::Symbol, ϵ::Real)
    metric === :operator || return I, I

    # ---- R, and the lengths of its columns, the lengths of the vectors M zⱼ ----
    R = [i <= j ? G[j][i] : 0.0 for i in 1:m, j in 1:m]
    D = [max(norm(view(R, :, j)), floatmin()) for j in 1:m]

    # ---- the SVD of R D⁻¹, without the directions that rounding sets ----
    _, s, V = svd(R ./ D')
    keep    = s .> ϵ * maximum(s)
    s       = s[keep]
    V       = V[:, keep]

    # ‖R y‖ = ‖diag(s) Vᵀ D y‖
    return s .* V' .* D', (V ./ s') ./ D
end


# ---------------------------------------------------------------------------- #
# recycled directions                                                          #
# ---------------------------------------------------------------------------- #
# Krylov recycling between Newton systems, in the manner of GCRO-DR (Parks et al. 2006): the
# directions U of the step carried from the last system are mapped by the new Jacobian, C = 𝒥 U,
# C is orthonormalised with the same combinations applied to U, and the Krylov space is built for
# the projected operator (I - C Cᵀ) 𝒥 B⁻¹ from r₀ = (I - C Cᵀ) b. With E = Cᵀ 𝒥 Zₙ,
#
#     𝒥 [U Zₙ] = [C Qₙ₊₁] G,     G = [I E; 0 H],     b = [C Qₙ₊₁] (Cᵀ b, β₀ e₁),
#
# and the hookstep is solved in the coordinates of [U Zₙ]. The directions kept for the next system
# are the right singular vectors of G of the smallest singular values, those along which the
# restricted Jacobian is nearly singular and on which GMRES spends its plateau. They come with the
# SVD that the hookstep needs anyway; harmonic Ritz vectors, the classical choice, would need the
# inner products of every Arnoldi vector with every preconditioned one in flexible GMRES.

# 𝒥 U = C R at the current point, C orthonormal and R upper triangular, by Gram-Schmidt on the images; U is
# kept as it was recycled, orthonormal in the metric of the trust region, so that its coordinates and those of
# the Krylov space are measured alike (normalising U by its images instead, ‖𝒥u‖ = 1, makes the slow directions,
# on which 𝒥 is small, long steps of unit coordinate, which the trust region no longer bounds); a direction whose
# image depends on the others is dropped
function _recycle!(U::Vector{X}, F::System, x::X) where {X}
    C    = X[]
    kept = X[]
    cols = Vector{Float64}[]
    for u in U
        c  = jacobian!(similar(x), F, x, u)
        n₀ = norm(c)
        r  = zeros(length(U))
        for _ in 1:2, (j, cj) in enumerate(C)
            h     = dot(cj, c)
            c   .-= h .* cj
            r[j] += h
        end
        n = norm(c)
        n > 1e-8 * n₀ || continue
        c ./= n
        r[length(C) + 1] = n
        push!(C, c)
        push!(kept, u)
        push!(cols, r)
    end
    empty!(U)
    append!(U, kept)
    k = length(C)
    R = k == 0 ? zeros(0, 0) : reduce(hcat, [col[1:k] for col in cols])
    return C, R
end

# out off the span of the orthonormal C, in place, with its coefficients pushed to E
function _deflate!(out::X, C::Vector{X}, E::Vector{Vector{Float64}}) where {X}
    e = zeros(length(C))
    for _ in 1:2, (i, c) in enumerate(C)
        h     = dot(c, out)
        out .-= h .* c
        e[i] += h
    end
    push!(E, e)
    return out
end

# the step U a + Zₙ y from the coordinates c = (a, y), into δx; z is a work vector
function _step!(δx::X, z::X, c::Vector, k::Int, U::Vector{X}, Z::Vector{X}, Q::Vector{X},
                F::System, flexible::Bool) where {X}
    y = c[k + 1:end]
    if isempty(y)
        δx .= 0
    elseif flexible
        lincomb!(δx, Z, y)
    else
        lincomb!(z, Q, y)
        _precondition!(δx, F, z)
    end
    # (the recycled directions take the last length(U) of the first k coordinates; a free parameter,
    #  with no direction of its own, the first)
    for (i, u) in enumerate(U)
        δx .+= c[k - length(U) + i] .* u
    end
    return δx
end

# the kk directions of the step along the right singular vectors of G of the smallest singular
# values: [U Zₙ] v for each such v, with Zₙ = B⁻¹ Qₙ formed once per direction without flexible GMRES
function _recycled(SH::SVD, kk::Int, k::Int, U::Vector{X}, Z::Vector{X}, Q::Vector{X}, F::System,
                   flexible::Bool, x::X) where {X}
    V = SH.V[:, end - kk + 1:end]
    z = similar(x)
    return [_step!(similar(x), z, V[:, i], k, U, Z, Q, F, flexible) for i in 1:kk]
end


# ---------------------------------------------------------------------------- #
# output                                                                       #
# ---------------------------------------------------------------------------- #
# radius: trust-region radius after the update; krylov: dimension of the Krylov space; linear:
# relative residual of the GMRES solution of the Newton system; step: newton, the full step inside the trust
# region, or hook, on its boundary; rejected: trial steps rejected before it, each shrinking the radius
_print_newton_header(io::IO) =
    println(io, "  iter      ‖r‖          ω          radius       ρ      krylov    linear    step   rejected")

function _print_newton_row(io::IO, iter, R, ω, Δ, ρ, k, lin, step, rejected)
    @printf(io, "%6d  %.4e  %.6e  %.3e  %8.3f  %6d  %.2e  %6s  %8s\n",
            iter, sqrt(2R), ω, Δ, ρ, k, lin, step, iter == 0 ? "" : string(rejected))
end
