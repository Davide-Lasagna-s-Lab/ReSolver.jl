using Test
using LinearAlgebra
using Random
using ReSolver

import ReSolver: objective, gradient!, jacobian!, jacobian_adjoint!, residual!

# Kuramoto–Sivashinsky and Kolmogorov flow models of the examples, which depend only on FFTW
include("../examples/kuramoto_sivashinsky/ks.jl")
include("../examples/kolmogorov/kolmogorov.jl")

Random.seed!(1)


# ---- problem: KS on L = 22, with a drift along x ----
g   = KSGrid(22, 15, 15)
nl  = KSNonlinear(g)
lin = KSLinearised(g)
adj = KSLinearised(g; adjoint=true)

# a random field of unit amplitude, and a random orbit with frequency ω and drift speed c, whose
# field has zero mean, as in the searches, which remove the mean with demean!
random_field() = KSField(g, randn(g.Nx, g.Ns))
random_orbit(ω, c...) = Orbit(demean!(random_field()), [log(ω), c...])

x = random_orbit(0.4, 0.2)
F = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), project=demean!)


@testset "KS model                                                " begin
    u, v, w = random_field(), random_field(), random_field()

    # ---- fields keep their type under broadcasting ----
    @test u .+ 2 .* v isa KSField

    # ---- spectral derivatives are skew-adjoint ----
    @test dot(ddx!(similar(u), u), v) ≈ -dot(u, ddx!(similar(v), v))
    @test dot(dds!(similar(u), u), v) ≈ -dot(u, dds!(similar(v), v))

    # ---- linearised = central difference of nonlinear, exact for a quadratic operator ----
    linearise!(lin, u)
    linearise!(adj, u)

    ε  = 1e-3
    FD = (nl(similar(u), u .+ ε .* v) .- nl(similar(u), u .- ε .* v)) ./ 2ε
    Lv = lin(similar(u), v)

    @test norm(FD .- Lv) / norm(Lv) < 1e-8

    # ---- the adjoint is exact ----
    @test dot(w, Lv) ≈ dot(adj(similar(w), w), v)
end


@testset "Orbit                                                   " begin
    y = copy(x)

    @test norm(y .- x) == 0
    @test (2 .* y).p == 2 .* x.p
    @test dot(x, y) ≈ dot(x.a, x.a) + sum(abs2, x.p)
    @test ReSolver.frequency(x) ≈ 0.4
    @test_throws BoundsError ReSolver.drift(x, 0)
end


@testset "Gradient                                                " begin
    g₁ = similar(x)
    R  = gradient!(g₁, F, x)

    @test R ≈ objective(F, x)

    # ---- directional derivative against central differences ----
    δ  = random_orbit(0.7, -0.3)
    ε  = 1e-6
    FD = (objective(F, x .+ ε .* δ) - objective(F, x .- ε .* δ)) / 2ε

    @test dot(g₁, δ) ≈ FD rtol=1e-6
end


@testset "Jacobian                                                " begin
    δ = random_orbit(0.7, -0.3)

    F.linearise!(F.lin, x.a)
    Jδ = jacobian!(similar(x), F, x, δ)

    # ---- residual part against central differences ----
    ε  = 1e-6
    rp = residual!(similar(x), F, x .+ ε .* δ)
    rm = residual!(similar(x), F, x .- ε .* δ)

    @test norm(Jδ.a .- (rp.a .- rm.a) ./ 2ε) / norm(Jδ.a) < 1e-6

    # ---- phase conditions ----
    @test Jδ.p[1] ≈ dot(dds!(similar(x.a), x.a), δ.a)
    @test Jδ.p[2] ≈ dot(ddx!(similar(x.a), x.a), δ.a)
end


@testset "Jacobian-free system                                    " begin
    Ff = System(nl, dds!, x; ddi=(ddx!,), project=demean!)
    δ  = random_orbit(0.7, -0.3)

    # ---- the finite-difference Jacobian action is close to the exact one ----
    F.linearise!(F.lin, x.a)
    Ff.linearise!(Ff.lin, x.a)
    Jδ  = jacobian!(similar(x), F, x, δ)
    Jfδ = jacobian!(similar(x), Ff, x, δ)
    d   = similar(x)
    d  .= Jδ .- Jfδ

    @test norm(d) / norm(Jδ) < 1e-6

    # ---- central differences are exact up to rounding for the quadratic operator of KS ----
    Fc = System(nl, dds!, x; difference=:central, ddi=(ddx!,), project=demean!)
    Fc.linearise!(Fc.lin, x.a)
    d .= Jδ .- jacobian!(similar(x), Fc, x, δ)

    @test norm(d) / norm(Jδ) < 1e-9
    @test_throws ArgumentError System(nl, dds!, x; difference=:backward, ddi=(ddx!,))

    # ---- the residual is the same ----
    @test objective(Ff, x) == objective(F, x)

    # ---- the hookstep descends, L-BFGS needs the adjoint ----
    y  = copy(x)
    R₀ = objective(Ff, y)
    solve!(y, Ff, NewtonHookstep(maxiter=3, verbose=false))

    @test objective(Ff, y) < R₀
    @test_throws ArgumentError solve!(copy(x), Ff, LBFGS(maxiter=3, verbose=false))

    # ---- with the linearised operator and no adjoint, the hookstep runs as before ----
    Fn = System(nl, lin, dds!, x; linearise!, ddi=(ddx!,), project=demean!)
    y₁ = copy(x)
    y₂ = copy(x)
    solve!(y₁, F, NewtonHookstep(maxiter=3, verbose=false))
    solve!(y₂, Fn, NewtonHookstep(maxiter=3, verbose=false))
    d .= y₁ .- y₂

    @test norm(d) == 0
    @test_throws ArgumentError solve!(copy(x), Fn, LBFGS(maxiter=3, verbose=false))
end


@testset "Preconditioner                                          " begin
    B = KSPreconditioner(x)
    p = random_orbit(0.7, -0.3)
    q = random_orbit(0.5, 0.1)

    # ---- B⁻¹ is self-adjoint and positive ----
    Bp = precondition!(similar(p), B, p)
    Bq = precondition!(similar(q), B, q)

    @test dot(Bp, q) ≈ dot(p, Bq)
    @test dot(Bp, p) > 0

    # ---- the complex :jacobian kind: B⁻⁺ is the adjoint of B⁻¹ ----
    C   = KSPreconditioner(x; kind=:jacobian)
    Cp  = precondition!(similar(p), C, p)
    Cᴴq = precondition_adjoint!(similar(q), C, q)

    @test dot(Cp, q) ≈ dot(p, Cᴴq)

    # ---- the identity default divides by one ----
    @test norm(precondition!(similar(p), I, p) .- p) == 0
end


@testset "Adjoint Jacobian                                        " begin
    F.linearise!(F.lin, x.a)
    F.linearise!(F.adj, x.a)

    v = random_orbit(0.7, -0.3)
    w = random_orbit(0.5, 0.1)

    # ---- ⟨w, 𝒥 v⟩ = ⟨𝒥⁺ w, v⟩, phase conditions and drift included ----
    Jv  = jacobian!(similar(x), F, x, v)
    Jᵀw = jacobian_adjoint!(similar(x), F, x, w)

    @test dot(w, Jv) ≈ dot(Jᵀw, v)

    # ---- the gradient of ½‖r‖² is 𝒥⁺ (r, 0) ----
    g = similar(x)
    gradient!(g, F, x)
    r = residual!(similar(x), F, x)

    @test norm(jacobian_adjoint!(similar(x), F, x, r) .- g) / norm(g) < 1e-12
end


@testset "Zero-mean subspace                                      " begin
    # ---- demean! is an orthogonal projection ----
    u  = random_field()
    v  = random_field()
    Pu = demean!(copy(u))

    @test abs(sum(Pu)) / length(Pu) < 1e-14
    @test norm(demean!(copy(Pu)) .- Pu) < 1e-14
    @test dot(Pu, v) ≈ dot(u, demean!(copy(v)))

    # ---- the residual of an orbit of zero mean has zero mean, without the projection ----
    y  = random_orbit(0.4, 0.2)
    Fy = System(nl, lin, adj, dds!, y; linearise!, ddi=(ddx!,))
    r  = residual!(similar(y), Fy, y)

    @test abs(sum(r.a)) / length(r.a) < 1e-12 * norm(r.a)
end


@testset "Symmetric subspace                                      " begin
    # ---- odd! is an orthogonal projection ----
    u  = random_field()
    v  = random_field()
    Pu = odd!(copy(u))

    @test norm(odd!(copy(Pu)) .- Pu) == 0
    @test dot(Pu, v) ≈ dot(u, odd!(copy(v)))

    # ---- the KS operators preserve the odd fields: an odd orbit has an odd residual ----
    y  = Orbit(odd!(random_field()), [log(0.4)])
    Fy = System(nl, lin, adj, dds!, y; linearise!)
    r  = residual!(similar(y), Fy, y)

    @test norm(odd!(copy(r.a)) .- r.a) / norm(r.a) < 1e-12

    # ---- with the projection, a search from a field that is not odd stays odd ----
    Fo = System(nl, lin, adj, dds!, y; linearise!, project=odd!)
    z  = Orbit(y.a .+ 0.1 .* random_field(), copy(y.p))

    solve!(z, Fo, NewtonHookstep(maxiter=3, verbose=false))

    @test norm(odd!(copy(z.a)) .- z.a) == 0
end


@testset "Kolmogorov model                                        " begin
    gk = KFGrid(8, 9; Re=40)

    # a random real field: random coefficients, Hermitian on the column kx = 0, zero mean
    function kf_field()
        a = randn(ComplexF64, gk.n + 1, 2gk.n + 1, gk.Ns)
        for l in 1:gk.Ns, k in 1:gk.n
            a[1, 2gk.n + 2 - k, l] = conj(a[1, k + 1, l])
        end
        a[1, 1, :] .= 0
        return KFField(gk, a)
    end

    u = kf_field()
    v = kf_field()
    w = kf_field()

    # ---- derivatives are skew-adjoint in the inner product of the fields ----
    @test abs(dot(ddx!(similar(u), u), v) + dot(u, ddx!(similar(v), v))) < 1e-12
    @test abs(dot(dds!(similar(u), u), v) + dot(u, dds!(similar(v), v))) < 1e-12

    # ---- linearised = central difference of nonlinear, exact for a quadratic operator ----
    knl  = KFNonlinear(gk)
    klin = linearise!(KFLinearised(gk), u)
    kadj = linearise!(KFLinearised(gk; adjoint=true), u)

    ε  = 1e-4
    FD = (knl(similar(u), u .+ ε .* v) .- knl(similar(u), u .- ε .* v)) ./ 2ε
    Lv = klin(similar(u), v)

    @test norm(FD .- Lv) / norm(Lv) < 1e-8

    # ---- the adjoint is exact ----
    @test dot(w, Lv) ≈ dot(kadj(similar(w), w), v)

    # ---- the laminar flow, ω = -(Re/kf) cos(kf y), is a steady solution ----
    lam = KFField(gk, gk.forcing .* (gk.Re / gk.kf^2))

    @test norm(knl(similar(lam), lam)) < 1e-12

    # ---- the factorised preconditioner rebuilt about another orbit is the one built about it ----
    x₁ = Orbit(u, [log(0.7), 0.1])
    x₂ = Orbit(w, [log(0.9), -0.3])
    B₁ = KFPreconditioner(x₁; kind=:factorised, σ=2.0, inner=3)
    B₂ = KFPreconditioner(x₂; kind=:factorised, σ=2.0, inner=3)
    ReSolver.relinearise!(B₁, x₁)
    ReSolver.relinearise!(B₁, x₂)
    q  = Orbit(v, [0.3, -0.2])
    z₁ = ReSolver.precondition!(similar(q), B₁, q)
    z₂ = ReSolver.precondition!(similar(q), B₂, q)

    @test norm(z₁ .- z₂) / norm(z₂) < 1e-12

    # ---- gradient and Jacobian of the residual of a relative periodic orbit ----
    xk = Orbit(u, [log(0.7), 0.1])
    Fk = System(knl, klin, kadj, dds!, xk; linearise!, ddi=(ddx!,), B=KFPreconditioner(xk))
    δk = Orbit(v, [0.3, -0.2])
    gk₁ = similar(xk)
    gradient!(gk₁, Fk, xk)

    Rp = objective(Fk, Orbit(xk.a .+ ε .* δk.a, xk.p .+ ε .* δk.p))
    Rm = objective(Fk, Orbit(xk.a .- ε .* δk.a, xk.p .- ε .* δk.p))

    @test abs((Rp - Rm) / 2ε - dot(gk₁, δk)) / abs(dot(gk₁, δk)) < 1e-6
end


@testset "Krylov recycling                                        " begin
    # ---- with Krylov spaces that solve every Newton system exactly and every full step accepted, the
    #      steps are Newton's and do not depend on the recycled directions: the same iterates with and
    #      without them (on the zero-mean fields, where the Jacobian is nonsingular) ----
    Fd       = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), project=demean!)
    exact(k) = NewtonHookstep(maxiter=3, krylov_dim=400, krylov_tol=1e-13, Δ=1e8, Δmax=1e8, eta=-Inf,
                              recycle=k, verbose=false)
    y₀ = copy(x)
    y₁ = copy(x)
    solve!(y₀, Fd, exact(0))
    solve!(y₁, Fd, exact(8))
    d  = similar(x)
    d .= y₀ .- y₁

    @test norm(d) / norm(y₀) < 1e-8

    # ---- with the usual inexact solves, recycling still reduces the residual, in flexible GMRES and
    #      not, and keeps the search on the invariant subspace ----
    FB = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=KSPreconditioner(x), project=demean!)
    for flexible in (false, true)
        y  = copy(x)
        R₀ = objective(FB, y)
        solve!(y, FB, NewtonHookstep(; maxiter=4, krylov_dim=30, recycle=10, flexible, verbose=false))

        @test objective(FB, y) < R₀
        @test abs(sum(y.a.data)) / norm(y.a) < 1e-10
    end
end


@testset "scan of the trust radius                                " begin
    FB = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=KSPreconditioner(x), project=demean!)

    # ---- one iteration from the same orbit: the scan tries the radius of the ratio rule among others,
    #      and keeps the trial of smallest residual ----
    y₁ = copy(x)
    y₂ = copy(x)
    solve!(y₁, FB, NewtonHookstep(maxiter=1, krylov_dim=30, verbose=false))
    solve!(y₂, FB, NewtonHookstep(maxiter=1, krylov_dim=30, trust=:scan, verbose=false))

    @test objective(FB, y₂) <= objective(FB, y₁) * (1 + 1e-12)

    # ---- several iterations, in flexible GMRES and not: the residual falls, every accepted step has
    #      ρ > eta, and rejected counts the other trials with ρ ≤ eta ----
    for flexible in (false, true)
        y    = copy(x)
        R₀   = objective(FB, y)
        info = []
        solve!(y, FB, NewtonHookstep(; maxiter=4, krylov_dim=30, trust=:scan, flexible, verbose=false,
                                       callback=i -> (push!(info, i.trust); false)))

        @test objective(FB, y) < R₀
        @test all(t -> t.ratio > 1e-3, info[2:end])
        @test all(t -> t.rejected == count(q -> !(q[2] > 1e-3), t.trials), info[2:end])
        @test all(t -> !isempty(t.trials), info[2:end])
    end

    @test_throws ArgumentError solve!(copy(x), FB, NewtonHookstep(trust=:line, verbose=false))
end


# the KS preconditioner applied forward, the metric of the trust region with metric=:operator
function ReSolver.metric!(out::Orbit, B::KSPreconditioner, p::Orbit)
    transform!(B.û, p.a)
    B.û.data .*= B.β
    transform!(out.a, B.û)
    out.p .= p.p .* B.s
    return out
end

@testset "metric of the trust region                              " begin
    FB = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=KSPreconditioner(x), project=demean!)

    # ---- with a fixed preconditioner M B⁻¹ = I, the metric of the operator is that of the Krylov space: the same
    #      iterates, with both updates of the radius ----
    for trust in (:ratio, :scan)
        y₁ = copy(x)
        y₂ = copy(x)
        solve!(y₁, FB, NewtonHookstep(; maxiter=3, krylov_dim=30, flexible=true, trust, verbose=false))
        solve!(y₂, FB, NewtonHookstep(; maxiter=3, krylov_dim=30, flexible=true, trust, metric=:operator,
                                        verbose=false))
        d = similar(y₁)
        d .= y₁ .- y₂

        @test norm(d) / norm(y₁) < 1e-8
    end

    # ---- vectors of lengths that spread over eight orders of magnitude, as when a coarse level stretches one slow
    #      direction: the metric keeps every short vector beside the long ones, and ‖R y‖ = ‖M Z y‖ ----
    s = random_orbit(0.4, 0.2)
    Z = [random_orbit(0.4, 0.2) for _ in 1:10]
    for z in Z
        z .+= 1e8 .* s
    end

    W = typeof(x)[]
    G = Vector{Float64}[]
    for z in Z
        ReSolver._metric!(W, G, FB, z, :operator)
    end
    Rm, Ri = ReSolver._root(G, 10, :operator, 100 * eps())

    y  = randn(10)
    My = similar(x)
    My .= 0
    for (yj, z) in zip(y, Z)
        My .+= yj .* ReSolver.metric!(similar(x), FB.B, z)
    end

    @test size(Rm, 1) == 10
    @test abs(norm(Rm * y) - norm(My)) / norm(My) < 1e-6
    @test norm(Rm * Ri - I) < 1e-6

    # ---- the metric of the operator needs the preconditioned vectors and the plain Krylov space ----
    @test_throws ArgumentError solve!(copy(x), FB, NewtonHookstep(metric=:operator, verbose=false))
    @test_throws ArgumentError solve!(copy(x), FB, NewtonHookstep(metric=:operator, flexible=true, recycle=5,
                                                                  verbose=false))
    @test_throws ArgumentError solve!(copy(x), FB, NewtonHookstep(metric=:fixed, verbose=false))
end


# the identity, counting the calls of relinearise!
struct CountingPreconditioner
    calls::Base.RefValue{Int} # number of calls of relinearise!
end

ReSolver.precondition!(out::Orbit, ::CountingPreconditioner, p::Orbit)         = (out .= p)
ReSolver.precondition_adjoint!(out::Orbit, ::CountingPreconditioner, p::Orbit) = (out .= p)
ReSolver.relinearise!(B::CountingPreconditioner, x::Orbit)                     = (B.calls[] += 1; B)

@testset "solve!                                                  " begin
    FB = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=KSPreconditioner(x),
                project=demean!)

    for S in (F, FB), method in (LBFGS(maxiter=5, verbose=false),
                                 LBFGS(memory=0, maxiter=5, verbose=false),
                                 NewtonHookstep(maxiter=3, verbose=false))
        y  = copy(x)
        R₀ = objective(S, y)

        solve!(y, S, method)

        @test objective(S, y) < R₀
    end

    # ---- with a fixed preconditioner the flexible hookstep takes the same steps, with fewer B⁻¹ ----
    y₁ = copy(x)
    y₂ = copy(x)
    F₁ = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=KSPreconditioner(x),
                project=demean!)
    F₂ = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=KSPreconditioner(x),
                project=demean!)
    solve!(y₁, F₁, NewtonHookstep(maxiter=3, verbose=false))
    solve!(y₂, F₂, NewtonHookstep(maxiter=3, verbose=false, flexible=true))
    d = similar(y₁)
    d .= y₁ .- y₂

    @test norm(d) / norm(y₁) < 1e-8
    @test F₂.evaluations.precondition < F₁.evaluations.precondition

    # ---- the preconditioner is rebuilt at every Newton iteration, unless relinearise=false ----
    for relinearise in (true, false)
        B     = CountingPreconditioner(Ref(0))
        trace = Trace()
        solve!(copy(x), System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B, project=demean!),
               NewtonHookstep(; maxiter=3, verbose=false, relinearise, callback=trace))

        @test B.calls[] == (relinearise ? trace.iter[end] : 0)
    end

    # ---- the callback records the residual and can stop the search ----
    history = Float64[]
    solve!(copy(x), F, LBFGS(maxiter=50, verbose=false,
                             callback=info -> (push!(history, info.res); info.iter == 3)))

    @test length(history) == 4
    @test issorted(history; rev=true)

    # ---- a trace follows two calls ----
    trace = Trace()
    y     = copy(x)
    solve!(y, FB, LBFGS(maxiter=4, verbose=false, callback=trace))
    solve!(y, FB, NewtonHookstep(maxiter=2, verbose=false, callback=trace))

    @test trace.iter == [0, 1, 2, 3, 4, 0, 1, 2]
    @test trace.res[end] ≈ sqrt(2 * objective(FB, y))
    @test issorted(trace.time)

    # ---- counters grow; Krylov histories only for the hookstep iterations ----
    @test issorted([e.residual for e in trace.evaluations])
    @test trace.evaluations[end].jacobian > 0
    @test trace.evaluations[end].precondition > 0
    @test all(isempty, trace.krylov[1:6]) && !any(isempty, trace.krylov[7:8])

    # ---- kinds of step only for the hookstep iterations ----
    @test all(isempty, trace.step[1:6]) && all(in(("newton", "hook")), trace.step[7:8])
end
