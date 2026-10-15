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

# a random field of unit amplitude, with frequency ω and drift speed c
random_field() = KSField(g, randn(g.Nx, g.Ns))
random_orbit(ω, c...) = Orbit(random_field(), [log(ω), c...])

x = random_orbit(0.4, 0.2)
F = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,))


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


@testset "solve!                                                  " begin
    FB = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=KSPreconditioner(x))

    for S in (F, FB), method in (LBFGS(maxiter=5, verbose=false),
                                 NewtonHookstep(maxiter=3, verbose=false))
        y  = copy(x)
        R₀ = objective(S, y)

        solve!(y, S, method)

        @test objective(S, y) < R₀
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
