# The slow directions of the preconditioned Jacobian at the converged prime orbits M = 1 and 8, for
# the analysis of their spatial modes (modes.py), and the dense Jacobian at M = 1, for the Floquet
# profiles. Writes into explore/slowdirs/ (not in the repository, about 600 MB):
#
#     H_M.bin   Hessenberg matrix of 120 M right-preconditioned Arnoldi steps, (m+1) × m
#     Z_M.bin   Z = B⁻¹ V, the Arnoldi vectors in the solution space: field (Nx × Ns) and log ω
#     J_1.bin   the Jacobian at M = 1 on the odd basis (point j = 2…33 of each slice and its mirror,
#               (δ_j - δ_j′)/√2), bordered by log ω and the phase condition
#     Binv_1.bin the inverse of the diagonal preconditioner B on the same basis
#
#     julia --project=.. slowdirs.jl
include(joinpath(@__DIR__, "..", "common.jl"))
import ReSolver: jacobian!, residual!, ArnoldiIteration, step!, _precondition!

const OUTD = joinpath(@__DIR__, "slowdirs")
mkpath(OUTD)

for M in (1, 8)
    A, ρ = prime_orbit(M)
    A0, ρ0 = prime_start(M)
    g  = KSGrid(L, NX, size(A, 2))
    x  = Orbit(KSField(g, A), [ρ])
    x0 = Orbit(KSField(g, A0), [ρ0])
    ops = (KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!)
    F0 = System(ops..., x0; linearise!, project=odd!)
    b  = residual!(similar(x0), F0, x0); b .*= -1             # a Newton right-hand side
    F  = System(ops..., x; linearise!, B=KSPreconditioner(x; kind=:jacobian), project=odd!)
    F.linearise!(F.lin, x.a)
    Z  = typeof(x)[]
    A!(out, v) = (z = _precondition!(similar(x), F, v); push!(Z, z); jacobian!(out, F, x, z))
    arn = ArnoldiIteration(A!, copy(b))
    m  = 120 * M
    t0 = time()
    for _ in 1:m; step!(arn); end
    open(joinpath(OUTD, "H_$M.bin"), "w") do io; write(io, arn.H); end
    open(joinpath(OUTD, "Z_$M.bin"), "w") do io; for z in Z; write(io, z.a.data); write(io, z.p); end; end
    @printf "M = %d: T = %.2f, Ns = %d, %d preconditioned Arnoldi steps, H[m+1,m] = %.2e (%.0f s)\n" M 2π / exp(ρ) g.Ns m arn.H[end, end] time() - t0

    M == 1 || continue
    # dense Jacobian and inverse preconditioner on the odd basis
    half = 2:(NX + 1) ÷ 2
    mirror(j) = mod(NX - j + 1, NX) + 1
    nb = length(half) * g.Ns
    J  = zeros(nb + 1, nb + 1); Bi = zeros(nb + 1, nb + 1)
    e  = Orbit(KSField(g), [0.0]); out = similar(x)
    function column!(J, apply!, c)
        apply!(out, e)
        k = 0
        for l in 1:g.Ns, j in half
            k += 1
            J[k, c] = (out.a.data[j, l] - out.a.data[mirror(j), l]) / sqrt(2)
        end
        J[nb + 1, c] = out.p[1]
    end
    c = 0
    for l in 1:g.Ns, j in half
        c += 1
        fill!(e.a.data, 0); e.p .= 0
        e.a.data[j, l] = 1 / sqrt(2); e.a.data[mirror(j), l] = -1 / sqrt(2)
        column!(J, (o, v) -> jacobian!(o, F, x, v), c)
        column!(Bi, (o, v) -> _precondition!(o, F, v), c)
    end
    fill!(e.a.data, 0); e.p .= 1
    column!(J, (o, v) -> jacobian!(o, F, x, v), nb + 1)
    column!(Bi, (o, v) -> _precondition!(o, F, v), nb + 1)
    open(joinpath(OUTD, "J_1.bin"), "w") do io; write(io, J); end
    open(joinpath(OUTD, "Binv_1.bin"), "w") do io; write(io, Bi); end
    @printf "M = 1: dense Jacobian and inverse preconditioner, %d × %d\n" nb + 1 nb + 1
end
