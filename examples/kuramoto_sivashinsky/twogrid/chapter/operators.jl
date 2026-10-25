# Dense operators of the two-grid preconditioner, for the exact analysis of the chapter. At the base
# orbit (T ≈ 25, Ns = 73, case "M1") and at the same orbit traversed twice (T ≈ 51, Ns = 147, "Q2"),
# on the odd basis of slowdirs.jl (point j = 2…33 of each slice and its mirror, (δ_j - δ_j′)/√2, then
# log ω), writes into chapter/dense/ (not in the repository):
#
#     <case>_J.bin, <case>_Binv.bin            the bordered Jacobian and B⁻¹ (M1 only)
#     <case>_eig_<config>.txt                  eigenvalues of the cycle E = I - M⁻¹ 𝒥 (Re, Im)
#     <case>_Minv_<config>.bin, _P_, _K_, _G_  for a few configurations (M1): the preconditioner
#                                              M⁻¹, the prolongation P, the coarse matrix K of
#                                              TwoGrid, and the exact Galerkin product Pᵀ𝒥P
#     <case>_eigJ.txt, <case>_eigJBinv.txt     eigenvalues of 𝒥 and of 𝒥B⁻¹
#
# Configurations: the spatial basis (sine modes; POD of the attractor, pod/pod_attractor.txt; POD of
# the orbit itself; SVD of the slow directions, ../explore/global_basis_1.txt; SVD of smoothed test
# vectors, ../explore/bases/; the resolvent bases of resolvent.py, bases/), r, t_c, the cycle, the
# order of the time differences and the neutral profile.
#
#     julia --project=.. operators.jl            all of the above
#     julia --project=.. operators.jl bases      only the comparison of the spatial bases, M1
#     julia --project=.. operators.jl fourier    only the Fourier restriction against the B-splines
#                                                in time, for t_c = 2…8 (M1, and Q2 at r = 12, 16)
include(joinpath(@__DIR__, "dense_common.jl"))

# ---- base orbit ----
A, ρ = prime_orbit(1)
g  = KSGrid(L, NX, size(A, 2))
x  = Orbit(KSField(g, A), [ρ])
F  = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, B=KSPreconditioner(x; kind=:jacobian), project=odd!)
F.linearise!(F.lin, x.a)
n  = 32 * g.Ns + 1
J  = dense((o, v) -> jacobian!(o, F, x, v), x, n)
Bi = dense((o, v) -> _precondition!(o, F, v), x, n)
writebin("M1_J.bin", J); writebin("M1_Binv.bin", Bi)
writeeig("M1_eigJ.txt", eigvals(J)); writeeig("M1_eigJBinv.txt", eigvals(J * Bi))
@printf "M1: T = %.2f, Ns = %d, n = %d\n" 2π / exp(ρ) g.Ns n
run = analyse("M1", x, g, J; save=["sine_r16_tc2_V22", "sine_r16_tc1_V11", "sine_r8_tc2_V22", "podA_r12_tc2_V22", "sine_r16_tc4_V22", "sine_r16_tc2_V22_fourier", "sine_r16_tc4_V22_fourier"])
if ARGS == ["bases"]
    for basis in ("podA", "test", "resS", "resS_in", "resF", "resM"), tc in (1, 2), r in (4, 6, 8, 10, 12, 14, 16, 20)
        run("$(basis)_r$(r)_tc$(tc)_V22"; basis, r, tc, pre=2, post=2)
    end
    exit()
end
if ARGS == ["fourier"]
    for time in (:splines, :fourier), tc in (2, 3, 4, 6, 8), neutral in (true, false), r in (12, 16)
        run("sine_r$(r)_tc$(tc)_V22_$(time)$(neutral ? "" : "_noneutral")"; r, tc, neutral, time)
    end
    x2, g2 = repeated_start(1, 2; ε=0.0)
    F2 = System(KSNonlinear(g2), KSLinearised(g2), dds!, x2; linearise!, B=KSPreconditioner(x2; kind=:jacobian), project=odd!)
    F2.linearise!(F2.lin, x2.a)
    J2 = dense((o, v) -> jacobian!(o, F2, x2, v), x2, 32 * g2.Ns + 1)
    run2 = analyse("Q2", x2, g2, J2)
    for time in (:splines, :fourier), tc in (2, 4), r in (12, 16)
        run2("sine_r$(r)_tc$(tc)_V22_$(time)"; r, tc, time)
    end
    exit()
end
for basis in ("sine", "podA", "podO", "svd"), tc in (1, 2, 4, 8), r in (4, 6, 8, 10, 12, 14, 16, 20), (pre, post) in ((1, 1), (2, 2))
    basis != "sine" && tc > 2 && continue
    run("$(basis)_r$(r)_tc$(tc)_V$(pre)$(post)"; basis, r, tc, pre, post)
end
for (pre, post) in ((1, 0), (0, 1), (3, 3))
    run("sine_r16_tc1_V$(pre)$(post)"; r=16, tc=1, pre, post)
end
for (pre, post) in ((1, 0), (0, 1))
    run("sine_r16_tc2_V$(pre)$(post)"; r=16, tc=2, pre, post)
end
run("sine_r16_tc1_additive"; r=16, tc=1, cycle=:additive)
for h in (1, 2, 4, 8)
    run("sine_r16_tc1_V11_h$h"; r=16, tc=1, pre=1, post=1, h)
end
for r in (12, 16)
    run("sine_r$(r)_tc2_V22_noneutral"; r, tc=2, neutral=false)
end

# ---- the same orbit traversed twice ----
x2, g2 = repeated_start(1, 2; ε=0.0)
F2 = System(KSNonlinear(g2), KSLinearised(g2), dds!, x2; linearise!, B=KSPreconditioner(x2; kind=:jacobian), project=odd!)
F2.linearise!(F2.lin, x2.a)
n2 = 32 * g2.Ns + 1
J2 = dense((o, v) -> jacobian!(o, F2, x2, v), x2, n2)
Bi2 = dense((o, v) -> _precondition!(o, F2, v), x2, n2)
writeeig("Q2_eigJ.txt", eigvals(J2)); writeeig("Q2_eigJBinv.txt", eigvals(J2 * Bi2))
Bi2 = nothing; GC.gc()
@printf "Q2: T = %.2f, Ns = %d, n = %d\n" 2π / frequency(x2) g2.Ns n2
run2 = analyse("Q2", x2, g2, J2)
for (basis, r) in (("sine", 8), ("sine", 12), ("sine", 16), ("podA", 12), ("podA", 16))
    run2("$(basis)_r$(r)_tc2_V22"; basis, r, tc=2)
end
