# where the time of one application of the two-grid preconditioner goes, T ≈ 1600
include(joinpath(@__DIR__, "..", "common.jl"))
import ReSolver: jacobian!, precondition!, restrict, prolong!, coarse_solve, relinearise!
x, g = repeated_start(1, 64)
for (tc, pre) in ((1, 1), (2, 2), (4, 2))
    coords, field! = odd_sine_coordinates(g)
    B0 = KSPreconditioner(x; kind=:jacobian)
    B  = TwoGrid(B0, coords, field!, 16; tc, pre, post=pre, solver=:banded)
    F  = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, B, project=odd!)
    F.linearise!(F.lin, x.a)
    tb = @elapsed relinearise!(B, F, x)
    tb = @elapsed relinearise!(B, F, x)
    p, out, t = copy(x), similar(x), similar(x)
    precondition!(out, B, F, p)
    n = 5
    tJ = @elapsed for _ in 1:n; jacobian!(t, F, x, p); end
    tS = @elapsed for _ in 1:n; precondition!(t, B0, p); end
    y  = restrict(B, p)
    tR = @elapsed for _ in 1:n; restrict(B, p); end
    tP = @elapsed for _ in 1:n; prolong!(t, B, y); end
    tK = @elapsed for _ in 1:n; coarse_solve(B.K, y); end
    tM = @elapsed for _ in 1:n; precondition!(out, B, F, p); end
    @printf "tc=%d V(%d,%d): build %.2f s | per call: J %.1f ms, B⁻¹ %.1f ms, restrict %.1f ms, prolong %.1f ms, coarse solve %.1f ms, whole cycle %.1f ms\n" tc pre pre tb 1e3tJ/n 1e3tS/n 1e3tR/n 1e3tP/n 1e3tK/n 1e3tM/n
end
