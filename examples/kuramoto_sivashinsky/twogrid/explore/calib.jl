include(joinpath(@__DIR__, "..", "common.jl"))
import ReSolver: residual!
function res0(x, g)
    F = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, project=odd!)
    r = residual!(similar(x), F, x); return norm(r)
end
for q in (1, 4)
    x, g = repeated_start(1, q; ε=0.0)
    @printf "q=%d unperturbed: T=%.2f Ns=%d ‖r‖=%.2e\n" q 2π/frequency(x) g.Ns res0(x, g)
    for ε in (0.005, 0.02, 0.05)
        x, g = repeated_start(1, q; ε)
        @printf "  ε=%.3f ‖r₀‖=%.2e\n" ε res0(x, g)
        println("    plain    ", fmt(run_newton(x, g; kind=:plain)))
        println("    two-grid ", fmt(run_newton(x, g; r=12)))
    end
end
for M in (2, 4, 8)
    x, g = prime_case(M); @printf "prime start M=%d: ‖r₀‖=%.2e\n" M res0(x, g)
end
