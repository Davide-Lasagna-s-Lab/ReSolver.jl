include(joinpath(@__DIR__, "..", "common.jl"))
import ReSolver: residual!
for q in (1, 16)
    x, g = repeated_start(1, q; ε=0.0)
    F = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, project=odd!)
    @printf "q=%d unperturbed ‖r‖ = %.2e\n" q norm(residual!(similar(x), F, x))
end
x, g = repeated_start(1, 16)
println("q=16 two-grid r=12 tc=1 ", fmt(run_newton(x, g; r=12)))
println("q=16 two-grid r=12 tc=4 ", fmt(run_newton(x, g; r=12, tc=4)))
println("q=16 plain              ", fmt(run_newton(x, g; kind=:plain, maxiter=1)))
x, g = repeated_start(1, 64)
println("q=64 two-grid r=12 tc=1 ", fmt(run_newton(x, g; r=12)))
println("q=64 two-grid r=12 tc=4 ", fmt(run_newton(x, g; r=12, tc=4)))
