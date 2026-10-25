include(joinpath(@__DIR__, "..", "common.jl"))
for M in (1, 4)
    x, g = prime_case(M)
    println("==== prime orbit M = $M")
    println("plain        ", fmt(run_newton(x, g; kind=:plain)))
    for (tc, solver) in ((1, :sparse), (1, :banded), (2, :banded), (4, :banded))
        println("two-grid r=12 tc=$tc $solver ", fmt(run_newton(x, g; r=12, tc, solver)))
    end
end
