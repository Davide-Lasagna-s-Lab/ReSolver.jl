include(joinpath(@__DIR__, "..", "common.jl"))
for q in (4, 16, 64)
    x, g = repeated_start(1, q)
    for r in (12, 16, 24)
        o = run_newton(x, g; r, maxiter=2)
        println("q=$q r=$r tc=1 ", fmt(o))
    end
end
