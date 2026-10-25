include(joinpath(@__DIR__, "..", "common.jl"))
for q in (4, 16, 64)
    x, g = repeated_start(1, q)
    for tc in (1, 2, 4), neutral in (false, true)
        o = run_newton(x, g; r=12, tc, neutral)
        println("q=$q r=12 tc=$tc neutral=$neutral ", fmt(o))
    end
end
