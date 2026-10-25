include(joinpath(@__DIR__, "..", "common.jl"))
x, g = repeated_start(1, 64)
for (r, tc, pre, post) in ((16, 1, 1, 1), (12, 2, 2, 2), (16, 2, 1, 1), (16, 2, 2, 2), (16, 4, 1, 1), (16, 4, 2, 2), (20, 4, 1, 1), (12, 4, 3, 3))
    o = run_newton(x, g; r, tc, pre, post)
    println("q=64 r=$r tc=$tc V($pre,$post) ", fmt(o))
end
