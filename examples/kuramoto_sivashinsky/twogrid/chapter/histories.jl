# GMRES residual histories and Newton residuals, for the chapter: the diagonal preconditioner B alone
# (first Newton step, T ≤ 406) and the two-grid preconditioner (whole searches, T ≤ 1624). Writes
# results/histories.txt, one line per Newton step: case q T method r tc pre post step ‖F‖ before the
# step, then the relative GMRES residuals of the step (comma separated).
#
#     julia --project=.. histories.jl
include(joinpath(@__DIR__, "..", "common.jl"))

function write_history(io, case, q, method, o; r=0, tc=0, pre=0, post=0)
    for (i, k) in enumerate(o.krylov)
        @printf(io, "%s %d %.4f %s %d %d %d %d %d %.3e %s\n", case, q, o.T, method, r, tc, pre, post, i,
                o.newton[i], join((@sprintf("%.3e", v) for v in k), ","))
    end
    @printf(io, "%s %d %.4f %s %d %d %d %d %d %.3e %s\n", case, q, o.T, method, r, tc, pre, post,
            length(o.krylov) + 1, o.newton[end], "")
    flush(io)
    println(rpad("$case q=$q $method r=$r tc=$tc V($pre,$post)", 40), fmt(o)); flush(stdout)
end

x, g = prime_case(1)                                       # compilation
run_newton(x, g; kind=:plain, maxiter=1); run_newton(x, g; r=16, tc=2, pre=2, post=2)

open(joinpath(HERE, "results", "histories.txt"), "w") do io
    cases = [[("prime", M, () -> prime_case(M)) for M in (1, 2, 4, 8)];
             [("repeat1", q, () -> repeated_start(1, q)) for q in (1, 2, 4, 8, 16, 32, 64)]]
    for (case, q, make) in cases
        x, g = make()
        T = 2π / frequency(x)
        T < 450 && write_history(io, case, q, "plain", run_newton(x, g; kind=:plain, maxiter=1))
        write_history(io, case, q, "twogrid", run_newton(x, g; r=16, tc=2, pre=2, post=2); r=16, tc=2, pre=2, post=2)
        case == "repeat1" && q == 8 || continue
        for (r, tc, pre) in ((16, 1, 1), (12, 2, 2), (8, 2, 2), (8, 2, 1))
            write_history(io, case, q, "twogrid", run_newton(x, g; r, tc, pre, post=pre); r, tc, pre, post=pre)
        end
    end
end
