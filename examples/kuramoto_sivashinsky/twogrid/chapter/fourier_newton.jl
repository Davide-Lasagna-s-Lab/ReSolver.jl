# Newton searches with the two-grid preconditioner, B-splines in time (time = :splines) against the
# Fourier transfers with the compact derivative (time = :fourier), r = 16, V(2,2), for t_c = 2, 3, 4
# and 6: the orbit M = 1 traversed q = 1…64 times from a perturbed start (T ≈ 25…1624), and the
# prime orbits M = 1, 2, 4, 8. Writes results/fourier_newton.txt, one line per search (see record in
# common.jl), with the method "splines" or "fourier".
#
#     julia --project=.. fourier_newton.jl
include(joinpath(@__DIR__, "..", "common.jl"))

x, g = prime_case(1)                                       # compilation
for time in (:splines, :fourier)
    run_newton(x, g; r=16, tc=2, time, pre=2, post=2)
end

open(joinpath(HERE, "results", "fourier_newton.txt"), "w") do io
    cases = [[("repeat1", q, () -> repeated_start(1, q)) for q in (1, 4, 16, 64)];
             [("prime", M, () -> prime_case(M)) for M in (1, 2, 4, 8)]]
    for (case, q, make) in cases
        x, g = make()
        for tc in (2, 3, 4, 6), time in (:splines, :fourier)
            o = run_newton(x, g; r=16, tc, time, pre=2, post=2)
            record(io, case, q, string(time), o; r=16, tc, pre=2, post=2, h=0)
            flush(io)
            println(rpad("$case q=$q $time tc=$tc", 30), fmt(o)); flush(stdout)
        end
    end
end
