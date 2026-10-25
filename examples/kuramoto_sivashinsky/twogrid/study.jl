# The two-grid study on Kuramoto–Sivashinsky: one line per Newton search in results/<part>.txt.
#
#     julia --project=. study.jl T        cost against the period, B only and two-grid
#                                         (Ttwogrid, into T_twogrid.txt, or Tplain for one part;
#                                         Tplain repeat1:64 … runs only the cases listed)
#     julia --project=. study.jl r        truncation level in space (r) and in time (tc)
#     julia --project=. study.jl cycle    cycles and time derivative of the coarse operator
#
# Columns: case q T Ns method r tc pre post h steps actions jacobian builds seconds krylov_MB
# coarse_MB coarse_unknowns final_residual GMRES_iterations_per_step(comma separated)
include("common.jl")

const OUT = joinpath(HERE, "results")
mkpath(OUT)

# the test cases: prime orbits from their starting points, and orbits traversed q times from a
# perturbed start
cases(which) = which === :prime    ? [("prime", M, () -> prime_case(M)) for M in (1, 2, 4, 8)] :
               which === :repeat1  ? [("repeat1", q, () -> repeated_start(1, q)) for q in (1, 2, 4, 8, 16, 32, 64)] :
                                     [("repeat8", q, () -> repeated_start(8, q)) for q in (2, 4, 8)]

part = isempty(ARGS) ? "T" : ARGS[1]

# the configurations of the two-grid preconditioner compared against the period
const CONFIGS = ((12, 1, 1, 1), (16, 1, 1, 1), (16, 2, 2, 2), (16, 4, 2, 2))   # r, tc, pre, post

# one search of every kind on the shortest orbit, so that compilation is not timed
function warmup()
    x, g = prime_case(1)
    run_newton(x, g; kind=:plain)
    for (r, tc, pre, post) in CONFIGS
        run_newton(x, g; r, tc, pre, post)
    end
end

if part in ("T", "Ttwogrid")
    # ---- two-grid: every case ----
    warmup()
    open(joinpath(OUT, part == "T" ? "T_scan.txt" : "T_twogrid.txt"), "a") do io
        for which in (:prime, :repeat1, :repeat8), (case, q, make) in cases(which)
            x, g = make()
            for (r, tc, pre, post) in CONFIGS
                o = run_newton(x, g; r, tc, pre, post)
                record(io, case, q, "twogrid", o; r, tc, pre, post, h=4)
            end
        end
    end
end
if part in ("T", "Tplain")
    # ---- B only: full Newton up to T ≈ 450, one Newton step up to T ≈ 800, nothing beyond (one
    #      Newton step at T ≈ 1500 runs for more than an hour); the cases "case:q" given after
    #      Tplain, or all ----
    only = ARGS[2:end]
    part == "Tplain" && warmup()
    open(joinpath(OUT, "T_scan.txt"), "a") do io
        for which in (:prime, :repeat1, :repeat8), (case, q, make) in cases(which)
            isempty(only) || "$case:$q" in only || continue
            x, g = make()
            T = 2π / frequency(x)
            T > 900 && continue
            o = run_newton(x, g; kind=:plain, maxiter=T < 450 ? 15 : 1)
            record(io, case, q, T < 450 ? "plain" : "plain1", o)
        end
    end
end
if part == "r"
    open(joinpath(OUT, "r_scan.txt"), "a") do io
        for (case, q, make) in (("repeat1", 1, () -> repeated_start(1, 1)),
                                ("repeat1", 8, () -> repeated_start(1, 8)),
                                ("repeat1", 64, () -> repeated_start(1, 64)))
            x, g = make()
            for ν in (1, 2), tc in (1, 2, 4, 8), r in (4, 8, 12, 14, 16, 20, 24)
                o = run_newton(x, g; r, tc, pre=ν, post=ν, maxiter=8)
                record(io, case, q, "twogrid", o; r, tc, pre=ν, post=ν, h=4)
            end
        end
    end
elseif part == "cycle"
    open(joinpath(OUT, "cycle_scan.txt"), "a") do io
        x, g = repeated_start(1, 8)
        r = 16
        for (cycle, tc, pre, post, h) in ((:additive, 1, 0, 0, 4), (:multiplicative, 1, 1, 0, 4),
                                          (:multiplicative, 1, 0, 1, 4), (:multiplicative, 1, 1, 1, 4),
                                          (:multiplicative, 1, 2, 2, 4), (:multiplicative, 1, 1, 1, 1),
                                          (:multiplicative, 1, 1, 1, 2), (:multiplicative, 1, 1, 1, 8),
                                          (:multiplicative, 1, 1, 0, 1), (:multiplicative, 1, 1, 0, 8),
                                          (:multiplicative, 2, 1, 0, 4), (:multiplicative, 2, 1, 1, 4),
                                          (:multiplicative, 2, 2, 2, 4))
            o = run_newton(x, g; r, tc, cycle, pre, post, h, maxiter=12)
            record(io, "repeat1", 8, string(cycle), o; r, tc, pre, post, h)
        end
    end
end
