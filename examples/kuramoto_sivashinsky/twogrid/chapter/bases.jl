# Spatial bases of the coarse space from the fluid-mechanics toolbox, against the sine modes: GMRES
# iterations against the number of modes r, t_c = 2, V(2,2). With POD modes the coarse operator is
# the space-time Newton matrix of the linearised POD-Galerkin model about the orbit; with resolvent
# modes it holds the response of the linear resolvent to the advection, the amplified part of the
# smoother. The bases, in sine coordinates (32 x 32, by column):
#
#     sine   the first r sine modes
#     podA   POD of a long chaotic trajectory (dns.py, pod/pod_attractor.txt)
#     resS   response modes of the space-time smoother S = B⁻¹A at the orbit T ≈ 25 (resolvent.py)
#     resF   response modes of the frozen-time smoother (iω - L0)⁻¹A(s_l), T ≈ 25 (resolvent.py)
#     resM   response modes of the resolvent about the mean of the orbit T ≈ 25 (resolvent.py)
#
# all computed once and used for every case. Writes results/bases.txt (columns as in study.jl,
# method = basis).
#
#     julia --project=.. bases.jl [case:q …]       (all cases, or those listed)
include(joinpath(@__DIR__, "..", "common.jl"))

basis(name) = readdlm(joinpath(@__DIR__, name == "podA" ? "pod/pod_attractor.txt" : "bases/$name.txt"))
const NAMES = ("podA", "resS", "resF", "resM")
const CASES = (("repeat1", 8, () -> repeated_start(1, 8), 6:2:16), ("prime", 8, () -> prime_case(8), 6:2:16),
               ("repeat8", 8, () -> repeated_start(8, 8), 12:2:16), ("repeat1", 64, () -> repeated_start(1, 64), 12:2:16))

x, g = prime_case(1)                                      # compilation, not recorded
run_newton(x, g; r=8, tc=2, pre=2, post=2, coordinates=basis_coordinates(g, basis("podA")))

open(joinpath(HERE, "results", "bases.txt"), "a") do io
    for (case, q, make, rs) in CASES
        isempty(ARGS) || "$case:$q" in ARGS || continue
        x, g = make()
        bases = [("sine", odd_sine_coordinates(g)); [(b, basis_coordinates(g, basis(b))) for b in NAMES]]
        for r in rs, (name, coordinates) in bases
            o = run_newton(x, g; r, tc=2, pre=2, post=2, maxiter=8, coordinates)
            record(io, case, q, name, o; r, tc=2, pre=2, post=2, h=4)
        end
    end
end
