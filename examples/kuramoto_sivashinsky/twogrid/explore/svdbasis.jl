# Other spatial bases for the coarse space, against the sine modes: GMRES iterations against the
# number of modes r, with t_c = 2 and V(2,2). The bases, all global (the same at every slice):
#
#     sine      the first r sine modes
#     svd1      the SVD modes of the slow directions of the orbit M = 1 (modes.py)
#     svd8      the same for the orbit M = 8
#     smoother  the SVD modes of test vectors: random fields smoothed ν = 4 times with
#               z ← z - B⁻¹ J z at the starting point of the search, as in adaptive and bootstrap
#               multigrid; written to bases/smoother_<case>_<q>.txt
#
# Writes results/svdbasis.txt (columns as in study.jl, method = basis). r from 6 at T ≈ 200, from 12
# at T ≈ 1500 (below, the searches fail and each takes minutes).
#
#     julia --project=.. svdbasis.jl [case:q …]       (all cases, or those listed)
include(joinpath(@__DIR__, "..", "common.jl"))
import ReSolver: jacobian!, _precondition!

# the left singular vectors of the slices of nv smoothed random fields, in sine coordinates
function smoother_basis(x::Orbit, g::KSGrid; ν::Int=4, nv::Int=8, seed::Int=1)
    F = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!,
               B=KSPreconditioner(x; kind=:jacobian), project=odd!)
    F.linearise!(F.lin, x.a)
    sc, _ = odd_sine_coordinates(g)
    rng = MersenneTwister(seed)
    z, Jz, BJz = similar(x), similar(x), similar(x)
    cols = Matrix{Float64}[]
    for _ in 1:nv
        z.a.data .= randn(rng, size(z.a.data)); odd!(z.a); z.p .= 0
        for _ in 1:ν
            jacobian!(Jz, F, x, z); _precondition!(BJz, F, Jz)
            z .-= BJz; z.p .= 0; odd!(z.a); z ./= norm(z)
        end
        push!(cols, sc(z.a))
    end
    return svd(reduce(hcat, cols)).U
end

basis(M) = readdlm(joinpath(@__DIR__, "global_basis_$M.txt"))
const CASES = (("repeat1", 8, () -> repeated_start(1, 8), 6:2:16), ("prime", 8, () -> prime_case(8), 6:2:16),
               ("repeat8", 8, () -> repeated_start(8, 8), 12:2:16), ("repeat1", 64, () -> repeated_start(1, 64), 12:2:16))
mkpath(joinpath(@__DIR__, "bases"))

x, g = prime_case(1)                                      # compilation, not recorded
for G in (Matrix(1.0I, 32, 32), basis(1), smoother_basis(x, g))
    run_newton(x, g; r=8, tc=2, pre=2, post=2, coordinates=basis_coordinates(g, G))
end

open(joinpath(HERE, "results", "svdbasis.txt"), "a") do io
    for (case, q, make, rs) in CASES
        isempty(ARGS) || "$case:$q" in ARGS || continue
        x, g = make()
        Gs = smoother_basis(x, g)
        writedlm(joinpath(@__DIR__, "bases", "smoother_$(case)_$q.txt"), Gs)
        bases = (("sine", odd_sine_coordinates(g)), ("svd1", basis_coordinates(g, basis(1))),
                 ("svd8", basis_coordinates(g, basis(8))), ("smoother", basis_coordinates(g, Gs)))
        for r in rs, (name, coordinates) in bases
            o = run_newton(x, g; r, tc=2, pre=2, post=2, maxiter=8, coordinates)
            record(io, case, q, name, o; r, tc=2, pre=2, post=2, h=4)
        end
    end
end
