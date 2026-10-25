# Helpers of operators.jl: dense matrices of linear maps of orbits on the odd basis (point j = 2…33
# of each slice and its mirror, (δ_j - δ_j′)/√2, then log ω), and the dense analysis of one
# configuration of TwoGrid.
include(joinpath(@__DIR__, "..", "common.jl"))
import ReSolver: jacobian!, _precondition!, precondition!, relinearise!, restrict, prolong!, coarse_solve

const OUTD = joinpath(@__DIR__, "dense")
mkpath(OUTD)
const HALF = 2:(NX + 1) ÷ 2
mirror(j) = mod(NX - j + 1, NX) + 1

function to_orbit!(o::Orbit, v::AbstractVector{Float64})
    fill!(o.a.data, 0)
    k = 0
    for l in axes(o.a.data, 2), j in HALF
        k += 1
        o.a.data[j, l] = v[k] / sqrt(2); o.a.data[mirror(j), l] = -v[k] / sqrt(2)
    end
    o.p[1] = v[end]
    return o
end

function to_vector!(v::AbstractVector{Float64}, o::Orbit)
    k = 0
    for l in axes(o.a.data, 2), j in HALF
        k += 1
        v[k] = (o.a.data[j, l] - o.a.data[mirror(j), l]) / sqrt(2)
    end
    v[end] = o.p[1]
    return v
end

# the matrix of a linear map of orbits, on the odd basis
function dense(apply!, x::Orbit, n::Int)
    M = zeros(n, n); e = similar(x); out = similar(x); ev = zeros(n)
    for c in 1:n
        fill!(ev, 0); ev[c] = 1
        apply!(out, to_orbit!(e, ev))
        to_vector!(view(M, :, c), out)
    end
    return M
end

writebin(name, A) = open(io -> write(io, A), joinpath(OUTD, name), "w")
writeeig(name, λ) = writedlm(joinpath(OUTD, name), [real.(λ) imag.(λ)])

# orthonormal POD basis of the snapshots of a field, in sine coordinates, by energy
function pod_orbit(g::KSGrid, a::KSField)
    sc, _ = odd_sine_coordinates(g)
    U = svd(sc(a)).U
    return U
end

function analyse(case, x, g, J; save=String[])
    n = size(J, 1)
    F = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, B=KSPreconditioner(x; kind=:jacobian), project=odd!)
    F.linearise!(F.lin, x.a)
    bases = Dict("sine" => Matrix(1.0I, 32, 32),
                 "podA" => readdlm(joinpath(@__DIR__, "pod", "pod_attractor.txt")),
                 "podO" => pod_orbit(g, x.a),
                 "svd"  => readdlm(joinpath(@__DIR__, "..", "explore", "global_basis_1.txt")),
                 "test" => readdlm(joinpath(@__DIR__, "..", "explore", "bases", "smoother_repeat1_8.txt")))
    for f in readdir(joinpath(@__DIR__, "bases"))              # resolvent bases (resolvent.py)
        bases[splitext(f)[1]] = readdlm(joinpath(@__DIR__, "bases", f))
    end
    function run(name; basis="sine", r=16, tc=2, pre=2, post=2, cycle=:multiplicative, h=4, neutral=true,
                 time=:splines)
        t0 = Base.time()
        coords, field! = basis_coordinates(g, bases[basis])
        tg = TwoGrid(KSPreconditioner(x; kind=:jacobian), coords, field!, r; tc, time, h, neutral, cycle, pre, post, solver=:sparse)
        relinearise!(tg, F, x)
        Minv = dense((o, v) -> precondition!(o, tg, F, v), x, n)
        λ = eigvals(I - Minv * J)
        writeeig("$(case)_eig_$name.txt", λ)
        if name in save
            nc = tg.nodes
            P = zeros(n, nc); ec = zeros(nc); o = similar(x)
            for c in 1:nc
                fill!(ec, 0); ec[c] = 1
                to_vector!(view(P, :, c), prolong!(o, tg, ec))
            end
            Kinv = reduce(hcat, [coarse_solve(tg.K, (fill!(ec, 0); ec[c] = 1; copy(ec))) for c in 1:nc])
            writebin("$(case)_Minv_$name.bin", Minv); writebin("$(case)_P_$name.bin", P)
            writebin("$(case)_K_$name.bin", inv(Kinv)); writebin("$(case)_G_$name.bin", transpose(P) * J * P)
            writedlm(joinpath(OUTD, "$(case)_dims_$name.txt"), [n nc tg.r + (tg.V === nothing ? 0 : size(tg.V, 2)) ReSolver.nblocks(tg)])
        end
        @printf "%s %-28s ρ(E) = %8.3g, |λ| > 0.1: %4d  (%.0f s)\n" case name maximum(abs, λ) count(>(0.1), abs.(λ)) Base.time() - t0
        flush(stdout)
    end
    return run
end

