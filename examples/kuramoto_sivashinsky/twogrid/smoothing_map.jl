# Where does the smoother fail? For each space-time Fourier mode v (sine mode k in space, cos or sin
# of n s in time) of an odd field, the amplification ‖(I - B⁻¹ 𝒥) v‖/‖v‖ of one smoothing step with
# ReSolver's diagonal preconditioner B, at a converged orbit. Values below 1 are damped by the
# smoother; values near or above 1 are what the coarse space must handle.
#     julia --project=. smoothing_map.jl      → results/smoothing_map_M{1,8}.txt
include("common.jl")
import ReSolver: jacobian!, precondition!

function smoothing_map(M::Int; nmax=nothing)
    A, ρ = prime_orbit(M)
    g = KSGrid(L, NX, size(A, 2))
    x = Orbit(KSField(g, copy(A)), [ρ])
    B = KSPreconditioner(x; kind=:jacobian)
    F = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, B, project=odd!)
    F.linearise!(F.lin, x.a)
    coords, field! = odd_sine_coordinates(g)
    c  = coords(x.a)
    nx, Ns = size(c)
    ω  = frequency(x)
    s  = 2π .* (0:Ns - 1) ./ Ns
    ns = 0:(nmax === nothing ? (Ns - 1) ÷ 2 : nmax)
    amp = zeros(nx, length(ns))
    v, Jv, w = similar(x), similar(x), similar(x)
    for (j, n) in enumerate(ns), k in 1:nx, f in (cos, sin)
        n == 0 && f === sin && continue
        fill!(c, 0); c[k, :] .= f.(n .* s)
        field!(v.a, c); fill!(v.p, 0)
        jacobian!(Jv, F, x, v)
        precondition!(w, B, Jv); odd!(w.a)
        w.a .= v.a .- w.a
        amp[k, j] = max(amp[k, j], norm(w.a) / norm(v.a))
    end
    return ω .* collect(ns), amp
end

mkpath(joinpath(HERE, "results"))
for M in (1, 8)
    ωn, amp = smoothing_map(M; nmax=M == 1 ? nothing : 120)
    writedlm(joinpath(HERE, "results", "smoothing_map_M$M.txt"), [ωn'; amp])
    @printf "M=%d: amplification > 0.5 for k ≤ %d and |ωn| ≤ %.2f; > 1 for k ≤ %d and |ωn| ≤ %.2f\n" M maximum(findall(vec(any(amp .> 0.5; dims=2)))) maximum(ωn[vec(any(amp .> 0.5; dims=1))]) maximum(findall(vec(any(amp .> 1; dims=2)))) maximum(ωn[vec(any(amp .> 1; dims=1))])
end
