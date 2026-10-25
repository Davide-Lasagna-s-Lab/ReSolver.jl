# Common set-up of the two-grid study on Kuramoto–Sivashinsky, L = 39, odd subspace.
#
#     load_case(name)        initial orbit and grid of a test case (see cases below)
#     run_newton(x, g; ...)  Newton–hookstep with B only or with TwoGrid, and its costs
using Printf, LinearAlgebra, DelimitedFiles, Random, SparseArrays
using BorderedMatrices
const HERE = @__DIR__
include(joinpath(HERE, "..", "ks.jl"))
using ReSolver
import ReSolver: frequency
using FFTW
BLAS.set_num_threads(1); FFTW.set_num_threads(1)   # single-threaded timings

const L   = 39
const NX  = 65
const DAT = joinpath(HERE, "data")
const T0  = readdlm(joinpath(DAT, "params.txt"))[3]

# converged prime orbits: field on the grid and log ω
function prime_orbit(M::Int)
    M == 1 && return readdlm(joinpath(DAT, "orbit.txt")), log(readdlm(joinpath(DAT, "params.txt"))[2])
    return readdlm(joinpath(DAT, "long$(M)_orbit.txt")), readdlm(joinpath(DAT, "long$(M)_orbit_p.txt"))[1]
end

# starting points of the prime orbits used in the earlier study: the smoothed near-recurrence for
# M = 1, the L-BFGS end points for M = 2, 4, 8
function prime_start(M::Int)
    M == 1 && return readdlm(joinpath(DAT, "guess.txt")), readdlm(joinpath(DAT, "params.txt"))[1]
    return readdlm(joinpath(DAT, "long$(M)_start.txt")), readdlm(joinpath(DAT, "long$(M)_start_p.txt"))[1]
end

odd_ns(n) = isodd(n) ? n : n + 1

# the prime orbit M traversed q times, on Ns ≈ q Ns(M) slices (odd), perturbed by a random field
# that is smooth in space and time (wavenumbers k ≤ 8, physical frequencies |ω n| ≤ 1) and not
# periodic with the period of the prime orbit, of relative size ε
function repeated_start(M::Int, q::Int; ε=0.002, seed=1)
    A, ρ = prime_orbit(M)
    Ns   = odd_ns(q * size(A, 2))
    g0   = KSGrid(L, NX, size(A, 2))
    g    = KSGrid(L, NX, Ns)
    # tile in time by spectral interpolation of the repeated field
    Â    = fft(A, 2)
    N0   = size(A, 2); nn = (N0 - 1) ÷ 2
    B̂    = zeros(ComplexF64, NX, Ns)
    for n in -nn:nn
        B̂[:, mod(q * n, Ns) + 1] = Â[:, mod(n, N0) + 1] * (Ns / N0)
    end
    u    = KSField(g, real(ifft(B̂, 2)))
    ω    = exp(ρ) / q
    # perturbation, smooth in space and time
    rng  = MersenneTwister(seed)
    v̂    = zeros(ComplexF64, length(g.k), Ns)
    for kk in 2:9, l in 1:Ns
        abs(ω * g.n[l]) <= 1 && (v̂[kk, l] = randn(rng) + im * randn(rng))
    end
    v    = KSField(g, irfft(v̂, NX, (1, 2)) .* NX .* Ns)
    odd!(v); odd!(u)
    u.data .+= (ε * norm(u) / norm(v)) .* v.data
    return Orbit(u, [log(ω)]), g
end

function prime_case(M::Int)
    A, ρ = prime_start(M)
    g = KSGrid(L, NX, size(A, 2))
    return Orbit(KSField(g, copy(A)), [ρ]), g
end

# ---- Newton–hookstep with a given preconditioner, returning its costs ----
# kind = :plain (B only) or :twogrid; keywords of TwoGrid passed through
function run_newton(x::Orbit, g::KSGrid; kind=:twogrid, r=12, tc=1, neutral=true, cycle=:multiplicative,
                    pre=1, post=1, h=4, solver=:banded, krylov_dim=nothing, maxiter=15, tol=1e-10,
                    krylov_tol=1e-3, verbose=false)
    x   = copy(x)
    ops = (KSNonlinear(g), KSLinearised(g))
    B0  = KSPreconditioner(x; kind=:jacobian)
    if kind === :plain
        B = B0
    else
        coords, field! = odd_sine_coordinates(g)
        B = TwoGrid(B0, coords, field!, r; tc, h, neutral, cycle, pre, post, solver)
    end
    F   = System(ops..., dds!, x; linearise!, B, project=odd!)
    T   = 2π / frequency(x)
    kd  = krylov_dim === nothing ? (kind === :plain ? round(Int, 2.2T + 150) : 60) : krylov_dim
    tr  = Trace()
    GC.gc()
    t   = @elapsed solve!(x, F, NewtonHookstep(; maxiter, tol, krylov_dim=kd, krylov_tol,
                                                Δ=0.1, Δmax=10, verbose, callback=tr))
    its  = [length(k) for k in tr.krylov[2:end]]
    jac  = tr.evaluations[end].jacobian
    bld  = kind === :plain ? 0 : B.builds
    N    = 32 * g.Ns + 1
    kryl = 8 * N * (maximum(its; init=0) + 1)      # the Krylov basis at its largest
    crs  = kind === :plain ? 0 : ReSolver.coarse_bytes(B)
    return (; T, Ns=g.Ns, its, steps=length(its), res=tr.res[end], jac, builds=bld,
            actions=jac + bld, time=t, krylov_MB=kryl / 1e6, coarse_MB=crs / 1e6,
            nodes=kind === :plain ? 0 : B.nodes, converged=tr.res[end] < tol)
end

fmt(o) = @sprintf("T=%7.1f Ns=%5d | steps %2d its %s | ‖r‖ %.1e | actions %5d (J %5d + build %4d) | %7.1f s | Krylov %7.1f MB, coarse %7.1f MB (%d unknowns)",
                  o.T, o.Ns, o.steps, string(o.its), o.res, o.actions, o.jac, o.builds, o.time, o.krylov_MB, o.coarse_MB, o.nodes)
