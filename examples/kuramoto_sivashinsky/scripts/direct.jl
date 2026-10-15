# Newton's method with a sparse direct solver, finite differences in time, against the matrix-free
# hookstep, for the orbit of the Kuramoto–Sivashinsky equation of example.jl.
#
#     julia --project=examples examples/kuramoto_sivashinsky/scripts/direct.jl
#
# The direct method is that of VaPOrE.jl and Lasagna (2018, appendix B): the orbit is sampled at M
# rescaled times, the derivative in time is a centred finite difference of order p = 2, …, 10, the
# Jacobian of the spatial right-hand side is a dense Nx × Nx block at each time, and the Newton
# matrix, cyclic block-banded with bandwidth p/2 blocks and bordered by two phase conditions on the
# first time level, is factorised by sparse LU (UMFPACK). For each order the number of times M is
# chosen so that the finite-difference derivative of the orbit is accurate to 10⁻⁶, and Newton's
# method is run to convergence to check the period; the size of the matrix and of its LU factors,
# and the time to assemble, factorise and solve, are measured on grids refined in space. The
# matrix-free hookstep of convergence.jl is timed on the same grids.

using Random
using Printf
using LinearAlgebra
using SparseArrays

include("../ks.jl")

Random.seed!(1)


# ============================================================================ #
# Converged orbit, as in convergence.jl                                        #
# ============================================================================ #

g₀ = KSGrid(22, 33, 49)
u₀ = 0.1 .* randn(g₀.Nx)
u₀ .-= sum(u₀) / g₀.Nx
U  = integrate(u₀, g₀, 0.05, 4000; every=4000)
U  = integrate(U[:, end], g₀, 0.05, 8000; every=5)

e, i, m, ℓ = recurrence(U, g₀, 0.25, (12, 20))
x★         = initial_orbit(U, g₀, 0.25, i, m, ℓ)

F₀ = System(KSNonlinear(g₀), KSLinearised(g₀), KSLinearised(g₀; adjoint=true), dds!, x★;
            linearise!, ddi=(ddx!,), B=KSPreconditioner(x★; kind=:jacobian), project=demean!)
solve!(x★, F₀, NewtonHookstep(maxiter=50, krylov_dim=150, Δ=0.1, Δmax=10, tol=1e-12, verbose=false))

T★ = 2π / ReSolver.frequency(x★)
@printf "spectral orbit: T = %.10f, c = %.2e\n" T★ x★.p[2]


# ============================================================================ #
# Finite differences in time                                                   #
# ============================================================================ #

# coefficients of the centred differences of order p, as in VaPOrE.jl: the derivative at sᵢ is
# Σⱼ cⱼ (u_{i+j} - u_{i-j}) / Δs
const FD = Dict(2  => [1] ./ 2,
                4  => [8, -1] ./ 12,
                6  => [45, -9, 1] ./ 60,
                8  => [672, -168, 32, -3] ./ 840,
                10 => [2100, -600, 150, -25, 2] ./ 2520)

# the finite-difference derivative in s of the columns of u, periodic
function fdds(u::AbstractMatrix, p::Int)
    M  = size(u, 2)
    du = zero(u)
    for (j, c) in enumerate(FD[p])
        du .+= c .* (circshift(u, (0, -j)) .- circshift(u, (0, j)))
    end
    return du .* (M / 2π)
end

# relative error of the finite-difference derivative of the spectral orbit sampled at M times
function fd_error(p::Int, M::Int)
    gM = KSGrid(22, x★.a.g.Nx, M)
    a  = resample(x★.a, gM)
    ds = dds!(similar(a), a)
    return norm(fdds(a.data, p) .- ds.data) / norm(ds.data)
end

# the smallest odd number of times with a derivative accurate to tol, up to Mmax
function times_for(p::Int, tol::Real; Mmax::Int=20001)
    M = 9
    while fd_error(p, M) > tol
        M = M < 200 ? M + 2 : 2(M ÷ 2 * 11 ÷ 10) + 1
        M > Mmax && return 0
    end
    return M
end


# ============================================================================ #
# The direct Newton method                                                     #
# ============================================================================ #

# spectral differentiation matrices on Nx points, by FFTs of the identity columns
function spectral_matrices(g::KSGrid)
    Nx = g.Nx
    D  = [zeros(Nx, Nx) for _ in 1:3]
    for j in 1:Nx
        e      = zeros(Nx)
        e[j]   = 1
        ê      = rfft(e)
        D[1][:, j] .= irfft(im .* g.k .* ê, Nx)
        D[2][:, j] .= irfft(-(g.k .^ 2) .* ê, Nx)
        D[3][:, j] .= irfft((g.k .^ 4) .* ê, Nx)
    end
    return D
end

# KS right-hand side at one time, -½ ∂x(u²) - ∂x²u - ∂x⁴u, and its Jacobian. The assembled matrix
# has no projection, so the spatial mean at each time is pinned instead, by the term -mean(u)
ks_rhs(u, D)      = -D[1] * (u .^ 2) ./ 2 .- D[2] * u .- D[3] * u .- sum(u) / length(u)
ks_jacobian(u, D) = -D[1] * Diagonal(u) .- D[2] .- D[3] .- ones(length(u), length(u)) ./ length(u)

# residual of the orbit u (Nx × M) with log-frequency ρ and drift speed c, and the bordered Newton
# matrix, with the phase conditions of VaPOrE.jl on the first time level
function newton_system(u::AbstractMatrix, ρ::Real, c::Real, p::Int, D)
    Nx, M = size(u)
    ω     = exp(ρ)
    n     = Nx * M + 2
    du    = fdds(u, p)

    # ---- residual ----
    r = similar(u)
    for l in 1:M
        r[:, l] .= ω .* du[:, l] .- c .* (D[1] * u[:, l]) .- ks_rhs(u[:, l], D)
    end

    # ---- matrix: dense diagonal blocks, scaled identities off the diagonal, borders ----
    I, J, V = Int[], Int[], Float64[]
    block(l) = (l - 1) * Nx .+ (1:Nx)
    for l in 1:M
        A = -c .* D[1] .- ks_jacobian(u[:, l], D)
        for (q, col) in enumerate(block(l)), (k, row) in enumerate(block(l))
            push!(I, row); push!(J, col); push!(V, A[k, q])
        end
        for (j, cj) in enumerate(FD[p]), (σ, sign) in ((j, 1), (-j, -1))
            lo = mod(l - 1 + σ, M) + 1
            for k in 1:Nx
                push!(I, block(l)[k]); push!(J, block(lo)[k]); push!(V, sign * ω * cj * M / 2π)
            end
        end
        append!(I, block(l)); append!(J, fill(n - 1, Nx)); append!(V, ω .* du[:, l])
        append!(I, block(l)); append!(J, fill(n, Nx));     append!(V, -(D[1] * u[:, l]))
    end
    append!(I, fill(n - 1, Nx)); append!(J, block(1)); append!(V, ks_rhs(u[:, 1], D))
    append!(I, fill(n, Nx));     append!(J, block(1)); append!(V, D[1] * u[:, 1])

    return vec(r), sparse(I, J, V, n, n)
end

# Newton's method from u, with the cost of each part of the first iteration
function direct_newton(u::AbstractMatrix, ρ::Real, c::Real, p::Int, g::KSGrid; maxiter::Int=10,
                       tol::Real=1e-9)
    D    = spectral_matrices(g)
    u    = copy(u)
    cost = nothing

    for it in 1:maxiter
        ta   = @elapsed (r, A) = newton_system(u, ρ, c, p, D)
        res  = norm(r) / sqrt(length(r))
        res < tol && return (; u, ρ, c, res, it = it - 1, cost)
        tf   = @elapsed LU = lu(A)
        ts   = @elapsed δ = LU \ [-r; 0; 0]
        cost === nothing && (cost = (; n = size(A, 1), nnzA = nnz(A), nnzLU = nnz(LU.L) + nnz(LU.U),
                                       assemble = ta, factorise = tf, solve = ts))
        u .+= reshape(δ[1:end - 2], size(u))
        ρ  += δ[end - 1]
        c  += δ[end]
    end

    return (; u, ρ, c, res = NaN, it = maxiter, cost)
end


# ============================================================================ #
# Orders and resolutions                                                       #
# ============================================================================ #

orders = (2, 4, 6, 8, 10)
Ms     = Dict(p => times_for(p, 1e-6) for p in orders)
for p in orders
    @printf "order %2d: M = %5d times for a derivative accurate to 1e-6\n" p Ms[p]
end

# ---- Newton with the direct solver on the base grid, from the spectral orbit at M times ----
for p in orders
    M = Ms[p]
    (M == 0 || M > 1001) && continue
    gM  = KSGrid(22, 33, M)
    a   = resample(x★.a, gM)
    out = direct_newton(a.data, x★.p[1], x★.p[2], p, g₀)
    c   = out.cost
    @printf "order %2d, 33 × %4d: T = %.10f (T - T★ = %.1e) in %d iterations; n = %6d, nnz(A) = %.2e, nnz(LU) = %.2e (%.0f MB); assemble %.3f s, factorise %.3f s, solve %.3f s\n" p M 2π / exp(out.ρ) 2π / exp(out.ρ) - T★ out.it c.n c.nnzA c.nnzLU 16e-6 * c.nnzLU c.assemble c.factorise c.solve
    flush(stdout)
end

# ---- one Newton iteration with the direct solver on grids refined in space, order 10 ----
for Nx in (33, 65, 129, 257)
    M  = Ms[10]
    gM = KSGrid(22, Nx, M)
    gx = KSGrid(22, Nx, 49)
    a  = resample(resample(x★.a, gx), gM)
    D  = spectral_matrices(gx)

    ta = @elapsed (r, A) = newton_system(a.data, x★.p[1], x★.p[2], 10, D)
    tf = @elapsed LU = lu(A)
    ts = @elapsed LU \ [-r; 0; 0]
    @printf "order 10, %3d × %4d: n = %7d, nnz(A) = %.2e, nnz(LU) = %.2e (%.0f MB); assemble %.2f s, factorise %.2f s, solve %.3f s\n" Nx M size(A, 1) nnz(A) nnz(LU.L) + nnz(LU.U) 16e-6 * (nnz(LU.L) + nnz(LU.U)) ta tf ts
    flush(stdout)
end

# ---- the matrix-free hookstep on the same spatial grids, 49 times, per Newton iteration ----
δ = resample(KSField(KSGrid(22, 7, 7), randn(7, 7)), g₀)
δ.data .*= 0.01 * norm(x★.a) / norm(δ)
for Nx in (33, 65, 129, 257)
    gx = KSGrid(22, Nx, 49)
    y  = Orbit(resample(x★.a, gx) .+ resample(δ, gx), x★.p .+ 0.01)
    Fy = System(KSNonlinear(gx), KSLinearised(gx), KSLinearised(gx; adjoint=true), dds!, y;
                linearise!, ddi=(ddx!,), B=KSPreconditioner(y; kind=:jacobian), project=demean!)

    solve!(copy(y), Fy, NewtonHookstep(maxiter=1, krylov_dim=200, Δ=10, Δmax=100, verbose=false))
    trace = Trace()
    solve!(y, Fy, NewtonHookstep(maxiter=12, krylov_dim=200, Δ=10, Δmax=100, tol=1e-9,
                                 verbose=false, callback=trace))

    # actions of this search only: the counters of the system also include the first, brief one
    its = length(trace) - 1
    kmx = maximum(length, trace.krylov)
    act = trace.evaluations[end].jacobian - trace.evaluations[1].jacobian
    @printf "matrix-free, %3d × 49: n = %7d, %d iterations, %.3f s per iteration, %d actions per iteration, %d stored vectors (%.1f MB)\n" Nx Nx * 49 + 2 its trace.time[end] / its act / its kmx + 1 8e-6 * (kmx + 1) * (Nx * 49 + 2)
    flush(stdout)
end
