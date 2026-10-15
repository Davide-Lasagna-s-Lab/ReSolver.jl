# Analysis of the preconditioners for the shortest periodic orbit of the Lorenz system.
#
#     julia --project=examples examples/lorenz/scripts/preconditioners.jl
#
# writes preconditioners.png in docs/src/assets/lorenz/. Three choices of B are
# compared:
#
#     none        B = I
#     frequency   B = (1 + ω₀ n) I on the mode n: the time derivative only
#     jacobian    B = i ω₀ n I - J̄ on the mode n: the linear space-time operator with the Jacobian
#                 frozen at the mean state of the orbit
#
# For each, the hookstep is run from the perturbed orbit of convergence.jl at K = 10, 20, 40 and 80
# modes, with the Newton system solved to 10⁻³, and the Jacobian actions needed to reach
# ‖r‖ < 10⁻¹² are recorded. The matrix of the preconditioned Jacobian 𝒥B⁻¹ at the converged orbit
# is then formed column by column for K = 20, in coordinates orthonormal for the inner product, and
# its eigenvalues and singular values computed: their clustering governs the convergence of the
# Krylov solver.

using Printf
using Random
using CairoMakie

include("../lorenz.jl")

import ReSolver: jacobian!, residual!, _hookstep


# ============================================================================ #
# Converged orbit and perturbation, as in convergence.jl                       #
# ============================================================================ #

dt = 0.005
U  = trajectory(trajectory([1.0, 1.0, 1.0], dt, 4000)[:, end], dt, 20000)

lags    = round(Int, 1.4 / dt):round(Int, 1.7 / dt)
e, i, m = minimum((norm(U[:, i + m] .- U[:, i]) / norm(U[:, i]), i, m)
                  for i in 1:size(U, 2) - last(lags), m in lags)

x = initial_orbit(U, dt, i, m, 20)
F = System(nonlinear!, LorenzLinearised(20), LorenzLinearised(20; adjoint=true), dds!, x;
           linearise!, B=LorenzPreconditioner(x))

solve!(x, F, NewtonHookstep(maxiter=20, krylov_dim=40, Δ=1, Δmax=100, tol=1e-12, verbose=false))

Random.seed!(2)
δ = LorenzField(5)
δ.data .= randn(ComplexF64, 3, 6)
δ.data[:, 1] .= real.(δ.data[:, 1])
δ.data .*= 0.01 * norm(x.a) / norm(δ)

start(K) = Orbit(resample(x.a, K) .+ resample(δ, K), x.p .+ 0.01)

# the preconditioner of a given kind for the orbit y
preconditioner(kind::Symbol, y::Orbit) = kind === :none ? I : LorenzPreconditioner(y; kind)

kinds  = (:none, :frequency, :jacobian)
marks  = Dict(:none => :circle, :frequency => :rect, :viscous => :diamond, :jacobian => :utriangle, :linear => :utriangle)
colors = Dict(:none => :gray40, :frequency => :orange, :jacobian => :crimson)


# ============================================================================ #
# Cost against the number of unknowns                                          #
# ============================================================================ #

Ks   = (10, 20, 40, 80)
cost = Dict()
traces = Dict()

for K in Ks, kind in kinds
    xK = start(K)
    FK = System(nonlinear!, LorenzLinearised(K), LorenzLinearised(K; adjoint=true), dds!, xK;
                linearise!, B=preconditioner(kind, xK))

    trace = Trace()
    solve!(copy(xK), FK, NewtonHookstep(maxiter=12, krylov_dim=200, krylov_tol=1e-3, Δ=10,
                                        Δmax=100, tol=1e-12, verbose=false, callback=trace))

    done             = trace.res[end] < 1e-12
    cost[(K, kind)]   = done ? trace.evaluations[end].jacobian : NaN
    traces[(K, kind)] = trace

    @printf "K = %2d (%3d unknowns), %-10s ‖r‖ = %.1e in %2d iterations, %5d Jacobian actions\n" K 6K + 4 kind trace.res[end] length(trace) - 1 trace.evaluations[end].jacobian
    flush(stdout)
end


# ============================================================================ #
# Spectrum of the preconditioned Jacobian, K = 20                              #
# ============================================================================ #

# Coordinates orthonormal for the inner product: Re û₀; √2 Re û_n and √2 Im û_n for n ≥ 1; the
# log-frequency. The imaginary part of û₀ is not a degree of freedom of a real orbit.
function tovector(p::Orbit)
    û = p.a.data
    return [vec(real.(û[:, 1])); sqrt(2) .* vec(real.(û[:, 2:end])); sqrt(2) .* vec(imag.(û[:, 2:end])); p.p]
end

function fromvector(v::Vector, K::Int)
    n = 3K
    û = zeros(ComplexF64, 3, K + 1)
    û[:, 1]     .= v[1:3]
    û[:, 2:end] .= (reshape(v[4:3 + n], 3, K) .+ im .* reshape(v[4 + n:3 + 2n], 3, K)) ./ sqrt(2)

    return Orbit(LorenzField(û), v[end:end])
end

# the matrix of v ↦ 𝒥 B⁻¹ v at the orbit y, column by column
function matrix(y::Orbit, kind::Symbol)
    K  = modes(y.a)
    N  = 6K + 4
    B  = preconditioner(kind, y)
    Fy = System(nonlinear!, LorenzLinearised(K), LorenzLinearised(K; adjoint=true), dds!, y;
                linearise!, B)
    Fy.linearise!(Fy.lin, y.a)

    A = zeros(N, N)
    w = similar(y)
    for j in 1:N
        e       = zeros(N)
        e[j]    = 1
        A[:, j] = tovector(jacobian!(similar(y), Fy, y, precondition!(w, B, fromvector(e, K))))
    end

    return A
end

spectra = Dict(kind => matrix(x, kind) for kind in kinds)

for kind in kinds
    σ = svdvals(spectra[kind])
    @printf "%-10s condition number of 𝒥B⁻¹: %.2e\n" kind σ[1] / σ[end]
end


# ============================================================================ #
# Figure                                                                       #
# ============================================================================ #

# Logarithmic ticks at integer powers of ten, with minor ticks at 2–9 times them.
function logticks!(ax, lo::Real, hi::Real; axis::Symbol=:x)
    k₁ = floor(Int, log10(lo))
    k₂ = ceil(Int, log10(hi))
    st = k₂ - k₁ > 6 ? 3 : 1

    ks    = filter(k -> mod(k, st) == 0, k₁:k₂)
    major = (10.0 .^ ks, [rich("10", superscript(string(k))) for k in ks])
    minor = st == 1 ? [m * 10.0^k for k in k₁:k₂ - 1 for m in 2:9] : 10.0 .^ (k₁:k₂)

    setproperty!(ax, Symbol(axis, :ticks), major)
    setproperty!(ax, Symbol(axis, :minorticks), minor)
    setproperty!(ax, Symbol(axis, :minorticksvisible), true)
    setproperty!(ax, Symbol(axis, :minorgridvisible), true)

    return ax
end

fig = Figure(size=(1300, 820), fontsize=13)

# ---- cost against the number of unknowns ----
ax1 = Axis(fig[1, 1]; title="cost of ‖r‖ < 10⁻¹²", xlabel="unknowns, 6K + 4",
           ylabel="Jacobian actions", xscale=log10, yscale=log10)
# converged searches as filled symbols; the others, open, at the cost they spent
for kind in kinds
    xs = [6K + 4 for K in Ks]
    ys = [cost[(K, kind)] for K in Ks]
    sp = [traces[(K, kind)].evaluations[end].jacobian for K in Ks]

    lines!(ax1, xs, ys; color=colors[kind])
    scatter!(ax1, xs, ys; color=colors[kind], marker=marks[kind], markersize=11, label=string(kind))
    scatter!(ax1, xs[isnan.(ys)], sp[isnan.(ys)]; color=:white, strokecolor=colors[kind],
             strokewidth=1.5, marker=marks[kind], markersize=11)
end
ylims!(ax1, 30, 1e4)
logticks!(ax1, 6 * first(Ks) + 4, 6 * last(Ks) + 4)
logticks!(ax1, 10, 1e4; axis=:y)
text!(ax1, 0.03, 0.97; space=:relative, align=(:left, :top), fontsize=11,
      text="open symbols: not converged in 12 Newton iterations")
axislegend(ax1; position=:rc)

# ---- inner solve of the first Newton iteration, K = 40 ----
ax2 = Axis(fig[1, 2]; title="inner solve, Newton iteration 1, K = 40", xlabel="Arnoldi step",
           ylabel="‖g - Hy‖ / β", yscale=log10)
for kind in kinds
    k = traces[(40, kind)].krylov[2]
    lines!(ax2, 1:length(k), k; color=colors[kind], label=string(kind))
end
ylims!(ax2, 1e-4, 2)
logticks!(ax2, 1e-4, 1; axis=:y)

# ---- singular values of 𝒥B⁻¹, K = 20 ----
ax3 = Axis(fig[1, 3]; title=L"\text{singular values of } \mathcal{J}B^{-1}\text{, } K = 20", xlabel="index",
           ylabel="σ", yscale=log10)
for kind in kinds
    σ = svdvals(spectra[kind])
    lines!(ax3, 1:length(σ), σ; color=colors[kind], label=string(kind))
end

# ---- eigenvalues of 𝒥B⁻¹, K = 20 ----
for (j, kind) in enumerate(kinds)
    λ  = eigvals(spectra[kind])
    ax = Axis(fig[2, j]; title=L"\text{eigenvalues of } \mathcal{J}B^{-1}\text{, %$(kind)}",
              xlabel=L"\mathrm{Re}\,\lambda", ylabel=L"\mathrm{Im}\,\lambda")

    # the axes first, the eigenvalues on top
    vlines!(ax, 0; color=:gray70)
    hlines!(ax, 0; color=:gray70)
    scatter!(ax, real.(λ), imag.(λ); color=colors[kind], markersize=7)
end

save(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "lorenz", "preconditioners.png"), fig)
