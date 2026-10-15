# Analysis of the preconditioners for the orbit of the Kuramoto–Sivashinsky
# equation of example.jl.
#
#     julia --project=examples examples/kuramoto_sivashinsky/scripts/preconditioners.jl
#
# writes preconditioners.png in docs/src/assets/kuramoto_sivashinsky/. Five choices of B are
# compared, all diagonal in Fourier, with multipliers on the mode (k, n):
#
#     none        1
#     frequency   1 + |ω₀ n - c₀ k|, the time derivative in the moving frame
#     viscous     1 + k⁴, the fourth-order dissipation
#     linear      1 + |i(ω₀ n - c₀ k) + k⁴ - k²|, the size of the linear space-time operator about u = 0
#     jacobian    i(ω₀ n - c₀ k) + k⁴ - k², the same operator, complex
#
# For each, the hookstep is run from the perturbed orbit of convergence.jl on grids refined by 1, 2,
# 4 and 8, with the Newton system solved to 10⁻³, and the Jacobian actions needed to reach
# ‖r‖ < 10⁻⁹ are recorded: below, the residual is limited by rounding errors amplified by the
# fourth derivative, about 10⁻¹⁰ on the finest grid. The matrix of the preconditioned Jacobian 𝒥B⁻¹ at the converged orbit
# is then formed column by column on the coarsest grid, on the fields of zero mean, and its
# eigenvalues and singular values computed.

using Random
using Printf
using CairoMakie

include("../ks.jl")

import ReSolver: jacobian!

Random.seed!(1)


# ============================================================================ #
# Converged orbit and perturbation, as in convergence.jl                       #
# ============================================================================ #

g  = KSGrid(22, 33, 49)
Δt = 0.25
u₀ = 0.1 .* randn(g.Nx)
u₀ .-= sum(u₀) / g.Nx
U  = integrate(u₀, g, 0.05, 4000; every=4000)
U  = integrate(U[:, end], g, 0.05, 8000; every=5)

e, i, m, ℓ = recurrence(U, g, Δt, (12, 20))
x          = initial_orbit(U, g, Δt, i, m, ℓ)

F = System(KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!, x;
           linearise!, ddi=(ddx!,), B=KSPreconditioner(x), project=demean!)

solve!(x, F, LBFGS(maxiter=300, verbose=false))
solve!(x, F, NewtonHookstep(maxiter=20, krylov_dim=150, Δ=0.1, Δmax=10, tol=1e-11, verbose=false))

@printf "orbit: T = %.4f, ‖r‖ = %.1e\n" 2π / ReSolver.frequency(x) sqrt(2 * ReSolver.objective(F, x))

δ = resample(KSField(KSGrid(22, 7, 7), randn(7, 7)), g)
δ.data .*= 0.01 * norm(x.a) / norm(δ)

grid(f)  = KSGrid(22, 32f + 1, 48f + 1)
start(f) = Orbit(resample(x.a, grid(f)) .+ resample(δ, grid(f)), x.p .+ 0.01)

# the preconditioner of a given kind for the orbit y
preconditioner(kind::Symbol, y::Orbit) = kind === :none ? I : KSPreconditioner(y; kind)

kinds  = (:none, :frequency, :viscous, :linear, :jacobian)
marks  = Dict(:none => :circle, :frequency => :rect, :viscous => :diamond, :jacobian => :star5, :linear => :utriangle)
colors = Dict(:none => :gray40, :frequency => :orange, :viscous => :royalblue, :linear => :crimson,
              :jacobian => :seagreen)


# ============================================================================ #
# Cost against the number of unknowns                                          #
# ============================================================================ #

factors = (1, 2, 4, 8)
cost    = Dict()
traces  = Dict()
sizes   = Dict(f => grid(f).Nx * grid(f).Ns + 2 for f in factors)

for f in factors, kind in kinds
    gf = grid(f)
    xf = start(f)
    Ff = System(KSNonlinear(gf), KSLinearised(gf), KSLinearised(gf; adjoint=true), dds!, xf;
                linearise!, ddi=(ddx!,), B=preconditioner(kind, xf), project=demean!)

    trace = Trace()
    solve!(copy(xf), Ff, NewtonHookstep(maxiter=12, krylov_dim=200, krylov_tol=1e-3, Δ=10,
                                        Δmax=100, tol=1e-9, verbose=false, callback=trace))

    done              = trace.res[end] < 1e-9
    cost[(f, kind)]   = done ? trace.evaluations[end].jacobian : NaN
    traces[(f, kind)] = trace

    @printf "%3d × %3d (%6d unknowns), %-10s ‖r‖ = %.1e in %2d iterations, %5d Jacobian actions, %6.1f s\n" gf.Nx gf.Ns sizes[f] kind trace.res[end] length(trace) - 1 trace.evaluations[end].jacobian trace.time[end]
    flush(stdout)
end


# ============================================================================ #
# Spectrum of the preconditioned Jacobian on the coarsest grid                 #
# ============================================================================ #

# Coordinates orthonormal for the inner product, the mean over the grid points, on the space of the
# search: the field values divided by √(Nx Ns), in an orthonormal basis Q of the fields of zero mean,
# and the parameters. In the coordinates of all the grid values the masked mean would add a zero
# singular value.
Q = nullspace(ones(1, g.Nx * g.Ns))

tovector(p::Orbit) = [Q' * vec(p.a.data) ./ sqrt(length(p.a.data)); p.p]

fromvector(v::Vector, g::KSGrid) =
    Orbit(KSField(g, reshape(Q * v[1:end - 2] .* sqrt(g.Nx * g.Ns), g.Nx, g.Ns)), v[end - 1:end])

# the matrix of v ↦ 𝒥 B⁻¹ v at the orbit y on the coarsest grid, column by column
function matrix(y::Orbit, kind::Symbol)
    gy = y.a.g
    N  = size(Q, 2) + 2
    B  = preconditioner(kind, y)
    Fy = System(KSNonlinear(gy), KSLinearised(gy), KSLinearised(gy; adjoint=true), dds!, y;
                linearise!, ddi=(ddx!,), B)
    Fy.linearise!(Fy.lin, y.a)

    A = zeros(N, N)
    w = similar(y)
    for j in 1:N
        e       = zeros(N)
        e[j]    = 1
        A[:, j] = tovector(jacobian!(similar(y), Fy, y, precondition!(w, B, fromvector(e, gy))))
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

fig = Figure(size=(1750, 820), fontsize=13)

# ---- cost against the number of unknowns ----
ax1 = Axis(fig[1, 1:2]; title="cost of ‖r‖ < 10⁻⁹", xlabel="unknowns, Nx Ns + 2",
           ylabel="Jacobian actions", xscale=log10, yscale=log10)
# converged searches as filled symbols; the others, open, at the cost they spent
for kind in kinds
    xs = [sizes[f] for f in factors]
    ys = [cost[(f, kind)] for f in factors]
    sp = [traces[(f, kind)].evaluations[end].jacobian for f in factors]

    lines!(ax1, xs, ys; color=colors[kind])
    scatter!(ax1, xs, ys; color=colors[kind], marker=marks[kind], markersize=11, label=string(kind))
    scatter!(ax1, xs[isnan.(ys)], sp[isnan.(ys)]; color=:white, strokecolor=colors[kind],
             strokewidth=1.5, marker=marks[kind], markersize=11)
end
# room on the right for the legend
xlims!(ax1, 1e3, 1e6)
ylims!(ax1, 1e2, 1e4)
logticks!(ax1, 1e3, 1e6)
logticks!(ax1, 1e2, 1e4; axis=:y)
text!(ax1, 0.03, 0.97; space=:relative, align=(:left, :top), fontsize=11,
      text="open symbols: not converged in 12 Newton iterations (none, frequency and viscous overlap)")
axislegend(ax1; position=:rc)

# ---- inner solve of the first Newton iteration, finest grid ----
ax2 = Axis(fig[1, 3]; title="inner solve, Newton iteration 1, $(grid(8).Nx) × $(grid(8).Ns)",
           xlabel="Arnoldi step", ylabel="‖g - Hy‖ / β", yscale=log10)
for kind in kinds
    k = traces[(8, kind)].krylov[2]
    lines!(ax2, 1:length(k), k; color=colors[kind])
end
ylims!(ax2, 1e-4, 2)
logticks!(ax2, 1e-4, 1; axis=:y)

# ---- singular values of 𝒥B⁻¹, coarsest grid ----
ax3 = Axis(fig[1, 4:5]; title=L"\text{singular values of } \mathcal{J}B^{-1}\text{, } 33 \times 49", xlabel="index",
           ylabel="σ", yscale=log10)
for kind in kinds
    σ = svdvals(spectra[kind])
    lines!(ax3, 1:length(σ), σ; color=colors[kind])
end

# ---- eigenvalues of 𝒥B⁻¹, coarsest grid ----
for (j, kind) in enumerate(kinds)
    λ  = eigvals(spectra[kind])
    ax = Axis(fig[2, j]; title=L"\text{eigenvalues of } \mathcal{J}B^{-1}\text{, %$(kind)}",
              xlabel=L"\mathrm{Re}\,\lambda", ylabel=L"\mathrm{Im}\,\lambda")

    # the axes first, the eigenvalues on top
    vlines!(ax, 0; color=:gray70)
    hlines!(ax, 0; color=:gray70)
    scatter!(ax, real.(λ), imag.(λ); color=colors[kind], markersize=7)
end

save(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "kuramoto_sivashinsky", "preconditioners.png"), fig)
