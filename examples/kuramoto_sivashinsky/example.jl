# An orbit of the Kuramoto–Sivashinsky equation on L = 22, the domain studied by Cvitanović,
# Davidchack and Siminos (2010), searched for as a relative periodic orbit. The initial guess is a
# near-recurrence of a chaotic trajectory, away from the equilibria, with period between 12 and 20,
# and the Newton–Krylov hookstep converges it, with the jacobian preconditioner of ks.jl: the
# linear space-time operator about the mean state u = 0, diagonal in Fourier. It finds the shortest
# pre-periodic orbit of the system, traversed twice: T = 20.5057, zero drift.
#
#     julia --project=examples examples/kuramoto_sivashinsky/example.jl
#
# writes example.png in docs/src/assets/kuramoto_sivashinsky/.

using Random
using Printf
using CairoMakie

include("ks.jl")

Random.seed!(1)


# ---- grid: 33 points along x, 49 in rescaled time ----
g = KSGrid(22, 33, 49)

# ---- chaotic trajectory: random snapshot of zero mean, which KS conserves, transient of 200,
#      then 400 time units ----
Δt = 0.25
u₀ = 0.1 .* randn(g.Nx)
u₀ .-= sum(u₀) / g.Nx
U  = integrate(u₀, g, 0.05, 4000; every=4000)
U  = integrate(U[:, end], g, 0.05, 8000; every=5)

# ---- initial guess: the best near-recurrence with period in [12, 20], with drift ----
e, i, m, ℓ = recurrence(U, g, Δt, (12, 20))
x₀         = initial_orbit(U, g, Δt, i, m, ℓ)

@printf "initial guess: recurrence error %.3f, T = %.2f, shift = %.2f\n" e m * Δt ℓ

# ---- operators ----
nl  = KSNonlinear(g)
lin = KSLinearised(g)
adj = KSLinearised(g; adjoint=true)

# ---- search: hookstep without and with the jacobian preconditioner, Newton systems solved to
#      10⁻³; without, the search makes no progress and is stopped after 10 iterations. The
#      projection demean! removes the mean, and with it the Galilean family of solutions ----
runs = map((("no preconditioner", I, 10), ("preconditioner", KSPreconditioner(x₀; kind=:jacobian), 50))) do (name, B, maxiter)
    F     = System(nl, lin, adj, dds!, x₀; linearise!, ddi=(ddx!,), B, project=demean!)
    x     = copy(x₀)
    trace = Trace()

    solve!(x, F, NewtonHookstep(; maxiter, krylov_dim=150, krylov_tol=1e-3, Δ=0.1, Δmax=10,
                                tol=1e-11, verbose=false, callback=trace))

    T = 2π / ReSolver.frequency(x)
    @printf "%-18s ‖r‖ = %.2e in %2d Newton iterations, %4d Jacobian actions; T = %.4f, shift = %.4f\n" name trace.res[end] length(trace) - 1 trace.evaluations[end].jacobian T -x.p[2] * T

    (; name, x, trace)
end

x = runs[2].x


# ============================================================================ #
# Figure                                                                       #
# ============================================================================ #

# Logarithmic ticks on an axis of ax: major ticks at integer powers of ten, labelled 10ⁿ, every
# decade or every third decade, at multiples of three, over a wide range; minor ticks at 2–9 times the powers of ten, or
# at every decade over a wide range.
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

# the orbit over one period in the fixed frame, u(x', t) = a(x' - c t, ωt)
function fixed_frame(x::Orbit)
    T = 2π / ReSolver.frequency(x)
    c = x.p[2]
    t = (0:g.Ns - 1) .* (T / g.Ns)
    u = reduce(hcat, [shift(x.a.data[:, l], -c * t[l], g) for l in 1:g.Ns])

    return t, u
end

fig = Figure(size=(1100, 1150), fontsize=14)

# ---- top row: initial guess and converged orbit, in a layout of their own ----
top = fig[1, 1:2] = GridLayout()
for (j, (title, y)) in enumerate((("initial guess", x₀), ("converged orbit", x)))
    t, u = fixed_frame(y)
    ax   = Axis(top[1, j]; title, xlabel="x", ylabel="t")
    hm   = heatmap!(ax, xpoints(g), t, u; colormap=:balance, colorrange=(-3, 3))
    j == 2 && Colorbar(top[1, 3], hm; label="u")
end

# ---- one row per search: Newton iterations and inner solves ----
res    = reduce(vcat, [run.trace.res for run in runs])
newton = Axis[]
inner  = Axis[]
for (row, (run, color)) in enumerate(zip(runs, (:gray40, :crimson)))
    tr  = run.trace
    its = 0:length(tr) - 1

    # Newton iterations; open symbols: hooksteps, filled: full Newton steps
    ax = Axis(fig[row + 1, 1]; title="Newton iterations, " * run.name, xlabel="iteration",
              ylabel="‖r‖", yscale=log10)
    lines!(ax, its, tr.res; color)
    for (kind, fill) in (("", color), ("hook", :white), ("newton", color))
        sel = tr.step .== kind
        scatter!(ax, its[sel], tr.res[sel]; color=fill, strokecolor=color, strokewidth=1.2,
                 markersize=8)
    end
    logticks!(ax, minimum(res), maximum(res); axis=:y)
    push!(newton, ax)

    # inner solves, one curve per Newton iteration, from dark (first) to light (last)
    ax    = Axis(fig[row + 1, 2]; title="GMRES, " * run.name, xlabel="Arnoldi step",
                 ylabel="‖g - Hy‖ / β", yscale=log10)
    niter = length(tr) - 1
    for j in 1:niter
        k = tr.krylov[j + 1]
        lines!(ax, 1:length(k), k; color=j, colormap=:viridis, colorrange=(1, niter))
    end
    hlines!(ax, 1e-3; color=:black, linestyle=:dash)
    ylims!(ax, 1e-4, 2)
    logticks!(ax, 1e-4, 1; axis=:y)
    Colorbar(fig[row + 1, 3]; colormap=:viridis, limits=(1, niter), label="Newton iteration")
    push!(inner, ax)
end
linkyaxes!(newton...)

# ---- the orbits take a little more room than the convergence rows ----
rowsize!(fig.layout, 1, Relative(0.4))

save(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "kuramoto_sivashinsky", "example.png"), fig)
