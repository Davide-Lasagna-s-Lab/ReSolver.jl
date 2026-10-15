# A periodic orbit of the Kuramoto–Sivashinsky equation on L = 39 in the subspace of odd fields,
# u(-x) = -u(x), invariant under the reflection u(x) → -u(-x) of the equation.
#
#     julia --project=examples examples/kuramoto_sivashinsky/symmetric.jl
#
# writes symmetric.png in the folder figures/ next to this file, the search in the odd subspace. In
# the odd subspace translations are not allowed, so the orbits are periodic, without drift, and the
# search restricts itself to the subspace with the keyword `project` of System and the projection
# odd! of ks.jl. The initial guess is a near-recurrence of a chaotic trajectory integrated in the
# subspace. For comparison, the same search is repeated in the full space, as a periodic orbit
# without the projection, and as a relative periodic orbit with the drift speed as an unknown; their
# costs are printed.

using Random
using Printf
using CairoMakie

include("ks.jl")

Random.seed!(2)


# ---- chaotic trajectory in the odd subspace: a random sine series, transient of 200, then 800
#      time units, kept odd at every step ----
L  = 39
Δt = 0.25
gt = KSGrid(L, 65, 97)
xs = xpoints(gt)
u₀ = sum(0.1 * randn() .* sin.(2π * k .* xs ./ L) for k in 1:6)
U  = integrate(u₀, gt, 0.05, 4000; every=4000, odd=true)
U  = integrate(U[:, end], gt, 0.05, 16000; every=5, odd=true)

# ---- initial guess: the best near-recurrence with period in [20, 30], without shift ----
e, i, m, _ = recurrence(U, gt, Δt, (20, 30); drift=false)

g  = KSGrid(L, 65, 73)
a₀ = odd!(initial_orbit(U, g, Δt, i, m, 0.0; drift=false).a)
ρ₀ = log(2π / (m * Δt))

@printf "initial guess: recurrence error %.3f, T = %.2f\n" e m * Δt

# ---- three searches from the same guess ----
ops   = (KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!)
cases = (("odd subspace, projected", Orbit(copy(a₀), [ρ₀]), (), odd!),
         ("full space, no drift", Orbit(copy(a₀), [ρ₀]), (), identity),
         ("full space, drift", Orbit(copy(a₀), [ρ₀, 0.0]), (ddx!,), identity))

# distance of a field from the odd subspace, relative
asymmetry(a) = norm(a .- odd!(copy(a))) / norm(a)

runs = map(cases) do (name, x, ddi, project)
    F     = System(ops..., x; linearise!, ddi, B=KSPreconditioner(x; kind=:jacobian), project)
    trace = Trace()
    asym  = Float64[]

    # the trace, and the asymmetry of the orbit at every iteration
    record = info -> (push!(asym, asymmetry(info.x.a)); trace(info))

    solve!(x, F, NewtonHookstep(maxiter=30, krylov_dim=200, Δ=0.1, Δmax=10, tol=1e-10,
                                verbose=false, callback=record))

    T = 2π / ReSolver.frequency(x)
    @printf "%-26s ‖r‖ = %.1e in %d Newton iterations, %3d Jacobian actions; T = %.6f, asymmetry %.1e\n" name trace.res[end] length(trace) - 1 trace.evaluations[end].jacobian T asym[end]

    (; name, x, trace, asym)
end


# ============================================================================ #
# Figure                                                                       #
# ============================================================================ #

# Logarithmic ticks on an axis of ax: major ticks at integer powers of ten, labelled 10ⁿ, every
# decade or every third decade over a wide range; minor ticks at 2–9 times the powers of ten, or at
# every decade over a wide range.
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

fig = Figure(size=(1300, 820), fontsize=14)
run = runs[1]   # the search in the odd subspace
tr  = run.trace

# ---- top: initial guess and converged orbit, odd about x = 0 and x = L/2 ----
top = fig[1, 1:3] = GridLayout()
for (j, (title, a, T)) in enumerate((("initial guess", a₀, m * Δt),
                                     ("converged orbit", run.x.a, 2π / ReSolver.frequency(run.x))))
    t  = (0:g.Ns - 1) .* (T / g.Ns)
    ax = Axis(top[1, j]; title, xlabel="x", ylabel="t")
    hm = heatmap!(ax, xpoints(g), t, a.data; colormap=:balance, colorrange=(-3, 3))
    vlines!(ax, L / 2; color=:black, linestyle=:dash, linewidth=1)
    j == 2 && Colorbar(top[1, 3], hm; label="u")
end

# ---- bottom left: Newton iterations; open symbols: hooksteps, filled: full Newton steps ----
its = 0:length(tr) - 1
ax  = Axis(fig[2, 1]; title="Newton iterations, odd subspace", xlabel="iteration", ylabel="‖r‖",
           yscale=log10)
lines!(ax, its, tr.res; color=:crimson)
for (kind, fill) in (("", :crimson), ("hook", :white), ("newton", :crimson))
    sel = tr.step .== kind
    scatter!(ax, its[sel], tr.res[sel]; color=fill, strokecolor=:crimson, strokewidth=1.2,
             markersize=9)
end
logticks!(ax, minimum(tr.res), maximum(tr.res); axis=:y)

# ---- bottom centre: inner solves, one curve per Newton iteration, from dark to light ----
ax    = Axis(fig[2, 2]; title="GMRES, one curve per Newton iteration", xlabel="Arnoldi step",
             ylabel="‖g - Hy‖ / β", yscale=log10)
niter = length(tr) - 1
for j in 1:niter
    k = tr.krylov[j + 1]
    lines!(ax, 1:length(k), k; color=j, colormap=:viridis, colorrange=(1, niter))
end
hlines!(ax, 1e-3; color=:black, linestyle=:dash)
ylims!(ax, 1e-4, 2)
logticks!(ax, 1e-4, 1; axis=:y)
Colorbar(fig[2, 3]; colormap=:viridis, limits=(1, niter), label="Newton iteration")

save(joinpath(@__DIR__, "figures", "symmetric.png"), fig)
