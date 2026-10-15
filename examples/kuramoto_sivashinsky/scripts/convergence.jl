# Convergence of the Newton–Krylov hookstep on the orbit of example.jl, at
# increasing resolution, without and with the preconditioner.
#
#     julia --project=examples examples/kuramoto_sivashinsky/scripts/convergence.jl
#
# writes convergence.png and tolerance.png in docs/src/assets/kuramoto_sivashinsky/. The orbit is
# first converged at the resolution of the example; it is then interpolated to grids refined by 1,
# 2, 4 and 8 in both space and rescaled time, its log-frequency and drift speed perturbed by 0.01
# and its field by 1% on the lowest modes, so that every search starts close to the solution with a
# residual spread over many directions.
#
# convergence.png: searches with the Newton system solved to a relative residual of 10⁻³, showing
# the residual against the Newton iterations, the Jacobian actions and the computing time, and the
# relative residual of the Newton system after each Arnoldi step of the first three iterations.
#
# tolerance.png: the cost of bringing the residual below 10⁻⁹, in Jacobian actions, time and Newton
# iterations, against the tolerance on the Newton system, from 0.95 to 10⁻¹⁰, with the
# preconditioner.

using Random
using Printf
using CairoMakie

include("../ks.jl")

Random.seed!(1)


# ============================================================================ #
# Converged orbit at the resolution of the example                             #
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


# ============================================================================ #
# Searches at increasing resolution                                            #
# ============================================================================ #

# ---- perturbation: 1% of the orbit on its lowest modes, |k| ≤ 3 and |n| ≤ 3, the same at every
#      resolution ----
δ = resample(KSField(KSGrid(22, 7, 7), randn(7, 7)), g)
δ.data .*= 0.01 * norm(x.a) / norm(δ)

factors  = (1, 2, 4, 8)
names    = ("no preconditioner", "preconditioner")
tol      = 1e-9
grid(f)  = KSGrid(22, 32f + 1, 48f + 1)
start(f) = Orbit(resample(x.a, grid(f)) .+ resample(δ, grid(f)), x.p .+ 0.01)

# the hookstep from the perturbed orbit on the grid refined by f, the Newton system solved to
# krylov_tol
function search(f::Int, name::String, krylov_tol::Real; maxiter::Int)
    gf = grid(f)
    xf = start(f)
    B  = name == "preconditioner" ? KSPreconditioner(xf) : I
    Ff = System(KSNonlinear(gf), KSLinearised(gf), KSLinearised(gf; adjoint=true), dds!, xf;
                linearise!, ddi=(ddx!,), B, project=demean!)

    trace = Trace()
    solve!(copy(xf), Ff, NewtonHookstep(; maxiter, krylov_dim=200, krylov_tol, Δ=10, Δmax=100, tol,
                                        verbose=false, callback=trace))

    return trace
end

# ---- searches with the Newton system solved to 10⁻³ ----
runs = []

for f in factors, name in names
    trace = search(f, name, 1e-3; maxiter=12)
    gf    = grid(f)

    @printf "%3d × %3d, %-18s ‖r‖ = %.1e → %.1e in %2d iterations, %5d Jacobian actions, %.1f s\n" gf.Nx gf.Ns name trace.res[1] trace.res[end] length(trace) - 1 trace.evaluations[end].jacobian trace.time[end]
    flush(stdout)

    push!(runs, (; f, size=(gf.Nx, gf.Ns), name, trace))
end

# ---- cost of ‖r‖ < 10⁻⁹ against the tolerance on the Newton system; NaN if not reached. Below
#      10⁻⁹ the residual is limited by rounding errors amplified by the fourth derivative ----
tols = [0.95; 0.9:-0.1:0.2; 10.0 .^ (-1:-1:-10)]
cost = Dict()

# only with the preconditioner: without, the hookstep converges at no tolerance, even on the
# coarsest grid, because the Krylov spaces of 200 vectors never reach it
for f in factors, τ in tols
    name  = "preconditioner"
    trace = search(f, name, τ; maxiter=200)
    done  = trace.res[end] < tol

    cost[(f, name, τ)] = (jacobian = done ? trace.evaluations[end].jacobian : NaN,
                          time     = done ? trace.time[end] : NaN,
                          newton   = done ? length(trace) - 1 : NaN)

    @printf "  tolerance %.2g, %3d × %3d: %s, %5.0f Jacobian actions, %3.0f Newton iterations, %6.2f s\n" τ grid(f).Nx grid(f).Ns done ? "converged    " : "not converged" cost[(f, name, τ)].jacobian cost[(f, name, τ)].newton cost[(f, name, τ)].time
    flush(stdout)
end


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

colors  = Makie.wong_colors()[1:length(factors)]
markers = (:circle, :rect, :utriangle, :diamond)
style  = Dict("no preconditioner" => :dash, "preconditioner" => :solid)

fig = Figure(size=(1300, 820), fontsize=13)

# ---- top: residual against Newton iterations, Jacobian actions and time ----
labels = ("Newton iteration", "Jacobian actions", "time [s]")
titles = ("Newton convergence", "cost: Jacobian actions", "cost: time")
top    = [Axis(fig[1, j]; title, xlabel, ylabel=j == 1 ? "‖r‖" : "", yscale=log10,
               xscale=j == 1 ? identity : log10)
          for (j, (title, xlabel)) in enumerate(zip(titles, labels))]

# ---- bottom: inner solve of the first three Newton iterations ----
bottom = [Axis(fig[2, j]; title="inner solve, Newton iteration $j", xlabel="Arnoldi step",
               ylabel=j == 1 ? "‖g - Hy‖ / β" : "", yscale=log10)
          for j in 1:3]

for run in runs
    color  = colors[findfirst(==(run.f), factors)]
    marker = markers[findfirst(==(run.f), factors)]
    ls     = style[run.name]
    t     = run.trace
    jac   = [e.jacobian for e in t.evaluations] .- t.evaluations[1].jacobian

    # the start has no Jacobian action and no time: it is left out of the logarithmic axes
    scatterlines!(top[1], 0:length(t) - 1, t.res; color, marker, linestyle=ls, markersize=7)
    scatterlines!(top[2], jac[2:end], t.res[2:end]; color, marker, linestyle=ls, markersize=7)
    scatterlines!(top[3], t.time[2:end], t.res[2:end]; color, marker, linestyle=ls, markersize=7)

    # relative residual of the Newton system after each Arnoldi step, iterations 1 to 3
    for (j, ax) in enumerate(bottom)
        j < length(t) || continue
        k = t.krylov[j + 1]
        lines!(ax, 1:length(k), k; color, linestyle=ls)
    end
end

# ---- logarithmic ticks ----
res = reduce(vcat, [run.trace.res for run in runs])
jac = reduce(vcat, [[e.jacobian for e in run.trace.evaluations[2:end]] for run in runs])
tim = reduce(vcat, [run.trace.time[2:end] for run in runs])

for ax in top
    logticks!(ax, minimum(res), maximum(res); axis=:y)
end
# the Newton system is solved to 10⁻³: what lies below is not part of the search
for ax in bottom
    ylims!(ax, 1e-4, 2)
    logticks!(ax, 1e-4, 1; axis=:y)
end
# the cost axes span whole decades, so that every one has a labelled major tick
for (ax, vals) in zip(top[2:3], (jac, tim))
    lo = 10.0^floor(log10(minimum(vals)))
    hi = 10.0^ceil(log10(maximum(vals)))
    xlims!(ax, lo, hi)
    logticks!(ax, lo, hi)
end

# ---- the same residual axis across each row ----
linkyaxes!(top...)
linkyaxes!(bottom...)

# ---- legend ----
Legend(fig[3, 1:3],
       [[[LineElement(color=c), MarkerElement(color=c, marker=m, markersize=9)] for (c, m) in zip(colors, markers)];
        [LineElement(color=:black, linestyle=style[n]) for n in keys(style)]],
       [["$(r.size[1]) × $(r.size[2])" for r in runs if r.name == "preconditioner"]; collect(keys(style))];
       orientation=:horizontal, framevisible=false)

save(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "kuramoto_sivashinsky", "convergence.png"), fig)


# ============================================================================ #
# Figure: cost against the tolerance on the Newton system                      #
# ============================================================================ #

fig = Figure(size=(1300, 470), fontsize=13)

panels = ((:jacobian, "Jacobian actions", "Jacobian actions", log10),
          (:time, "time", "time [s]", log10),
          (:newton, "Newton iterations", "Newton iterations", log10))
axc    = [Axis(fig[1, j]; title="cost of ‖r‖ < 10⁻⁹: " * title,
               xlabel="tolerance on the Newton system", ylabel, xscale=log10, yscale,
               xreversed=true)
          for (j, (_, title, ylabel, yscale)) in enumerate(panels)]

for f in factors, name in names
    haskey(cost, (f, name, first(tols))) || continue
    color  = colors[findfirst(==(f), factors)]
    marker = markers[findfirst(==(f), factors)]
    for (ax, (field, _, _, _)) in zip(axc, panels)
        ys = [getfield(cost[(f, name, τ)], field) for τ in tols]
        scatterlines!(ax, tols, ys; color, marker, linestyle=style[name], markersize=8)
    end
end

for ax in axc
    logticks!(ax, minimum(tols), maximum(tols))
end
# the logarithmic axes span whole decades, so that every one has a labelled major tick
for (ax, field) in zip(axc, (:jacobian, :time, :newton))
    vals = filter(isfinite, [getfield(c, field) for c in values(cost)])
    lo   = 10.0^floor(log10(minimum(vals)))
    hi   = 10.0^ceil(log10(maximum(vals)))
    ylims!(ax, lo, hi)
    logticks!(ax, lo, hi; axis=:y)
end

Legend(fig[2, 1:3],
       [[LineElement(color=c), MarkerElement(color=c, marker=m, markersize=9)] for (c, m) in zip(colors, markers)],
       ["$(grid(f).Nx) × $(grid(f).Ns), preconditioner" for f in factors];
       orientation=:horizontal, framevisible=false)

save(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "kuramoto_sivashinsky", "tolerance.png"), fig)
