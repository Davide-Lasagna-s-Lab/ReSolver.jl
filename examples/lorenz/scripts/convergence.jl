# Convergence of the Newton–Krylov hookstep on the shortest periodic orbit of the Lorenz system, at
# increasing resolution in time, without and with the preconditioner.
#
#     julia --project=examples examples/lorenz/scripts/convergence.jl
#
# writes convergence.png and tolerance.png in docs/src/assets/lorenz/. The orbit is
# first converged with K = 20 modes; it is then resampled to K = 10, 20, 40 and 80 modes, its
# log-frequency perturbed by 0.01 and its field by 1% on the lowest modes, so that every search
# starts close to the solution with a residual spread over many directions.
#
# convergence.png: searches with the Newton system solved to a relative residual of 10⁻³, showing
# the residual against the Newton iterations, the Jacobian actions and the computing time, and the
# relative residual of the Newton system after each Arnoldi step of the first three iterations.
#
# tolerance.png: the cost of bringing the residual below 10⁻¹², in Jacobian actions, time and Newton
# iterations, against the tolerance on the Newton system, from 0.95 to 10⁻¹⁰.

using Printf
using Random
using CairoMakie

include("../lorenz.jl")


# ============================================================================ #
# Converged orbit with K = 20 modes                                            #
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

@printf "orbit: T = %.8f\n" 2π / ReSolver.frequency(x)


# ============================================================================ #
# Searches at increasing resolution                                            #
# ============================================================================ #

# ---- perturbation: 1% of the orbit on its five lowest modes, the same at every resolution ----
Random.seed!(2)
δ = LorenzField(5)
δ.data .= randn(ComplexF64, 3, 6)
δ.data[:, 1] .= real.(δ.data[:, 1])
δ.data .*= 0.01 * norm(x.a) / norm(δ)

Ks       = (10, 20, 40, 80)
names    = ("no preconditioner", "preconditioner")
tol      = 1e-12
start(K) = Orbit(resample(x.a, K) .+ resample(δ, K), x.p .+ 0.01)

# the hookstep from the perturbed orbit with K modes, the Newton system solved to krylov_tol
function search(K::Int, name::String, krylov_tol::Real; maxiter::Int)
    xK = start(K)
    B  = name == "preconditioner" ? LorenzPreconditioner(xK) : I
    FK = System(nonlinear!, LorenzLinearised(K), LorenzLinearised(K; adjoint=true), dds!, xK;
                linearise!, B)

    trace = Trace()
    solve!(copy(xK), FK, NewtonHookstep(; maxiter, krylov_dim=200, krylov_tol, Δ=10, Δmax=100, tol,
                                        verbose=false, callback=trace))

    return trace
end

# ---- searches with the Newton system solved to 10⁻³ ----
runs = []

for K in Ks, name in names
    trace = search(K, name, 1e-3; maxiter=12)

    @printf "K = %2d, %-18s ‖r‖ = %.1e → %.1e in %2d iterations, %5d Jacobian actions, %.2f s\n" K name trace.res[1] trace.res[end] length(trace) - 1 trace.evaluations[end].jacobian trace.time[end]
    flush(stdout)

    push!(runs, (; K, name, trace))
end

# ---- cost of ‖r‖ < 10⁻¹² against the tolerance on the Newton system; NaN if not reached ----
tols = [0.95; 0.9:-0.1:0.2; 10.0 .^ (-1:-1:-10)]
cost = Dict()

# without preconditioner the searches with K > 20 do not converge: they are left out
for K in Ks, name in names, τ in tols
    (name == "no preconditioner" && K > 20) && continue

    trace = search(K, name, τ; maxiter=200)
    done  = trace.res[end] < tol

    cost[(K, name, τ)] = (jacobian = done ? trace.evaluations[end].jacobian : NaN,
                          time     = done ? trace.time[end] : NaN,
                          newton   = done ? length(trace) - 1 : NaN)
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

colors  = Makie.wong_colors()[1:length(Ks)]
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
    color  = colors[findfirst(==(run.K), Ks)]
    marker = markers[findfirst(==(run.K), Ks)]
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
       [["K = $K" for K in Ks]; collect(keys(style))];
       orientation=:horizontal, framevisible=false)

save(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "lorenz", "convergence.png"), fig)


# ============================================================================ #
# Figure: cost against the tolerance on the Newton system                      #
# ============================================================================ #

fig = Figure(size=(1300, 470), fontsize=13)

panels = ((:jacobian, "Jacobian actions", "Jacobian actions", log10),
          (:time, "time", "time [s]", log10),
          (:newton, "Newton iterations", "Newton iterations", log10))
axc    = [Axis(fig[1, j]; title="cost of ‖r‖ < 10⁻¹²: " * title,
               xlabel="tolerance on the Newton system", ylabel, xscale=log10, yscale,
               xreversed=true)
          for (j, (_, title, ylabel, yscale)) in enumerate(panels)]

for K in Ks, name in names
    haskey(cost, (K, name, first(tols))) || continue
    color  = colors[findfirst(==(K), Ks)]
    marker = markers[findfirst(==(K), Ks)]
    for (ax, (field, _, _, _)) in zip(axc, panels)
        ys = [getfield(cost[(K, name, τ)], field) for τ in tols]
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
       [[[LineElement(color=c), MarkerElement(color=c, marker=m, markersize=9)] for (c, m) in zip(colors, markers)];
        [LineElement(color=:black, linestyle=style[n]) for n in keys(style)]],
       [["K = $K" for K in Ks]; collect(keys(style))];
       orientation=:horizontal, framevisible=false)

save(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "lorenz", "tolerance.png"), fig)
