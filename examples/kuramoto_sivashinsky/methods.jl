# L-BFGS against the Newton–Krylov hookstep for the orbit of the Kuramoto–Sivashinsky equation of
# example.jl.
#
#     julia --project=examples examples/kuramoto_sivashinsky/methods.jl
#
# writes methods.png, memory.png and hybrid.png in the folder figures/ next to this file.
#
# The cost of the two methods is compared in operator applications, one nonlinear, linearised or
# adjoint operator each, and in wall-clock time. The first measure is fair only if the three
# operators cost about the same, which is checked first. Two starts are compared, the
# near-recurrence of example.jl, far from the solution, and the converged orbit perturbed as in
# convergence.jl, close to it, each on the base grid and on a grid refined four times.
# L-BFGS runs with the linear and with the jacobian preconditioner, the hookstep with the jacobian
# one.
#
# methods.png: residual against the operator applications and the time, for the four searches.
#
# memory.png: the search from the near-recurrence with a limited number M of stored orbits: M/2
# curvature pairs for L-BFGS, a Krylov space of M - 1 vectors for the hookstep.
#
# hybrid.png: the far start on the base grid, with L-BFGS until ‖r‖ < r_T and the hookstep from
# there, for a few thresholds r_T, against the two methods alone.

using Random
using Printf
using CairoMakie

include("ks.jl")

import ReSolver: residual!, gradient!, jacobian!

Random.seed!(1)


# ============================================================================ #
# Initial guess and converged orbit, as in example.jl                          #
# ============================================================================ #

g  = KSGrid(22, 33, 49)
Δt = 0.25
u₀ = 0.1 .* randn(g.Nx)
u₀ .-= sum(u₀) / g.Nx
U  = integrate(u₀, g, 0.05, 4000; every=4000)
U  = integrate(U[:, end], g, 0.05, 8000; every=5)

e, i, m, ℓ = recurrence(U, g, Δt, (12, 20))
x₀         = initial_orbit(U, g, Δt, i, m, ℓ)

# the system on the grid of the orbit y, with a preconditioner of the given kind
system(y::Orbit, kind::Symbol) =
    System(KSNonlinear(y.a.g), KSLinearised(y.a.g), KSLinearised(y.a.g; adjoint=true), dds!, y;
           linearise!, ddi=(ddx!,), B=KSPreconditioner(y; kind))

x★ = copy(x₀)
solve!(x★, system(x★, :jacobian), NewtonHookstep(maxiter=50, krylov_dim=150, Δ=0.1, Δmax=10,
                                                 tol=1e-12, verbose=false))

@printf "orbit: T = %.4f\n" 2π / ReSolver.frequency(x★)


# ============================================================================ #
# Cost of the operators                                                        #
# ============================================================================ #

# seconds per call of f(), the best of five batches of n calls, after a call to compile
function seconds(f::Function; n::Int=200)
    f()
    return minimum(@elapsed(for _ in 1:n; f(); end) for _ in 1:5) / n
end

# the operators on a grid refined f times, about the converged orbit
for f in (1, 4)
    gf  = KSGrid(22, 32f + 1, 48f + 1)
    y   = Orbit(resample(x★.a, gf), copy(x★.p))
    v   = Orbit(resample(x★.a, gf), copy(x★.p))
    out = similar(y)
    Fy  = system(y, :jacobian)
    nl  = KSNonlinear(gf)
    lin = linearise!(KSLinearised(gf), y.a)
    adj = linearise!(KSLinearised(gf; adjoint=true), y.a)

    @printf "%3d × %3d: nonlinear %.1f μs, linearised %.1f μs, adjoint %.1f μs\n" gf.Nx gf.Ns 1e6 * seconds(() -> nl(out.a, y.a)) 1e6 * seconds(() -> lin(out.a, v.a)) 1e6 * seconds(() -> adj(out.a, v.a))
    @printf "           residual %.1f μs, gradient %.1f μs, Jacobian action %.1f μs, preconditioner %.1f μs\n" 1e6 * seconds(() -> residual!(out, Fy, y)) 1e6 * seconds(() -> gradient!(out, Fy, y)) 1e6 * seconds(() -> jacobian!(out, Fy, y, v)) 1e6 * seconds(() -> precondition!(out, Fy.B, v))
end


# ============================================================================ #
# Searches                                                                     #
# ============================================================================ #

# ---- two starts, each on the base grid and on a grid refined four times: far, the near-recurrence
#      of example.jl; close, the converged orbit perturbed as in convergence.jl ----
δ = resample(KSField(KSGrid(22, 7, 7), randn(7, 7)), g)
δ.data .*= 0.01 * norm(x★.a) / norm(δ)

far(gf)   = Orbit(resample(x₀.a, gf), copy(x₀.p))
close(gf) = Orbit(resample(x★.a, gf) .+ resample(δ, gf), x★.p .+ 0.01)

# operator applications: the nonlinear operator N, once per residual, the adjoint operator L⁺, once
# per gradient, and the linearised operator L, once per Jacobian action
applications(e::NamedTuple) = e.residual + e.gradient + e.jacobian

# operator applications and time at which the residual of a trace first falls below each level;
# NaN if it never does
function reached(trace::Trace, levels)
    out = map(levels) do level
        j = findfirst(<(level), trace.res)
        j === nothing ? (NaN, NaN) : (applications(trace.evaluations[j]), trace.time[j])
    end
    return join([@sprintf("‖r‖ < %.0e: %6.0f, %7.3f s", l, n, t) for (l, (n, t)) in zip(levels, out)], ";  ")
end

# a trace that stops the search after tmax seconds
function stopper(trace::Trace, tmax::Real)
    return info -> (trace(info); trace.time[end] > tmax)
end

# one search with a method and a preconditioner, from y, for at most tmax seconds; the method is
# run once briefly first, so that the time does not include compilation
function search(y::Orbit, method::Symbol, kind::Symbol, tol::Real, tmax::Real;
                memory::Int=10, krylov_dim::Int=150)
    F     = system(y, kind)
    trace = Trace()

    make(maxiter, callback) = method === :lbfgs ?
        LBFGS(; memory, maxiter, tol, verbose=false, callback) :
        NewtonHookstep(; krylov_dim, maxiter, tol, Δ=0.1, Δmax=10, verbose=false, callback)

    solve!(copy(y), F, make(2, info -> false))
    F = system(y, kind)
    solve!(copy(y), F, make(100_000, stopper(trace, tmax)))

    return trace
end

g₁ = KSGrid(22, 33, 49)
g₄ = KSGrid(22, 129, 193)

# name, initial orbit, tolerance and time limit: the comparison stops at 10⁻⁶, a residual small
# enough for the orbit to be identified, which L-BFGS reaches in a reasonable time
starts = (("far start, 33 × 49 (1 619 unknowns)", far(g₁), 1e-6, 20.0),
          ("close start, 33 × 49 (1 619 unknowns)", close(g₁), 1e-6, 20.0),
          ("far start, 129 × 193 (24 899 unknowns)", far(g₄), 1e-6, 60.0),
          ("close start, 129 × 193 (24 899 unknowns)", close(g₄), 1e-6, 60.0))

# method and preconditioner: linear, B = 1 + |Â₀|; jacobian, B = Â₀, with Â₀ the symbol of the
# linear space-time operator about u = 0 (ks.jl)
runs   = (("L-BFGS, B = 1 + |Â₀| (linear)", :lbfgs, :linear),
          ("L-BFGS, B = Â₀ (jacobian)", :lbfgs, :jacobian),
          ("hookstep, B = Â₀ (jacobian)", :hookstep, :jacobian))
colors = Dict(runs[1][1] => :royalblue, runs[2][1] => :seagreen, runs[3][1] => :crimson)

traces = Dict()
for (sname, y, tol, tmax) in starts, (rname, method, kind) in runs
    trace = search(y, method, kind, tol, tmax)
    traces[(sname, rname)] = trace

    @printf "%-42s %-30s ‖r‖ = %.1e → %.1e in %5d iterations, %6d operator applications, %6.2f s\n" sname rname trace.res[1] trace.res[end] length(trace) - 1 applications(trace.evaluations[end]) trace.time[end]
    println("    ", reached(trace, (1e-3, 1e-6, tol)))
    flush(stdout)
end


# ============================================================================ #
# Limited memory                                                               #
# ============================================================================ #

budgets = (5, 10, 20, 50, 100, 200)
memtr   = Dict()
for M in budgets
    memtr[(M, :lbfgs)]    = search(far(g₁), :lbfgs, :linear, 1e-6, 20.0; memory=M ÷ 2)
    memtr[(M, :hookstep)] = search(far(g₁), :hookstep, :jacobian, 1e-6, 20.0; krylov_dim=M - 1)

    for method in (:lbfgs, :hookstep)
        t = memtr[(M, method)]
        @printf "M = %3d, %-8s ‖r‖ = %.1e in %5d iterations, %6d operator applications, %6.2f s\n" M method t.res[end] length(t) - 1 applications(t.evaluations[end]) t.time[end]
    end
    flush(stdout)
end


# ============================================================================ #
# Hybrid: L-BFGS until ‖r‖ < r_T, then the hookstep                            #
# ============================================================================ #

# the hybrid search from y: L-BFGS with the linear preconditioner, the better metric for L-BFGS,
# stopped at ‖r‖ < rT, then the hookstep with the jacobian one, the better for the Newton
# systems. The two phases use two systems, whose counters are added in a single trace; both
# methods are run once briefly first, so that the time does not include compilation
function hybrid(y::Orbit, rT::Real, tol::Real, tmax::Real)
    solve!(copy(y), system(y, :linear), LBFGS(maxiter=2, verbose=false))
    solve!(copy(y), system(y, :jacobian), NewtonHookstep(maxiter=2, verbose=false))

    x     = copy(y)
    trace = Trace()
    F₁    = system(y, :linear)
    solve!(x, F₁, LBFGS(maxiter=1_000_000, tol=rT, verbose=false, callback=stopper(trace, tmax)))
    nl    = length(trace)

    # the second phase continues the counts and the clock of the first
    e₁   = trace.evaluations[end]
    t₀   = trace.t₀[]
    F₂   = system(x, :jacobian)
    more = info -> (e = info.evaluations;
                    trace(merge(info, (evaluations = (residual     = e.residual + e₁.residual,
                                                      gradient     = e.gradient + e₁.gradient,
                                                      jacobian     = e.jacobian + e₁.jacobian,
                                                      precondition = e.precondition + e₁.precondition),)));
                    trace.t₀[] = t₀;
                    trace.time[end] > tmax)
    solve!(x, F₂, NewtonHookstep(; krylov_dim=150, maxiter=1000, tol, Δ=0.1, Δmax=10, verbose=false,
                                 callback=more))

    return trace, nl
end

thresholds = (3e-2, 2e-2, 1e-2)
hybrids    = Dict()
for rT in thresholds
    hybrids[rT] = hybrid(far(g₁), rT, 1e-6, 20.0)

    t, nl = hybrids[rT]
    @printf "hybrid, r_T = %.0e: ‖r‖ = %.1e, %4d L-BFGS and %3d Newton iterations, %5d operator applications, %5.3f s\n" rT t.res[end] nl - 1 length(t) - nl - 1 applications(t.evaluations[end]) t.time[end]
end


# ============================================================================ #
# Figures                                                                      #
# ============================================================================ #

# Logarithmic ticks at integer powers of ten, with minor ticks at 2–9 times them; every third
# decade over a wide range.
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

# a logarithmic axis spanning whole decades around vals
function decades!(ax, vals; axis::Symbol=:x)
    vals = filter(v -> isfinite(v) && v > 0, vals)
    lo   = 10.0^floor(log10(minimum(vals)))
    hi   = 10.0^ceil(log10(maximum(vals)))
    axis === :x ? xlims!(ax, lo, hi) : ylims!(ax, lo, hi)
    logticks!(ax, lo, hi; axis)
    return ax
end

# ---- methods.png: one row per start, residual against applications and time ----
fig = Figure(size=(1100, 1550), fontsize=13)

for (row, (sname, _, tol, _)) in enumerate(starts)
    ax1 = Axis(fig[row, 1]; title=sname, xlabel="operator applications (each N, L or L⁺ counts 1)", ylabel="‖r‖",
               xscale=log10, yscale=log10)
    ax2 = Axis(fig[row, 2]; title=sname, xlabel="time [s]", ylabel="‖r‖", xscale=log10,
               yscale=log10)

    xs, ts, rs = Float64[], Float64[], Float64[]
    for (rname, _, _) in runs
        tr = traces[(sname, rname)]
        n  = applications.(tr.evaluations)
        t  = max.(tr.time, 1e-4)

        lines!(ax1, max.(n, 1), tr.res; color=colors[rname], label=rname)
        lines!(ax2, t, tr.res; color=colors[rname], label=rname)
        append!(xs, max.(n, 1)); append!(ts, t); append!(rs, tr.res)
    end
    hlines!(ax1, tol; color=:black, linestyle=:dash)
    hlines!(ax2, tol; color=:black, linestyle=:dash)

    decades!(ax1, xs); decades!(ax2, ts)
    decades!(ax1, [rs; tol]; axis=:y); decades!(ax2, [rs; tol]; axis=:y)
    linkyaxes!(ax1, ax2)
    row == 1 && axislegend(ax1; position=:lb)
end

save(joinpath(@__DIR__, "figures", "methods.png"), fig)

# ---- memory.png: residual against time for each budget, and the cost against the budget ----
fig = Figure(size=(1300, 420), fontsize=13)
cmap = cgrad(:viridis, length(budgets); categorical=true)

axes = [Axis(fig[1, j]; title, xlabel="time [s]", ylabel="‖r‖", xscale=log10, yscale=log10)
        for (j, title) in enumerate(("L-BFGS, B = 1 + |Â₀|, M/2 curvature pairs",
                                     "hookstep, B = Â₀, M - 1 Krylov vectors"))]
ts, rs = Float64[], Float64[]
for (k, M) in enumerate(budgets), (ax, method) in zip(axes, (:lbfgs, :hookstep))
    tr = memtr[(M, method)]
    t  = max.(tr.time, 1e-4)
    lines!(ax, t, tr.res; color=cmap[k], label="M = $M")
    append!(ts, t); append!(rs, tr.res)
end
for ax in axes
    hlines!(ax, 1e-6; color=:black, linestyle=:dash)
    decades!(ax, ts); decades!(ax, [rs; 1e-6]; axis=:y)
end
linkyaxes!(axes...)
axislegend(axes[1]; position=:lb)
axislegend(axes[2]; position=:lb)

# time to reach the tolerance against the budget; missing if not reached
ax = Axis(fig[1, 3]; title="time to ‖r‖ < 10⁻⁶", xlabel="stored orbits M", ylabel="time [s]",
          xscale=log10, yscale=log10)
ys = Float64[]
for (method, color, label) in ((:lbfgs, :royalblue, "L-BFGS, B = 1 + |Â₀|"),
                               (:hookstep, :crimson, "hookstep, B = Â₀"))
    local y = [memtr[(M, method)].res[end] < 1e-6 ? memtr[(M, method)].time[end] : NaN for M in budgets]
    println(method, ": time to the tolerance against M: ", y)
    scatterlines!(ax, collect(budgets), y; color, label, markersize=9)
    append!(ys, y)
end
logticks!(ax, 1, 1000)
xlims!(ax, 3, 300)
decades!(ax, ys; axis=:y)
# the lower left corner is empty: no method reaches the tolerance with the smallest budgets fast
axislegend(ax; position=:lb)

save(joinpath(@__DIR__, "figures", "memory.png"), fig)

# ---- hybrid.png: far start on the base grid, L-BFGS, hookstep and the hybrid ----
fig  = Figure(size=(1100, 430), fontsize=13)
hmap = cgrad(:plasma, length(thresholds) + 1; categorical=true)
sname, _, tol, _ = starts[1]

ax1 = Axis(fig[1, 1]; title=sname, xlabel="operator applications (each N, L or L⁺ counts 1)",
           ylabel="‖r‖", xscale=log10, yscale=log10)
ax2 = Axis(fig[1, 2]; title=sname, xlabel="time [s]", ylabel="‖r‖", xscale=log10, yscale=log10)

curves = Any[(rname, traces[(sname, rname)], colors[rname], :solid) for (rname, _, _) in runs]
for (k, rT) in enumerate(thresholds)
    push!(curves, ("hybrid, r_T = $rT", first(hybrids[rT]), hmap[k], :dash))
end

xs, ts, rs = Float64[], Float64[], Float64[]
for (label, tr, color, linestyle) in curves
    n = max.(applications.(tr.evaluations), 1)
    t = max.(tr.time, 1e-4)
    lines!(ax1, n, tr.res; color, linestyle, label)
    lines!(ax2, t, tr.res; color, linestyle, label)
    append!(xs, n); append!(ts, t); append!(rs, tr.res)
end
hlines!(ax1, tol; color=:black, linestyle=:dot)
hlines!(ax2, tol; color=:black, linestyle=:dot)

decades!(ax1, xs); decades!(ax2, ts)
decades!(ax1, [rs; tol]; axis=:y); decades!(ax2, [rs; tol]; axis=:y)
linkyaxes!(ax1, ax2)
Legend(fig[1, 3], ax1; labelsize=11, framevisible=false)

save(joinpath(@__DIR__, "figures", "hybrid.png"), fig)
