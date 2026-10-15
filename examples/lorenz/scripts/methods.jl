# L-BFGS against the Newton–Krylov hookstep for the shortest periodic orbit of the Lorenz system.
#
#     julia --project=examples examples/lorenz/scripts/methods.jl
#
# writes methods.png and memory.png in docs/src/assets/lorenz/.
#
# The cost of the two methods is compared in operator applications, one nonlinear, linearised or
# adjoint operator each, and in wall-clock time. The first measure is fair only if the three
# operators cost about the same, which is checked first. Two starts are compared: the converged
# orbit perturbed by 10% on its five lowest modes and by 0.05 in the log-frequency, far from the
# solution, and the orbit perturbed by 1% and 0.01, as in convergence.jl, close to it, each with
# K = 20 and K = 80 modes. Both methods use the jacobian preconditioner of lorenz.jl.
#
# methods.png: residual against the operator applications and the time, for the four searches.
#
# memory.png: the search from the far start with K = 40 and a limited number M of stored orbits: M/2
# curvature pairs for L-BFGS, a Krylov space of M - 1 vectors for the hookstep.

using Random
using Printf
using CairoMakie

include("../lorenz.jl")

import ReSolver: residual!, gradient!, jacobian!


# ============================================================================ #
# Converged orbit with K = 20 modes, as in convergence.jl                      #
# ============================================================================ #

dt = 0.005
U  = trajectory(trajectory([1.0, 1.0, 1.0], dt, 4000)[:, end], dt, 20000)

lags    = round(Int, 1.4 / dt):round(Int, 1.7 / dt)
e, i, m = minimum((norm(U[:, i + m] .- U[:, i]) / norm(U[:, i]), i, m)
                  for i in 1:size(U, 2) - last(lags), m in lags)

# the system for the orbit y, with the jacobian preconditioner
function system(y::Orbit)
    K = modes(y.a)
    return System(nonlinear!, LorenzLinearised(K), LorenzLinearised(K; adjoint=true), dds!, y;
                  linearise!, B=LorenzPreconditioner(y))
end

x★ = initial_orbit(U, dt, i, m, 20)
solve!(x★, system(x★), NewtonHookstep(maxiter=20, krylov_dim=40, Δ=1, Δmax=100, tol=1e-12,
                                      verbose=false))

@printf "orbit: T = %.8f\n" 2π / ReSolver.frequency(x★)


# ============================================================================ #
# Cost of the operators                                                        #
# ============================================================================ #

# seconds per call of f(), the best of five batches of n calls, after a call to compile
function seconds(f::Function; n::Int=2000)
    f()
    return minimum(@elapsed(for _ in 1:n; f(); end) for _ in 1:5) / n
end

K   = 40
y   = Orbit(resample(x★.a, K), copy(x★.p))
v   = Orbit(resample(x★.a, K), copy(x★.p))
out = similar(y)
Fy  = system(y)
lin = linearise!(LorenzLinearised(K), y.a)
adj = linearise!(LorenzLinearised(K; adjoint=true), y.a)

@printf "K = %d: nonlinear %.2f μs, linearised %.2f μs, adjoint %.2f μs\n" K 1e6 * seconds(() -> nonlinear!(out.a, y.a)) 1e6 * seconds(() -> lin(out.a, v.a)) 1e6 * seconds(() -> adj(out.a, v.a))
@printf "        residual %.2f μs, gradient %.2f μs, Jacobian action %.2f μs, preconditioner %.2f μs\n" 1e6 * seconds(() -> residual!(out, Fy, y)) 1e6 * seconds(() -> gradient!(out, Fy, y)) 1e6 * seconds(() -> jacobian!(out, Fy, y, v)) 1e6 * seconds(() -> precondition!(out, Fy.B, v))


# ============================================================================ #
# Searches                                                                     #
# ============================================================================ #

# ---- perturbation of the five lowest modes, of unit norm relative to the orbit ----
Random.seed!(2)
δ = LorenzField(5)
δ.data .= randn(ComplexF64, 3, 6)
δ.data[:, 1] .= real.(δ.data[:, 1])
δ.data .*= norm(x★.a) / norm(δ)

far(K)   = Orbit(resample(x★.a, K) .+ 0.10 .* resample(δ, K), x★.p .+ 0.05)
close(K) = Orbit(resample(x★.a, K) .+ 0.01 .* resample(δ, K), x★.p .+ 0.01)

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

# One search with a method, from y, for at most tmax seconds. The method is run once briefly first,
# with the same callback, so that no time is spent compiling; a search that reaches the tolerance
# is then run three times and the fastest kept, against the noise of the clock. Each run has a
# system of its own, whose counters start from zero.
function search(y::Orbit, method::Symbol, tol::Real, tmax::Real; memory::Int=10,
                krylov_dim::Int=100)
    make(maxiter, callback) = method === :lbfgs ?
        LBFGS(; memory, maxiter, tol, verbose=false, callback) :
        NewtonHookstep(; krylov_dim, maxiter, tol, Δ=1, Δmax=100, verbose=false, callback)

    # ---- compilation ----
    solve!(copy(y), system(y), make(2, stopper(Trace(), tmax)))

    # ---- timed runs ----
    best = Trace()
    for run in 1:3
        trace = Trace()
        solve!(copy(y), system(y), make(1_000_000, stopper(trace, tmax)))
        (run == 1 || trace.time[end] < best.time[end]) && (best = trace)
        trace.res[end] < tol || break
    end

    return best
end

# name, initial orbit, tolerance and time limit
starts = (("far start, K = 20 (124 unknowns)", far(20), 1e-12, 10.0),
          ("close start, K = 20 (124 unknowns)", close(20), 1e-12, 10.0),
          ("far start, K = 80 (484 unknowns)", far(80), 1e-12, 10.0),
          ("close start, K = 80 (484 unknowns)", close(80), 1e-12, 10.0))

# method; both with the jacobian preconditioner, B = iω₀n I - J(ū) on the mode n (lorenz.jl)
runs   = (("L-BFGS, B = iω₀n I - J(ū) (jacobian)", :lbfgs),
          ("hookstep, B = iω₀n I - J(ū) (jacobian)", :hookstep))
colors = Dict(runs[1][1] => :seagreen, runs[2][1] => :crimson)

traces = Dict()
for (sname, y, tol, tmax) in starts, (rname, method) in runs
    trace = search(y, method, tol, tmax)
    traces[(sname, rname)] = trace

    @printf "%-36s %-40s ‖r‖ = %.1e → %.1e in %6d iterations, %7d operator applications, %6.3f s\n" sname rname trace.res[1] trace.res[end] length(trace) - 1 applications(trace.evaluations[end]) trace.time[end]
    println("    ", reached(trace, (1e-3, 1e-6, tol)))
    flush(stdout)
end


# ============================================================================ #
# Limited memory                                                               #
# ============================================================================ #

budgets = (5, 10, 20, 50, 100, 200)
memtr   = Dict()
for M in budgets
    memtr[(M, :lbfgs)]    = search(far(40), :lbfgs, 1e-12, 10.0; memory=M ÷ 2)
    memtr[(M, :hookstep)] = search(far(40), :hookstep, 1e-12, 10.0; krylov_dim=M - 1)

    for method in (:lbfgs, :hookstep)
        t = memtr[(M, method)]
        @printf "M = %3d, %-8s ‖r‖ = %.1e in %6d iterations, %7d operator applications, %6.3f s\n" M method t.res[end] length(t) - 1 applications(t.evaluations[end]) t.time[end]
    end
    flush(stdout)
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
    for (rname, _) in runs
        tr = traces[(sname, rname)]
        n  = applications.(tr.evaluations)
        t  = clamp.(tr.time, 1e-4, 10)   # the searches stop after 10 s

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

save(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "lorenz", "methods.png"), fig)

# ---- memory.png: residual against time for each budget, and the cost against the budget ----
fig = Figure(size=(1300, 420), fontsize=13)
cmap = cgrad(:viridis, length(budgets); categorical=true)

axes = [Axis(fig[1, j]; title, xlabel="time [s]", ylabel="‖r‖", xscale=log10, yscale=log10)
        for (j, title) in enumerate(("L-BFGS, M/2 curvature pairs",
                                     "hookstep, M - 1 Krylov vectors"))]
ts, rs = Float64[], Float64[]
for (k, M) in enumerate(budgets), (ax, method) in zip(axes, (:lbfgs, :hookstep))
    tr = memtr[(M, method)]
    t  = clamp.(tr.time, 1e-4, 10)   # the searches stop after 10 s
    lines!(ax, t, tr.res; color=cmap[k], label="M = $M")
    append!(ts, t); append!(rs, tr.res)
end
for ax in axes
    hlines!(ax, 1e-12; color=:black, linestyle=:dash)
    decades!(ax, ts); decades!(ax, [rs; 1e-12]; axis=:y)
end
linkyaxes!(axes...)
axislegend(axes[1]; position=:lb)
axislegend(axes[2]; position=:lb)

# time to reach the tolerance against the budget; missing if not reached
ax = Axis(fig[1, 3]; title="time to ‖r‖ < 10⁻¹²", xlabel="stored orbits M", ylabel="time [s]",
          xscale=log10, yscale=log10)
ys = Float64[]
for (method, color, label) in ((:lbfgs, :seagreen, "L-BFGS"),
                               (:hookstep, :crimson, "hookstep"))
    local y = [memtr[(M, method)].res[end] < 1e-12 ? memtr[(M, method)].time[end] : NaN for M in budgets]
    println(method, ": time to the tolerance against M: ", y)
    scatterlines!(ax, collect(budgets), y; color, label, markersize=9)
    append!(ys, y)
end
logticks!(ax, 1, 1000)
xlims!(ax, 3, 300)
decades!(ax, ys; axis=:y)
# the lower left corner is empty: no method reaches the tolerance with the smallest budgets fast
axislegend(ax; position=:lb)

save(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "lorenz", "memory.png"), fig)
