# The shortest periodic orbit of the Lorenz system, for the standard parameters σ = 10, ρ = 28,
# β = 8/3, with period T ≈ 1.5587. The initial guess is a near-recurrence of a chaotic trajectory;
# the Newton–Krylov hookstep, with the block preconditioner of lorenz.jl, converges it with
# K = 5, 10 and 20 Fourier modes in time.
#
#     julia --project=examples examples/lorenz/example.jl
#
# writes example.png in the folder figures/ next to this file.

using Printf
using CairoMakie

include("lorenz.jl")


# ---- chaotic trajectory: transient of 20, then 100 time units ----
dt = 0.005
U  = trajectory(trajectory([1.0, 1.0, 1.0], dt, 4000)[:, end], dt, 20000)

# ---- best near-recurrence with period in [1.4, 1.7] ----
lags    = round(Int, 1.4 / dt):round(Int, 1.7 / dt)
e, i, m = minimum((norm(U[:, i + m] .- U[:, i]) / norm(U[:, i]), i, m)
                  for i in 1:size(U, 2) - last(lags), m in lags)

@printf "initial guess: recurrence error %.3f, T = %.4f\n" e m * dt

# ---- search, with K = 5, 10 and 20 modes ----
orbits = map((5, 10, 20)) do K
    x₀ = initial_orbit(U, dt, i, m, K)
    F  = System(nonlinear!,
                LorenzLinearised(K),
                LorenzLinearised(K; adjoint=true),
                dds!,
                x₀;
                linearise!,
                B=LorenzPreconditioner(x₀))

    x     = copy(x₀)
    trace = Trace()
    solve!(x, F, NewtonHookstep(maxiter=20, krylov_dim=40, Δ=1, Δmax=100, tol=1e-12,
                                verbose=false, callback=trace))

    T = 2π / ReSolver.frequency(x)
    @printf "K = %2d: ‖r‖ = %.2e → %.2e in %d iterations, T = %.8f\n" K trace.res[1] trace.res[end] length(trace) - 1 T

    (; K, x, T)
end


# ============================================================================ #
# Figure: the orbits on the attractor                                          #
# ============================================================================ #

fig = Figure(size=(700, 600), fontsize=14)
ax  = Axis3(fig[1, 1]; title=@sprintf("shortest periodic orbit, T = %.4f", orbits[end].T),
            xlabel="x", ylabel="y", zlabel="z", azimuth=0.3π)

lines!(ax, U[1, :], U[2, :], U[3, :]; color=(:gray60, 0.4), linewidth=0.5)
for (orbit, color) in zip(orbits, (:orange, :royalblue, :crimson))
    v = curve(orbit.x)
    lines!(ax, v[1, :], v[2, :], v[3, :]; color, linewidth=2.5, label="K = $(orbit.K)")
end
axislegend(ax; position=:rt)

save(joinpath(@__DIR__, "figures", "example.png"), fig)
