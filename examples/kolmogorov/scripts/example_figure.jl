# The figure of the Kolmogorov example of the README, from data/example.h5 written by example.jl:
# (a) the converged orbit in the plane of the energy input I and the dissipation D, over their laminar
# values, on the turbulent trajectory it was found from, (b) the vorticity at four instants of the
# orbit, marked in (a), in the frame drifting with it, and (c) the residual against the seconds of
# the search on the GPU, one marker per Newton iteration, open for hooksteps and filled for full
# Newton steps.
#
#     julia --project=examples examples/kolmogorov/scripts/example_figure.jl
#
# Writes docs/src/assets/kolmogorov/example.png.

using Printf
using HDF5
using CairoMakie

include(joinpath(@__DIR__, "..", "kolmogorov.jl"))


# ============================================================================ #
# Data                                                                         #
# ============================================================================ #

const FILE = joinpath(@__DIR__, "..", "data", "example.h5")

data = h5open(FILE) do f
    A = HDF5.attributes(f)
    (a      = read(f["orbit_real"]) .+ im .* read(f["orbit_imag"]),
     res    = read(f["res"]),
     time   = read(f["time"]),
     step   = read(f["step"]),
     I      = read(f["I"]),
     D      = read(f["D"]),
     T      = read(A["period"]),
     c      = read(A["drift"]),
     yshift = read(A["yshift"]),
     n      = read(A["n"]),
     Re     = read(A["Re"]),
     kf     = read(A["kf"]),
     device = read(A["device"]))
end

(; a, n, Re, kf, T) = data
Ns = size(a, 3)

# ---- energy input and dissipation along the orbit, over their laminar values Re/(2kf²) ----
w  = [1; fill(2, n)]
Io = [-real(a[1, kf + 1, l]) / kf for l in 1:Ns] ./ (Re / 2kf^2)
Do = [sum(w .* sum(abs2, a[:, :, l]; dims=2)) / Re for l in 1:Ns] ./ (Re / 2kf^2)

# ---- four instants, a quarter of a period apart, and the vorticity there ----
g     = KFGrid(n, Ns; Re, kf, yshift=data.yshift)
u     = KFField(g, a)
snaps = [1 + round(Int, k * Ns / 4) for k in 0:3]
ω     = [physical(u, l) for l in snaps]
xs    = range(0, 2π; length=size(ω[1], 1) + 1)[1:end - 1]

@printf "T = %.4f, c = %.5f, %d Newton iterations, ‖r‖ = %.1e in %.0f s on %s\n" T data.c length(data.res) - 1 data.res[end] data.time[end] data.device


# ============================================================================ #
# Figure                                                                       #
# ============================================================================ #

fig = Figure(size=(1150, 400), fontsize=14)
RED = :crimson

# ---- (a) the orbit on the turbulent trajectory, in the plane (I, D) ----
axa = Axis(fig[1, 1]; xlabel="I / I_lam", ylabel="D / D_lam", title="(a) orbit, T = $(round(T; digits=2))",
           aspect=1)
lines!(axa, data.I, data.D; color=(:gray60, 0.5), linewidth=0.6)
lines!(axa, [Io; Io[1]], [Do; Do[1]]; color=RED, linewidth=2.5)
scatter!(axa, Io[snaps], Do[snaps]; color=:white, strokecolor=:black, strokewidth=1.2, markersize=16)
text!(axa, Io[snaps], Do[snaps]; text=string.(1:4), align=(:center, :center), fontsize=10)

# ---- (b) the vorticity at the four instants ----
grid = fig[1, 2] = GridLayout()
lim  = maximum(maximum(abs, z) for z in ω)
for (k, z) in enumerate(ω)
    ax = Axis(grid[(k - 1) ÷ 2 + 1, (k - 1) % 2 + 1]; aspect=1, title="$k", titlesize=12)
    heatmap!(ax, xs, xs, z; colormap=:balance, colorrange=(-lim, lim))
    hidedecorations!(ax)
end
Label(grid[0, 1:2], "(b) vorticity"; font=:bold)
rowgap!(grid, 4)
colgap!(grid, 4)

# ---- (c) the residual against the seconds, one marker per Newton iteration ----
axc = Axis(fig[1, 3]; xlabel="seconds", ylabel="‖r‖", yscale=log10, title="(c) search on $(data.device)")
lines!(axc, data.time, data.res; color=RED)
for (kind, fill) in (("", RED), ("hook", :white), ("newton", RED))
    sel = data.step .== kind
    scatter!(axc, data.time[sel], data.res[sel]; color=fill, strokecolor=RED, strokewidth=1.2, markersize=8)
end

colsize!(fig.layout, 1, Relative(0.3))
colsize!(fig.layout, 2, Relative(0.3))

out = joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "kolmogorov", "example.png")
save(out, fig; px_per_unit=2)
println("written docs/src/assets/kolmogorov/example.png")
