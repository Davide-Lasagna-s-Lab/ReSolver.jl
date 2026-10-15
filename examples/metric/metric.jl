# A preconditioner as a change of metric, on a system of two quadratic equations in two unknowns.
#
#     julia --project=examples examples/metric/metric.jl
#
# writes metric.png in docs/src/assets/metric/. The residual
#
#     r₁(x, y) = 4 (x - 1/2) + (y - 1)²
#     r₂(x, y) = (x - 1/2)² + (y - 1)
#
# vanishes at p★ = (1/2, 1), where its Jacobian is diag(4, 1): the problem is four times stiffer
# along x than along y, and the Hessian of R = ½‖r‖² sixteen times. The preconditioner
# B = diag(4, 1) captures this, and the variables q = B p undo it. Three panels:
#
#     left    R = ½‖r‖² in the variables p = (x, y): steepest descent in the plain metric, and in the
#             metric of B, each with a backtracking line search, from the same start;
#     centre  the same in the variables q = B p, where the metric of B is the plain one;
#     right   at the start, the trust regions ‖δp‖ ≤ Δ and ‖B δp‖ ≤ Δ, and the hookstep curves of the
#             two metrics, from the Newton step (μ = 0) to steepest descent (μ → ∞).

using LinearAlgebra
using Printf
using CairoMakie


# ============================================================================ #
# Problem                                                                      #
# ============================================================================ #

s  = 4.0 # stiffness along x

residual(p)  = [s * (p[1] - 0.5) + (p[2] - 1)^2, (p[1] - 0.5)^2 + (p[2] - 1)]
jacobian(p)  = [s 2 * (p[2] - 1); 2 * (p[1] - 0.5) 1]
objective(p) = norm(residual(p))^2 / 2

B  = Diagonal([s, 1.0])
p₀ = [0.3, 1.55]


# ============================================================================ #
# Steepest descent in a metric                                                 #
# ============================================================================ #

# steepest descent of R in the metric ⟨p, q⟩_M = pᵀ M q: direction -M⁻¹ ∇R, with ∇R = Jᵀ r, and
# backtracking from the step that the first direction would need to reach R = 0 to first order
function descent(p₀, M; iters=25)
    path = [copy(p₀)]
    p    = copy(p₀)

    for _ in 1:iters
        r = residual(p)
        J = jacobian(p)
        g = J' * r
        d = -(M \ g)

        # ---- converged: the gradient vanishes to rounding, and so does the step ----
        norm(g) < 1e-14 && break

        # ---- step: the Gauss–Newton length along d, then halved until Armijo holds, at most 60
        #      times ----
        α = -dot(g, d) / norm(J * d)^2
        for _ in 1:60
            objective(p .+ α .* d) <= objective(p) + 1e-4 * α * dot(g, d) && break
            α /= 2
        end

        p .+= α .* d
        push!(path, copy(p))
    end

    return reduce(hcat, path)
end

plain = descent(p₀, I(2))
metric = descent(p₀, B' * B)

@printf "plain metric:  R = %.1e after %d iterations\n" objective(plain[:, end]) size(plain, 2) - 1
@printf "metric of B:   R = %.1e after %d iterations\n" objective(metric[:, end]) size(metric, 2) - 1


# ============================================================================ #
# Hookstep curves and trust regions at the start                               #
# ============================================================================ #

# the hookstep in the metric M for the multiplier μ: minimiser of ‖r + J δp‖ with ‖δp‖_M fixed,
# δp(μ) = -(JᵀJ + μ M)⁻¹ Jᵀ r, from the Newton step at μ = 0 to steepest descent as μ → ∞
hook(p, M, μ) = -((jacobian(p)' * jacobian(p) + μ * M) \ (jacobian(p)' * residual(p)))

μs     = 10.0 .^ range(-4, 6; length=400)
curveI = reduce(hcat, [p₀ .+ hook(p₀, I(2), μ) for μ in μs])
curveB = reduce(hcat, [p₀ .+ hook(p₀, B' * B, μ) for μ in μs])
newton = p₀ .- jacobian(p₀) \ residual(p₀)

# the hookstep on the boundary of each trust region of radius Δ, by bisection on log μ
function boundary(p, M, Δ)
    lo, hi = -8.0, 8.0
    for _ in 1:100
        μ = 10^((lo + hi) / 2)
        δ = hook(p, M, μ)
        sqrt(δ' * M * δ) > Δ ? (lo = log10(μ)) : (hi = log10(μ))
    end
    return p .+ hook(p, M, 10^hi)
end

Δ  = 0.1
θ  = range(0, 2π; length=200)
ΔI = [p₀[1] .+ Δ .* cos.(θ) p₀[2] .+ Δ .* sin.(θ)]'           # ‖δp‖ ≤ Δ
ΔB = [p₀[1] .+ Δ .* cos.(θ) ./ s p₀[2] .+ Δ .* sin.(θ)]'      # ‖B δp‖ ≤ Δ, i.e. δp = B⁻¹ z, ‖z‖ ≤ Δ
hI = boundary(p₀, I(2), Δ)
hB = boundary(p₀, B' * B, Δ)


# ============================================================================ #
# Figure                                                                       #
# ============================================================================ #

fig = Figure(size=(1400, 620), fontsize=13)

cplain = :gray30
cmetric = :crimson

# contours of log₁₀ R on a box around the start and the solution
xs = range(0.2, 0.8; length=300)
ys = range(0.45, 1.65; length=300)
LR = [log10(objective([x, y]) + 1e-16) for x in xs, y in ys]
levels = -5:0.25:0

# the zero sets r₁ = 0 and r₂ = 0, whose intersection is the solution
z₁ = [0.5 - (y - 1)^2 / s for y in ys]

# ---- left: variables p ----
ax1 = Axis(fig[1, 1]; title="variables p = (x, y)", xlabel="x", ylabel="y")
contour!(ax1, xs, ys, LR; levels, colormap=:Blues, linewidth=0.8)
lines!(ax1, z₁, ys; color=:black, linestyle=:dash, linewidth=1)
lines!(ax1, plain[1, :], plain[2, :]; color=cplain, label="steepest descent, plain metric")
scatter!(ax1, plain[1, :], plain[2, :]; color=cplain, markersize=4)
lines!(ax1, metric[1, :], metric[2, :]; color=cmetric, label="steepest descent, metric of B")
scatter!(ax1, metric[1, :], metric[2, :]; color=cmetric, markersize=6)
scatter!(ax1, [0.5], [1.0]; color=:black, marker=:star5, markersize=16, label="solution")
scatter!(ax1, [p₀[1]], [p₀[2]]; color=:black, marker=:rect, markersize=10, label="start")
xlims!(ax1, 0.2, 0.8); ylims!(ax1, 0.45, 1.65)
ax1.aspect = DataAspect()

# ---- centre: variables q = B p ----
ax2 = Axis(fig[1, 2]; title="variables q = B p", xlabel="4 x", ylabel="y")
contour!(ax2, s .* xs, ys, LR; levels, colormap=:Blues, linewidth=0.8)
lines!(ax2, s .* z₁, ys; color=:black, linestyle=:dash, linewidth=1)
lines!(ax2, s .* plain[1, :], plain[2, :]; color=cplain)
scatter!(ax2, s .* plain[1, :], plain[2, :]; color=cplain, markersize=4)
lines!(ax2, s .* metric[1, :], metric[2, :]; color=cmetric)
scatter!(ax2, s .* metric[1, :], metric[2, :]; color=cmetric, markersize=6)
scatter!(ax2, [s / 2], [1.0]; color=:black, marker=:star5, markersize=16)
scatter!(ax2, [s * p₀[1]], [p₀[2]]; color=:black, marker=:rect, markersize=10)
xlims!(ax2, s * 0.2, s * 0.8); ylims!(ax2, 0.45, 1.65)
ax2.aspect = DataAspect()


# ---- right: trust regions and hookstep curves at the start ----
ax3 = Axis(fig[1, 3]; title="trust regions at the start", xlabel="x", ylabel="y")
contour!(ax3, xs, ys, LR; levels, colormap=:Blues, linewidth=0.8)
poly!(ax3, Point2f.(eachcol(ΔI)); color=(cplain, 0.12), strokecolor=cplain, strokewidth=1,
      label="‖δp‖ ≤ Δ")
poly!(ax3, Point2f.(eachcol(ΔB)); color=(cmetric, 0.15), strokecolor=cmetric, strokewidth=1,
      label="‖B δp‖ ≤ Δ")
lines!(ax3, curveI[1, :], curveI[2, :]; color=cplain, linewidth=2, label="hookstep curve, plain")
lines!(ax3, curveB[1, :], curveB[2, :]; color=cmetric, linewidth=2, label="hookstep curve, metric of B")
scatter!(ax3, [hI[1]], [hI[2]]; color=cplain, markersize=10)
scatter!(ax3, [hB[1]], [hB[2]]; color=cmetric, markersize=10)
scatter!(ax3, [newton[1]], [newton[2]]; color=:black, marker=:xcross, markersize=12,
         label="Newton step")
scatter!(ax3, [0.5], [1.0]; color=:black, marker=:star5, markersize=16)
scatter!(ax3, [p₀[1]], [p₀[2]]; color=:black, marker=:rect, markersize=10)
xlims!(ax3, 0.2, 0.8); ylims!(ax3, 0.45, 1.65)
ax3.aspect = DataAspect()

# ---- the legends of the three panels, in a row below them ----
legends = fig[2, 1:3] = GridLayout()
Legend(legends[1, 1], ax1; orientation=:horizontal, nbanks=2, framevisible=false, labelsize=11)
Legend(legends[1, 2], ax3; orientation=:horizontal, nbanks=2, framevisible=false, labelsize=11)

# ---- each panel as wide as its data, and the figure fitted to the panels ----
colsize!(fig.layout, 1, Aspect(1, 0.5))
colsize!(fig.layout, 2, Aspect(1, 2.0))
colsize!(fig.layout, 3, Aspect(1, 0.5))
resize_to_layout!(fig)

save(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "metric", "metric.png"), fig)

@printf "hookstep on the boundary, plain metric: δp = (%.3f, %.3f); metric of B: δp = (%.3f, %.3f); Newton step: δp = (%.3f, %.3f)\n" (hI .- p₀)... (hB .- p₀)... (newton .- p₀)...
