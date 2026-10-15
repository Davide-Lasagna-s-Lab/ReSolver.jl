# A relative periodic orbit of the Kolmogorov flow at Re = 40, kf = 4, converged on a GPU from a
# near-recurrence of a turbulent trajectory. The trajectory is integrated on the CPU, from the
# laminar flow perturbed on the large scales, after a transient of 200 time units. The initial guess
# is its best near-recurrence with period between 5 and 15, up to a shift along x and one across the
# force by a multiple of 2π/kf. The guess is resampled on about four instants per time unit, copied
# to the GPU, and converged by the hookstep with the factorised preconditioner of the paper
# (docs/src/examples/kolmogorov_flow.md), σ = min(Ns/T, 2.5) and 10 inner steps on the blocks, and
# flexible GMRES with up to 300 Krylov vectors and a Krylov tolerance of 1e-2.
#
#     julia --project=<an environment with CUDA> examples/kolmogorov/example.jl [gpu|cpu] [seed]
#
# Writes data/example.h5 with the converged orbit, the trace of the search and the energy input and
# dissipation of the trajectory. scripts/example_figure.jl draws the figure of the README from it.

using Random
using Printf
using LinearAlgebra
using HDF5

include("kolmogorov.jl")


# ============================================================================ #
# Device                                                                       #
# ============================================================================ #

const DEVICE = get(ARGS, 1, "gpu")
const SEED   = parse(Int, get(ARGS, 2, "1"))

if DEVICE == "gpu"
    using CUDA
    const array = CuArray
    println("device: ", CUDA.name(CUDA.device()))
else
    const array = Array
    println("device: ", Sys.cpu_info()[1].model)
end

Random.seed!(SEED)

const n  = 33     # largest wavenumber
const Re = 40     # Reynolds number
const kf = 4      # forcing wavenumber
const dt = 0.02   # step of the time integration
const Δt = 0.1    # spacing of the stored states


# ============================================================================ #
# A turbulent trajectory and its best near-recurrence, on the CPU              #
# ============================================================================ #

g₁ = KFGrid(n, 1; Re, kf)
kx = Array(g₁.kx)[:, :, 1]
ky = Array(g₁.ky)[:, :, 1]

# ---- the laminar flow, perturbed on the large scales, after a transient of 200 ----
ω̂  = Array(g₁.forcing)[:, :, 1] .* (Re / kf^2)
ω̂ .+= 0.1 .* randn(ComplexF64, n + 1, 2n + 1) .* (kx .<= 4) .* (abs.(ky) .<= 4) .* (kx .> 0)
ω̂  = integrate(ω̂, g₁, dt, round(Int, 200 / dt); every=round(Int, 200 / dt))[:, :, end]

# ---- 400 time units, a state every 0.1 ----
W = integrate(ω̂, g₁, dt, round(Int, 400 / dt); every=round(Int, Δt / dt))

# ---- the best near-recurrence with period in [5, 15] ----
e, i, m, ℓx, ℓy = recurrence(W, g₁, Δt, (5, 15))
T₀              = m * Δt

@printf "initial guess: recurrence error %.3f, T = %.2f, shift along x %.3f, along y %.3f\n" e T₀ ℓx ℓy


# ============================================================================ #
# The search, on the device                                                    #
# ============================================================================ #

# ---- the guess on about four instants per time unit, an odd number ----
Ns = max(33, 2 * ceil(Int, 2T₀) + 1)
g  = KFGrid(n, Ns; Re, kf, yshift=ℓy, array)
x  = initial_orbit(W, g, Δt, i, m, ℓx)

# ---- the factorised preconditioner and the system, without the adjoint: the hookstep only ----
B = KFPreconditioner(x; kind=:factorised, σ=min(Ns / T₀, 2.5), inner=10)
F = System(KFNonlinear(g), KFLinearised(g), dds!, x; linearise!, ddi=(ddx!,), B)

# ---- hookstep, with flexible GMRES since the inner solves change B⁻¹ at every Arnoldi step ----
@printf "Ns = %d: %d unknowns\n" Ns ((2n + 1)^2 - 1) * Ns + 2
trace = Trace()
t     = @elapsed solve!(x, F, NewtonHookstep(maxiter=40, krylov_dim=300, krylov_tol=1e-2, Δ=0.1,
                                             Δmax=10, tol=1e-10, flexible=true, callback=trace))

T = 2π / ReSolver.frequency(x)
@printf "‖r‖ = %.2e after %d Newton iterations and %d Jacobian actions, %.1f s\n" trace.res[end] length(trace) - 1 trace.evaluations[end].jacobian t
@printf "T = %.6f, drift speed %.6f, shift along y %.3f\n" T x.p[2] ℓy


# ============================================================================ #
# Output                                                                       #
# ============================================================================ #

# ---- energy input and dissipation of the trajectory, over their laminar values Re/(2kf²) ----
w = [1; fill(2, n)]
I = [-real(W[1, kf + 1, l]) / kf for l in axes(W, 3)] ./ (Re / 2kf^2)
D = [sum(w .* sum(abs2, W[:, :, l]; dims=2)) / Re for l in axes(W, 3)] ./ (Re / 2kf^2)

mkpath(joinpath(@__DIR__, "data"))
h5open(joinpath(@__DIR__, "data", "example.h5"), "w") do f
    a = Array(x.a.data)
    f["orbit_real"] = real.(a)
    f["orbit_imag"] = imag.(a)
    f["res"]        = trace.res
    f["time"]       = trace.time
    f["step"]       = trace.step
    f["jacobian"]   = [ev.jacobian for ev in trace.evaluations]
    f["I"]          = I
    f["D"]          = D

    A = HDF5.attributes(f)
    A["period"]     = T
    A["drift"]      = x.p[2]
    A["yshift"]     = ℓy
    A["guess_T"]    = T₀
    A["recurrence"] = e
    A["seed"]       = SEED
    A["n"]          = n
    A["Re"]         = Re
    A["kf"]         = kf
    A["device"]     = DEVICE == "gpu" ? CUDA.name(CUDA.device()) : Sys.cpu_info()[1].model
end
println("written ", joinpath(@__DIR__, "data", "example.h5"))
