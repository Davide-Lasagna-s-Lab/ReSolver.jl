module ReSolver

# Space-time search for periodic and relative periodic orbits of a dynamical system ∂t a = N(a),
#
#     r(a, ω, c) = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a) = 0,
#
# with space-time field a, frequency ω = 2π/T, rescaled time s ∈ [0, 2π) and drift speeds cᵢ along
# directions of translational symmetry. Two methods share one residual:
#
#     LBFGS            minimise ½‖r‖² with L-BFGS, gradient from the adjoint operator
#     NewtonHookstep   solve r = 0 with Newton–Krylov hookstep, Jacobian from the linearised operator,
#                      GMRES or flexible GMRES with right preconditioning
#
# The operators N, L and L⁺, the derivatives and the preconditioner are given by the user, as
# functions. L can be replaced by finite differences of N, and L⁺ is needed by L-BFGS only. The
# examples/ folder has the Lorenz system, the Kuramoto–Sivashinsky equation and two-dimensional
# Kolmogorov flow. The search only broadcasts and takes inner products of the fields, so it runs
# wherever the fields live, on a CPU or a GPU.
#
# L-BFGS and the Arnoldi iteration (from GMRES.jl) are local implementations,
# free to be adapted.

using LinearAlgebra
using Printf

export Orbit, System, LBFGS, NewtonHookstep, Trace, solve!

include("orbit.jl")
include("finitedifference.jl")
include("system.jl")
include("precondition.jl")
include("trace.jl")
include("lbfgs/linesearch.jl")
include("lbfgs/lbfgs.jl")
include("hookstep/arnoldi.jl")
include("hookstep/hookstep.jl")

end
