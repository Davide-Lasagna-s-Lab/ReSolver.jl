# ReSolver.jl

*Space-time search for periodic and relative periodic orbits of dynamical systems.*

Developed by Davide Lasagna, University of Southampton.

Many dynamical systems, from low-dimensional models to turbulent flows, organise their chaotic
dynamics around unstable periodic orbits: solutions that repeat exactly after a period ``T``, or
repeat up to a translation along a direction of symmetry, the *relative* periodic orbits. Finding
them is hard because they are unstable: integrating forward in time from a nearby state, as a
shooting method does, amplifies every error exponentially over the period, and the longer or more
unstable the orbit, the worse the problem.

ReSolver avoids time integration altogether. Consider a system

```math
\partial_t u = N(u) ,
```

with ``u`` a function of time and of an arbitrary number of spatial coordinates, possibly
equivariant under translations along some of them. A relative periodic orbit of period ``T``
repeats after ``T`` up to shifts ``\ell_i`` along these directions,
``u(x_i + \ell_i, t + T) = u(x_i, t)``; in the frame moving with the drift speeds ``c_i = -\ell_i/T``
it is periodic, and a periodic orbit is the case without drift. ReSolver treats the whole orbit as
a single unknown: the field over one period, written in terms of the rescaled time
``s = \omega t \in [0, 2\pi)``, together with the frequency ``\omega = 2\pi/T`` and the drift
speeds ``c_i``. An orbit is a zero of the space-time residual

```math
r(u, \omega, c) = \omega\,\partial_s u - \sum_i c_i\,\partial_i u - N(u) ,
```

which measures, at every point in space and rescaled time, how far the candidate field is from
satisfying the equation. Two methods are implemented in this package to drive this residual to zero:

| method | solves | uses | best |
|---|---|---|---|
| [`LBFGS`](@ref) | ``\min \tfrac12\lVert r\rVert^2`` | gradient from the adjoint operator, line search | far from a solution |
| [`NewtonHookstep`](@ref) | ``r = 0`` | Jacobian from the linearised operator, Krylov space, trust region | close to a solution |

## Features

- **Generic.** The search knows nothing about the system beyond a handful of functions supplied by
  the user: the nonlinear operator, its linearisation and adjoint, and the derivatives in rescaled
  time and along the drift directions. The same code serves an ODE such as the Lorenz system, a PDE
  such as the Kuramoto–Sivashinsky equation, or a large flow solver, in any discretisation.
- **Relative periodic orbits.** Drift speeds along any set of symmetry directions are unknowns of
  the search, with the phase conditions that make the Newton system square.
- **Symmetric subspaces.** A projection, given as one function, restricts the search to the
  solutions fixed by a discrete symmetry, without redefining the fields or the operators.
- **Preconditioning.** A preconditioner changes the metric in which both methods work, through a
  two-function interface; the examples show that it decides whether the search converges at all.
- **Tracing.** A [`Trace`](@ref) records the residual, the parameters, the evaluations of the
  operators and the convergence of every inner solve.
- **Self-contained.** The package depends only on the Julia standard library; L-BFGS and the
  Arnoldi iteration are local, readable implementations.

## A first example

The Kuramoto–Sivashinsky equation

```math
\partial_t u = -u\,\partial_x u - \partial_x^2 u - \partial_x^4 u , \qquad x \in [0, L) ,
```

with periodic boundary conditions, ``u(x + L, t) = u(x, t)``, is the simplest partial differential
equation with spatio-temporal chaos. It was derived independently by Kuramoto (1976), for the phase
of oscillations in reaction–diffusion systems, and by Sivashinsky (1977), for the instability of
laminar flame fronts, and it also describes thin films flowing down an inclined plane. The
second-derivative term, with negative diffusion, destabilises the large scales, the fourth-derivative
term damps the small ones, and the nonlinear term transfers energy between them. On a domain of
length ``L = 22`` the dynamics is chaotic but low-dimensional, and its equilibria, travelling waves
and relative periodic orbits were mapped in detail by Cvitanović, Davidchack & Siminos (2010). The
equation is equivariant under translations ``u(x) \to u(x + \ell)``, so its recurrent solutions are
relative periodic orbits, which repeat after a period ``T`` up to a shift ``\ell``.

Here an orbit is searched for from a near-recurrence of a chaotic trajectory, with ``u`` expanded in
Fourier modes in space and rescaled time, and the Newton–Krylov hookstep converges it with a
preconditioner built from the linear part of the equation. The search finds the shortest
pre-periodic orbit of the system, traversed twice (see [Kuramoto–Sivashinsky equation](@ref) for
the details and the full analysis):

```julia
using ReSolver
include("examples/kuramoto_sivashinsky/ks.jl")

g = KSGrid(22, 33, 49)                          # L = 22, 33 points in x, 49 in rescaled time

U          = integrate(u₀, g, 0.05, 8000; every=5)   # chaotic trajectory
e, i, m, ℓ = recurrence(U, g, 0.25, (12, 20))        # its best near-recurrence
x          = initial_orbit(U, g, 0.25, i, m, ℓ)      # Orbit: field, [log ω, c]

F = System(KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!, x;
           linearise!, ddi=(ddx!,), B=KSPreconditioner(x; kind=:jacobian))

solve!(x, F, NewtonHookstep(maxiter=50, krylov_dim=150, Δ=0.1, Δmax=10))
```

![Shortest pre-periodic orbit of Kuramoto–Sivashinsky on L = 22](assets/ks_example.png)

*Top: the initial guess and the converged orbit over one period, in the fixed frame. Middle and
bottom: the same search without and with the preconditioner; left, the residual against the Newton
iterations, with hooksteps on the boundary of the trust region (open symbols) and full Newton steps
(filled); right, the relative residual of the GMRES solution of the Newton system against the
Arnoldi steps, one curve per Newton iteration, with the tolerance ``10^{-3}`` dashed.*

Without preconditioner GMRES makes almost no progress on any Newton system: after 150 Arnoldi
steps the relative residual is still between ``0.3`` and ``1``, the Newton steps are poor, and the
residual of the orbit stalls. With the preconditioner every Newton system is solved to ``10^{-3}`` in
45 to 55 Arnoldi steps. The first five iterations are nonetheless hooksteps: far from the solution
the Newton step is longer than the trust region, and only part of it is taken. From the sixth
iteration the full Newton step is taken, and the residual drops from ``10^{-3}`` to ``10^{-12}`` in
three iterations. The search costs 441 Jacobian actions.

**Size of the problem.** On the grid of ``33 \times 49`` points in space and rescaled time, the orbit
has ``33 \times 49 = 1\,617`` field values, the log-frequency and the drift speed: ``N = 1\,619``
real unknowns. The search costs 441 Jacobian actions, about a quarter of an action per unknown, each
the cost of a few FFTs of the space-time grid. Newton's method with a direct solver would instead
form the Jacobian at every iteration, ``N`` actions for the ``1\,619 \times 1\,619`` dense matrix of
the spectral discretisation, and factorise it, about ``2N^3/3 \approx 2.8 \times 10^{9}`` operations.
A finite-difference discretisation in time makes the Jacobian banded and the factorisation much
cheaper, but its cost grows as the cube of the number of spatial unknowns, while the number of
Jacobian actions of the preconditioned Krylov solver does not grow with the resolution: see
[Cost compared with direct solvers](@ref) for an estimate.

### References

- Y. Kuramoto and T. Tsuzuki, *Persistent propagation of concentration waves in dissipative media
  far from thermal equilibrium*, Prog. Theor. Phys. 55, 356–369 (1976).
- G. I. Sivashinsky, *Nonlinear analysis of hydrodynamic instability in laminar flames — I.
  Derivation of basic equations*, Acta Astronaut. 4, 1177–1206 (1977).
- P. Cvitanović, R. L. Davidchack and E. Siminos, *On the state space geometry of the
  Kuramoto–Sivashinsky flow in a periodic domain*, SIAM J. Appl. Dyn. Syst. 9, 1–33 (2010).

## Contents

```@contents
Pages = ["usage.md",
         "theory/formulation.md", "theory/optimisation.md", "theory/root_finding.md",
         "theory/preconditioning.md",
         "examples/lorenz.md", "examples/kuramoto_sivashinsky.md", "examples/ks_symmetric.md",
         "examples/methods.md",
         "api.md"]
Depth = 1
```

## A note on how this package was written

This package was written with AI assistance (Claude, by Anthropic). Developing it would have taken
me about two weeks, with a great deal of reasoning, manual derivations and reading. Claude did it
in a day. — Davide Lasagna
