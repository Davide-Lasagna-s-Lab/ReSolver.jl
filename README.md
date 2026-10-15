<p align="center">
  <img src="assets/logo.svg" alt="ReSolver.jl" width="760">
</p>

<p align="center">
  <a href="https://davide-lasagna-s-lab.github.io/ReSolver.jl/stable/"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable documentation"></a>
  <a href="https://davide-lasagna-s-lab.github.io/ReSolver.jl/dev/"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Development documentation"></a>
  <a href="https://github.com/Davide-Lasagna-s-Lab/ReSolver.jl/actions/workflows/CI.yml?query=branch%3Amain"><img src="https://github.com/Davide-Lasagna-s-Lab/ReSolver.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://codecov.io/gh/Davide-Lasagna-s-Lab/ReSolver.jl"><img src="https://codecov.io/gh/Davide-Lasagna-s-Lab/ReSolver.jl/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
</p>

ReSolver finds unstable periodic and relative periodic orbits of dynamical systems by a
variational method. A whole orbit is a single unknown, a closed loop in state space. The loop is
deformed until it satisfies the equations of motion at every instant.

For a system $\partial_t u = N(u)$, possibly equivariant under translations along some directions,
the loop is written in the rescaled time $s = \omega t \in [0, 2\pi)$, in the frame that drifts
with the speeds $c_i$ along the directions of symmetry. The unknowns are the field over one period,
the frequency $\omega = 2\pi/T$ and the drift speeds. An orbit is a zero of the space-time residual

$$
r(u, \omega, c) = \omega\,\partial_s u - \sum_i c_i\,\partial_i u - N(u) .
$$

No time integration is involved, so errors do not grow exponentially along long or strongly
unstable orbits. Two methods drive the residual to zero. `LBFGS` minimises $\tfrac12\lVert r\rVert^2$
and is robust far from an orbit. `NewtonHookstep` solves $r = 0$ by a matrix-free Newton–Krylov
method and converges fast close to one. The package knows nothing about the system beyond a few
functions supplied by the user. The same code serves an ODE, a PDE or a flow solver, in any
discretisation, on a CPU or a GPU.

## Features

- **Two methods on one residual.** L-BFGS descent, with gradients from the adjoint operator. With
  `memory=0` it is preconditioned gradient descent with Barzilai–Borwein steps. And a Newton–Krylov
  hookstep in a trust region, with GMRES or flexible GMRES.
- **Exact or Jacobian-free.** Three constructors of `System`, by the operators given. With the
  nonlinear, linearised and adjoint operators both methods run. Without the adjoint the hookstep
  runs. With the nonlinear operator alone the hookstep applies the linearised operator by finite
  differences, forward or central (second order). The residual stays exact, so the orbits are as
  accurate.
- **Relative periodic orbits.** Any number of drift directions, each with an unknown speed and a
  phase condition. Discrete shifts across a direction of symmetry are handled by the model, as in
  the Kolmogorov flow example.
- **Symmetric subspaces.** An orthogonal projection, passed as `project`, restricts the search to
  the fields fixed by a symmetry. The projection is applied to every residual, gradient, Jacobian
  action and preconditioner output, so the search never leaves the subspace.
- **Preconditioning.** A preconditioner changes the metric of both methods. It is applied on the
  right in GMRES, and can be rebuilt about the current orbit at every Newton iteration. Inner
  iterative solves are allowed with flexible GMRES. The examples include diagonal Fourier
  preconditioners and the factorised space-time preconditioner of Kolmogorov flow.
- **Tracing.** A `Trace` passed as callback records the residual, the parameters, the wall-clock
  time, the counts of residuals, Jacobian actions and preconditioner applications, the GMRES
  history of every Newton iteration and the trust-region radius, step and ratio. A callback can
  also stop a search.
- **Any array, any device.** The search only broadcasts fields and takes their inner products. The
  fields can be CPU arrays, GPU arrays, single or double precision.
- **Light.** The package depends only on the Julia standard library.

## Installation

ReSolver is not registered. From the Julia REPL,

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ReSolver.jl")
```

## The system

A `System` collects the operators. There are three constructors, by the operators given.

```julia
System(nl, lin, adj, dds!, x; linearise!, ddi, B, project)   # LBFGS and NewtonHookstep
System(nl, lin, dds!, x; linearise!, ddi, B, project)        # NewtonHookstep
System(nl, dds!, x; difference=:forward, ddi, B, project)    # NewtonHookstep, Jacobian-free
```

Here `nl`, `lin` and `adj` are the nonlinear, linearised and adjoint operators, `linearise!` sets
the point of the linearisation, `dds!` is the derivative in rescaled time and `ddi` holds the
derivatives along the drift directions. `B` is the preconditioner and `project` the projection on a
symmetric subspace, both optional. `x` is an `Orbit`, the field and the parameters `[log ω, c...]`.

Writing the linearised operator of a large model takes time, and writing its adjoint takes more.
The Jacobian-free constructor needs neither. The hookstep then applies the linearised operator by
a finite difference of `nl` about the current orbit $u$,

$$
L\,b \approx \frac{N(u + \varepsilon b) - N(u)}{\varepsilon}
\qquad \text{or} \qquad
L\,b \approx \frac{N(u + \varepsilon b) - N(u - \varepsilon b)}{2\varepsilon} ,
$$

with `difference=:forward` or `difference=:central`. The step $\varepsilon$ is scaled with the norms
of $u$ and $b$ and with the machine precision of the fields. A forward action costs one evaluation
of `nl`, a central one two. The central difference is exact up to rounding when $N$ is quadratic,
as for the Navier–Stokes equations. The residual itself is never approximated, so an orbit
converged without the linearised operator satisfies the equations to the same tolerance. On the
Kuramoto–Sivashinsky example below the forward difference takes the same 9 Newton iterations and
441 Jacobian actions as the exact linearised operator.

## Example 1. Kuramoto–Sivashinsky on a CPU

The Kuramoto–Sivashinsky equation $\partial_t u = -u\,\partial_x u - \partial_x^2 u - \partial_x^4 u$
on a periodic domain of length $L = 22$. The model is in `examples/kuramoto_sivashinsky/ks.jl`.
A guess is cut from a near recurrence of a chaotic trajectory, and the hookstep converges it with a
preconditioner built from the linear part of the equation. Run it from the root of the repository,
in the environment of the examples, `julia --project=examples`.

```julia
using Random
using ReSolver
include("examples/kuramoto_sivashinsky/ks.jl")      # fields, operators, preconditioners

Random.seed!(1)
g = KSGrid(22, 33, 49)                              # L = 22, 33 points in x, 49 instants in s

# ---- a chaotic trajectory and its best near recurrence ----
u₀  = 0.1 .* randn(g.Nx)
u₀ .-= sum(u₀) / g.Nx
U   = integrate(u₀, g, 0.05, 4000; every=4000)      # transient
U   = integrate(U[:, end], g, 0.05, 8000; every=5)  # 400 time units, a state every 0.25

e, i, m, ℓ = recurrence(U, g, 0.25, (12, 20))
x          = initial_orbit(U, g, 0.25, i, m, ℓ)     # Orbit: field, [log ω, c]

# ---- the system and the search ----
F = System(KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!, x;
           linearise!, ddi=(ddx!,), B=KSPreconditioner(x; kind=:jacobian))

solve!(x, F, NewtonHookstep(maxiter=50, krylov_dim=150, Δ=0.1, Δmax=10))

2π / ReSolver.frequency(x)                          # the period
```

The search finds the shortest pre-periodic orbit of the system, traversed twice, with period
$T = 20.5057$.

![Shortest pre-periodic orbit of Kuramoto–Sivashinsky on L = 22](docs/src/assets/kuramoto_sivashinsky/example.png)

Top, the initial guess and the converged orbit. Middle and bottom, the search without and with the
preconditioner. Left, the residual against the Newton iterations, with hooksteps open and full
Newton steps filled. Right, the convergence of GMRES in each Newton iteration.

## Example 2. Kolmogorov flow on a GPU

Two-dimensional Kolmogorov flow at $Re = 40$ with forcing wavenumber 4. The model is in
`examples/kolmogorov/kolmogorov.jl`, and its operators act on all instants of the loop at once, by
fast Fourier transforms and broadcasts. A turbulent trajectory is integrated on the CPU. Its best
near recurrence with period between 5 and 15 is resampled on about four instants per time unit and
copied to the GPU. The hookstep converges it with the factorised space-time preconditioner and
flexible GMRES. The system is built without the adjoint, which the hookstep does not need.

```julia
using CUDA
using ReSolver
include("examples/kolmogorov/kolmogorov.jl")        # fields, operators, preconditioner

# ---- a turbulent trajectory on the CPU, and its best near recurrence ----
g₁ = KFGrid(33, 1; Re=40, kf=4)                     # one state, 33 wavenumbers
W  = integrate(ω̂, g₁, 0.02, 20000; every=5)         # 400 time units, a state every 0.1
e, i, m, ℓx, ℓy = recurrence(W, g₁, 0.1, (5, 15))

# ---- the loop on the GPU, about four instants per time unit ----
Ns = max(33, 2 * ceil(Int, 2 * m * 0.1) + 1)
g  = KFGrid(33, Ns; Re=40, kf=4, yshift=ℓy, array=CuArray)
x  = initial_orbit(W, g, 0.1, i, m, ℓx)

# ---- the system without the adjoint, and the search ----
B = KFPreconditioner(x; kind=:factorised, σ=min(Ns / (m * 0.1), 2.5), inner=10)
F = System(KFNonlinear(g), KFLinearised(g), dds!, x; linearise!, ddi=(ddx!,), B)

solve!(x, F, NewtonHookstep(maxiter=40, krylov_dim=300, krylov_tol=1e-2, Δ=0.1, Δmax=10,
                            tol=1e-10, flexible=true))
```

The full script is `examples/kolmogorov/example.jl`, run with
`julia --project=<an environment with CUDA> examples/kolmogorov/example.jl gpu`.

## More examples

Each example has a model file, a script `example.jl` and a page of the documentation. The scripts
that draw the figures of the documentation are in the `scripts` folder of each example. The
examples have their own environment, `examples/Project.toml`.

| folder | system | what it shows |
|---|---|---|
| `examples/lorenz` | Lorenz system | the shortest periodic orbit, the full Jacobian, three preconditioners |
| `examples/kuramoto_sivashinsky` | Kuramoto–Sivashinsky equation | relative periodic orbits on $L = 22$, convergence up to $10^5$ unknowns, five preconditioners, the odd subspace, L-BFGS against the hookstep |
| `examples/kolmogorov` | two-dimensional Kolmogorov flow | relative periodic orbits at $Re = 40$ on a CPU or a GPU, with a factorised space-time preconditioner and flexible GMRES |
| `examples/metric` | two equations in two unknowns | a preconditioner as a change of metric |

```
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/kuramoto_sivashinsky/example.jl
julia --project=examples examples/lorenz/example.jl
```

## Documentation and tests

The [documentation](https://davide-lasagna-s-lab.github.io/ReSolver.jl/stable/) covers the usage,
the theory of the two methods and of preconditioning, and the examples. To build it locally and to
run the tests,

```
julia --project=docs docs/make.jl
julia --project -e 'using Pkg; Pkg.test()'
```

## References

ReSolver builds on the variational method of Cvitanović & Lan (2003) and Lan & Cvitanović (2004),
who first posed the search for a periodic orbit as the deformation of a closed loop until it
satisfies the equations of motion. Farazmand (2016) found invariant solutions of the Navier–Stokes
equations by adjoint-based descent. Azimi, Ashtari & Schneider (2022) turned this into a
variational method for the periodic orbits of high-dimensional chaotic systems. Parker & Schneider
(2022) applied variational methods to the periodic orbits of the incompressible Navier–Stokes
equations, and Ashtari & Schneider (2023a, 2023b) to three-dimensional shear flows and to
connecting orbits. The Newton–Krylov hookstep follows Viswanath (2007), and the Jacobian-free
option follows Knoll & Keyes (2004). The Kolmogorov flow example is set as in Chandler & Kerswell
(2013).

- P. Cvitanović and Y. Lan, *Turbulent fields and their recurrences*, in Correlations and Fluctuations in QCD, World Scientific, 313–325 (2003). [doi:10.1142/9789812704641_0032](https://doi.org/10.1142/9789812704641_0032)
- Y. Lan and P. Cvitanović, *Variational method for finding periodic orbits in a general flow*, Phys. Rev. E 69, 016217 (2004). [doi:10.1103/PhysRevE.69.016217](https://doi.org/10.1103/PhysRevE.69.016217)
- M. Farazmand, *An adjoint-based approach for finding invariant solutions of Navier–Stokes equations*, J. Fluid Mech. 795, 278–312 (2016). [doi:10.1017/jfm.2016.203](https://doi.org/10.1017/jfm.2016.203)
- S. Azimi, O. Ashtari and T. M. Schneider, *Constructing periodic orbits of high-dimensional chaotic systems by an adjoint-based variational method*, Phys. Rev. E 105, 014217 (2022). [doi:10.1103/PhysRevE.105.014217](https://doi.org/10.1103/PhysRevE.105.014217)
- J. P. Parker and T. M. Schneider, *Variational methods for finding periodic orbits in the incompressible Navier–Stokes equations*, J. Fluid Mech. 941, A17 (2022). [doi:10.1017/jfm.2022.299](https://doi.org/10.1017/jfm.2022.299)
- O. Ashtari and T. M. Schneider, *Identifying invariant solutions of wall-bounded three-dimensional shear flows using robust adjoint-based variational techniques*, J. Fluid Mech. 977, A7 (2023a). [doi:10.1017/jfm.2023.927](https://doi.org/10.1017/jfm.2023.927)
- O. Ashtari and T. M. Schneider, *Jacobian-free variational method for computing connecting orbits in nonlinear dynamical systems*, Chaos 33, 073134 (2023b). [doi:10.1063/5.0143923](https://doi.org/10.1063/5.0143923)
- D. Viswanath, *Recurrent motions within plane Couette turbulence*, J. Fluid Mech. 580, 339–358 (2007). [doi:10.1017/S0022112007005459](https://doi.org/10.1017/S0022112007005459)
- D. A. Knoll and D. E. Keyes, *Jacobian-free Newton–Krylov methods. A survey of approaches and applications*, J. Comput. Phys. 193, 357–397 (2004). [doi:10.1016/j.jcp.2003.08.010](https://doi.org/10.1016/j.jcp.2003.08.010)
- G. J. Chandler and R. R. Kerswell, *Invariant recurrent solutions embedded in a turbulent two-dimensional Kolmogorov flow*, J. Fluid Mech. 722, 554–595 (2013). [doi:10.1017/jfm.2013.122](https://doi.org/10.1017/jfm.2013.122)

## Citing

The method and the Kolmogorov flow example are described in the following paper, in preparation.

```bibtex
@unpublished{lasagna_spacetime_newton_krylov,
  author = {Lasagna, Davide},
  title  = {Space-time {Newton--Krylov} searches for unstable periodic orbits of
            two-dimensional {Kolmogorov} flow},
  note   = {In preparation},
  year   = {2026}
}
```

## A note on how this package was written

This package was written with AI assistance (Claude, by Anthropic). Developing it would have taken
me about two weeks, with a great deal of reasoning, manual derivations and reading. Claude did it
in a day. — Davide Lasagna

## Authors

Developed by Davide Lasagna, University of Southampton.

## License

MIT, see [LICENSE](LICENSE).
