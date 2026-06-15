<p align="center">
  <img src="assets/logo.svg" alt="ReSolver.jl logo" width="680">
</p>

# ReSolver.jl

`ReSolver.jl` finds **periodic orbits** (and equilibria) of dynamical systems by
recasting them as a global optimisation problem: the entire space-time loop is
optimised at once to minimise a residual that measures how far a candidate
trajectory is from satisfying the governing equations.

Rather than time-stepping or shooting, a whole trajectory is represented over one
period and the objective

```
R(x, T) = ½ ‖ ω · dx/ds − N(x) ‖²,        ω = 2π/T
```

is driven to zero with gradient-based optimisers. The gradient with respect to
both the loop `x` and the period `T` is supplied analytically through an adjoint
operator, so large state spaces remain tractable.

This is the engine behind the `ReSolver-*` flow-specific packages
(channel flow, square duct, rotating plane Couette flow, Rayleigh-Bénard, ...),
but it is fully generic: any system that provides a time-derivative operator, a
nonlinear right-hand side, and its adjoint can be optimised.

## Features

- `Residual` functional that evaluates the global space-time residual and, when
  called with a gradient buffer, returns the residual together with the loop and
  period gradients in one pass.
- `optimise!` driver built on [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
  optimising the loop and its period `T` jointly.
- A choice of optimisers re-exported for convenience: `LBFGS`,
  `ConjugateGradient`, and `GradientDescent` (with `LineSearches`).
- Custom inner-product weighting (`norm_weight`) and a gradient-rescaling hook
  (`grad_scale`) for states stored with real FFTs and Hermitian symmetry.
- Generic over the state type `X`: bring your own loop representation provided it
  is broadcastable and supports `dot`, `similar`, and the weighted-norm interface.
- `ToySystems` submodule with the Lorenz, Van der Pol, and a quadratic system for
  demonstration and testing.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/ReSolver.jl")
```

## Example

A minimal scalar example showing the full workflow — equip a state with a
time-derivative operator, a right-hand side, and its adjoint, then optimise:

```julia
using ReSolver

# State x and initial period T
x0, T0 = [1.0], 1.0

# Residual for ẋ = N(x), with N(x) = x² and adjoint of its linearisation
RdR! = Residual(x0,
                (dxds, x)    -> (dxds[1] = 0.0;        dxds),  # time-derivative operator
                (F, x)       -> (F[1]    = x[1]^2;     F),     # nonlinear right-hand side
                (G, x, r)    -> (G[1]    = 2x[1]*r[1]; G))     # adjoint linearised operator

opts = OptOptions(alg = LBFGS(), maxiter = 1000, verbose = true)

x, T, output = optimise!(copy(x0), T0, RdR!; opts = opts)
```

`optimise!` returns the converged loop `x`, its period `T`, and an `output`
dictionary holding the `Optim.jl` result and the final optimisation state.

### Toy dynamical systems

The `ToySystems` submodule provides ready-made operators for exploring the
method on chaotic systems, e.g. searching for periodic orbits of the Lorenz
system:

```julia
using ReSolver
using ReSolver.ToySystems
```

Each toy system supplies a state type, a time-derivative operator `dds!`, and a
system object that acts both as the nonlinear right-hand side and as its adjoint
— exactly the three ingredients `Residual` expects.

## How it works

| Ingredient    | Role                                                            |
|---------------|-----------------------------------------------------------------|
| `dds!`        | time-derivative operator `ω · dx/ds` over the periodic loop     |
| `rhs!`        | nonlinear dynamics `N(x)` of the system                         |
| `adj!`        | adjoint of the linearised dynamics, used to form the gradient   |
| `norm_weight` | symmetric positive-definite weighting of the inner product      |
| `grad_scale`  | per-mode gradient rescaling for real-FFT state representations   |

The optimiser walks `(x, T)` downhill on `R` until the residual reaches
`res_tol`; a zero of `R` is a periodic orbit (or, in the limit `ω → 0`, an
equilibrium) of the system.

## Related packages

These flow-specific packages build on `ReSolver.jl` and `NSEBase.jl`:

- [ReSolver-ChannelFlow.jl](https://github.com/Davide-Lasagna-s-Lab/ReSolver-ChannelFlow.jl) — pressure-driven channel flow
- [ReSolver-SquareDuct.jl](https://github.com/Davide-Lasagna-s-Lab/ReSolver-SquareDuct.jl) — square-duct flow
- [ReSolver-RPCF.jl](https://github.com/Davide-Lasagna-s-Lab/ReSolver-RPCF.jl) — rotating plane Couette flow
- [ReSolver-RayleighBenard.jl](https://github.com/Davide-Lasagna-s-Lab/ReSolver-RayleighBenard.jl) — Rayleigh-Bénard convection
