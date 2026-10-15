# Usage

## Installation

ReSolver is not registered. Install it from the Julia REPL with

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ReSolver.jl")
```

The package depends only on the Julia standard library. The examples have their own environment,
`examples/Project.toml`, with FFTW, HDF5 and CairoMakie. From the root of the repository, run

```
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/kuramoto_sivashinsky/example.jl
```

## The three objects

A search involves three objects.

- An [`Orbit`](@ref) holds the unknowns. These are the space-time field `a`, in any
  discretisation, and the parameter vector `p = [log ω, c₁, …, c_m]`, the logarithm of the
  frequency followed by one drift speed per drift direction. `ReSolver.frequency(x)` returns
  $\omega$.
- A [`System`](@ref) holds the functions that define the problem. These are the nonlinear operator
  and the derivatives, and optionally the linearised and adjoint operators, a preconditioner and a
  projection.
- A method, [`LBFGS`](@ref) or [`NewtonHookstep`](@ref), holds the options of the search, and
  [`solve!`](@ref) runs it.

```julia
x = Orbit(a₀, [log(2π / T₀), c₀])                    # initial guess
F = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=P)

solve!(x, F, LBFGS(maxiter=300))                     # far from a solution
solve!(x, F, NewtonHookstep(krylov_dim=150))         # close to it
```

`solve!` overwrites `x` with the result.

## Defining a system

The search knows nothing about the system beyond the functions passed to [`System`](@ref). The
full constructor is

```julia
System(nl, lin, adj, dds!, x; linearise!, ddi=(), B=I, project=identity)
```

and two shorter ones leave out the adjoint operator, or both the linearised and the adjoint
operators (see [Choosing the constructor](#Choosing-the-constructor)). With `a` the field of an
`Orbit`, the arguments are the following.

| argument | call | does |
|---|---|---|
| `nl` | `nl(out, a)` | nonlinear operator $N(a)$ |
| `lin` | `lin(out, b)` | linearised operator $L\{u_0\}\, b$, about the point set by `linearise!` |
| `adj` | `adj(out, b)` | adjoint operator $L^+\{u_0\}\, b$, about the same point |
| `linearise!` | `linearise!(op, a)` | sets the linearisation point of `op`, `lin` or `adj`, to `a` |
| `dds!` | `dds!(out, a)` | derivative $\partial_s$ in rescaled time |
| `ddi` | `dd!(out, a)` for each `dd!` in the tuple | derivative $\partial_i$ along each drift direction |
| `B` | `precondition!(out, B, p)`, `precondition_adjoint!(out, B, p)` | apply $\mathcal{B}^{-1}$ and $\mathcal{B}^{-+}$ to an orbit |
| `project` | `project(a)` | project a field, in place, on a subspace invariant under the operators (see [Symmetric subspaces](#Symmetric-subspaces)) |

All write into `out` and return it. `x` is an orbit of the right shape, used to allocate the
workspace. Its parameter vector fixes the number of drift directions, which must equal
`length(ddi)`.

### Choosing the constructor

`System` has three constructors, which differ in the operators given.

```julia
System(nl, lin, adj, dds!, x; linearise!, ddi=(), B=I, project=identity)   # both methods
System(nl, lin, dds!, x; linearise!, ddi=(), B=I, project=identity)        # hookstep only
System(nl, dds!, x; difference=:forward, ddi=(), B=I, project=identity)    # hookstep only, Jacobian-free
```

| constructor | needs | serves | operators per Jacobian action |
|---|---|---|---|
| `System(nl, lin, adj, dds!, x; linearise!)` | `nl`, `lin`, `adj`, `linearise!` | [`LBFGS`](@ref) and [`NewtonHookstep`](@ref) | one `lin` |
| `System(nl, lin, dds!, x; linearise!)` | `nl`, `lin`, `linearise!` | [`NewtonHookstep`](@ref) | one `lin` |
| `System(nl, dds!, x; difference=:forward)` | `nl` | [`NewtonHookstep`](@ref) | one `nl`, plus one `nl` per Newton iteration |
| `System(nl, dds!, x; difference=:central)` | `nl` | [`NewtonHookstep`](@ref) | two `nl` |

Only L-BFGS applies the adjoint operator, for the gradient. A system built without it serves the
hookstep, and `solve!` with an `LBFGS` method throws an `ArgumentError`. The functions
`ReSolver.gradient!` and `ReSolver.jacobian_adjoint!` need the adjoint operator too.

The Jacobian-free constructor needs only the nonlinear operator. The hookstep then applies the
linearised operator about the current field $a$ to a field $b$ by a finite difference of `nl`,
with the step $\varepsilon = \delta\,(1 + \lVert a\rVert)/\lVert b\rVert$.

- `difference=:forward`, the default, takes
  $L b \approx \big(N(a + \varepsilon b) - N(a)\big)/\varepsilon$, with $\delta$ the square root
  of the machine precision of the fields. A Jacobian action costs one evaluation of `nl`, and
  $N(a)$ is evaluated once per Newton iteration, when the linearisation point is set.
- `difference=:central` takes
  $L b \approx \big(N(a + \varepsilon b) - N(a - \varepsilon b)\big)/2\varepsilon$, with $\delta$
  the cube root of the machine precision. It is of second order and costs two evaluations of `nl`
  per action. For a quadratic $N$ it is exact up to rounding.

The relative error of an action is about $10^{-8}$ for forward differences and
$5 \times 10^{-11}$ for central ones in double precision, and about $3 \times 10^{-4}$ and
$2 \times 10^{-5}$ in single precision. These errors lie far below the usual tolerance of the
Krylov solver, so the Newton steps hardly change. The residual is never approximated, and a
converged orbit satisfies the equations to the same tolerance as with the linearised operator.
For the search of [A first example](@ref) the forward and central versions take the same 9 Newton
iterations and 441 Jacobian actions as the exact Jacobian, and converge to the same period,
20.505745. In the examples an evaluation of `nl` costs about as much as one of `lin` (see Table 2
of [L-BFGS against the hookstep](@ref)). A forward action then costs about as much as an exact one,
and a central action about twice as much.

The Jacobian-free system is a quick way to test a model before its linearised operator is written.
The exact linearised and adjoint operators are needed for L-BFGS and give exact Jacobian actions.
The preconditioner is independent of the constructor, and a preconditioner built from a linearised
operator, such as the factorised one of the [Kolmogorov flow](examples/kolmogorov_flow.md) example,
brings its own.

### Requirements on the field type

The field `a` of an orbit can be of any type that meets four requirements. The first two are needed
for the code to run. The last two are needed for the gradient and the Jacobian to be exact, and for
the methods to converge as expected.

1. The type broadcasts in place, with `out .= a`, `out .= a .+ b`, `out .= α .* a .+ b` and the
   like, writing into an existing field without allocating. It supports `similar(a)` and
   `copy(a)`.
2. `LinearAlgebra.dot(a, b)` returns a real number, the inner product of two fields, and
   `norm(a) = sqrt(dot(a, a))`. This inner product defines every norm of the package, that of the
   residual, of the stopping criteria, of the trust region and of the tolerances. A discretisation
   of the continuous inner product, such as the mean over the grid points, makes them independent
   of the resolution.
3. `dds!` and every function in `ddi` are exactly skew-adjoint in that inner product,
   $\langle \partial u, v\rangle = -\langle u, \partial v\rangle$, as the derivatives of the
   continuous problem are on periodic functions. The gradient with respect to the frequency and the
   drift speeds relies on it.
4. `adj`, when given, is the exact adjoint of `lin` in the same inner product,
   $\langle w, L v\rangle = \langle L^+ w, v\rangle$ for all fields. It is the adjoint of the
   discrete operator, not the discretisation of the continuous adjoint. L-BFGS uses it for the
   gradient.

A plain `Array` meets requirements 1 and 2. The examples define their own field types, `KSField`
with values on a space-time grid and `LorenzField` with Fourier coefficients in time, to keep the
transforms planned and the operators free of allocations. With spectral derivatives, use an odd
number of points or modes in every periodic direction. An unpaired Nyquist mode breaks
requirement 3.

### Checking a system

Before a search, check the following, with random orbits `x`, `v` and `w` of the right shape.
Checks 1 and 2 apply to the operators given to the system.

1. `lin`, linearised about `x.a`, is the derivative of `nl`. The central difference of `nl` along
   `v.a` agrees with `lin` applied to `v.a` to $O(\varepsilon^2)$, and exactly for a quadratic
   operator.
2. `adj` is the adjoint of `lin`. The difference
   $\langle w, L v\rangle - \langle L^+ w, v\rangle$ is at the level of rounding.
3. The derivatives are skew-adjoint. The sum
   $\langle \partial u, v\rangle + \langle u, \partial v\rangle$ is at the level of rounding, for
   `dds!` and for every function in `ddi`.
4. `ReSolver.gradient!` and `ReSolver.jacobian!` agree with finite differences of
   `ReSolver.objective` and `ReSolver.residual!`, including the components along the frequency and
   the drift speeds.
5. `precondition_adjoint!` is the adjoint of `precondition!`,
   $\langle \mathcal{B}^{-1} p, q\rangle = \langle p, \mathcal{B}^{-+} q\rangle$.
6. The Jacobian has no null space at a converged orbit. On a tiny grid, form $\mathcal{J}$ column
   by column with `jacobian!`, as the scripts `examples/*/scripts/preconditioners.jl` do, and check
   that its smallest singular value is not zero. With a projection, form it in a basis of the
   subspace. A zero reveals a family of equivalent solutions that the phase conditions do not
   remove. The Galilean invariance of the Kuramoto–Sivashinsky equation is one. The projection
   `demean!` on the fields of zero mean removes it (see [Kuramoto–Sivashinsky equation](@ref)).

The first three checks in code read

```julia
linearise!(lin, x.a); linearise!(adj, x.a)

# 1. linearisation: central difference of nl
ε  = 1e-3
FD = (nl(similar(x.a), x.a .+ ε .* v.a) .- nl(similar(x.a), x.a .- ε .* v.a)) ./ 2ε
norm(FD .- lin(similar(x.a), v.a)) / norm(FD)                         # ≈ 1e-10 or less

# 2. adjoint: ⟨w, L v⟩ = ⟨L⁺ w, v⟩
dot(w.a, lin(similar(x.a), v.a)) - dot(adj(similar(x.a), w.a), v.a)   # ≈ rounding

# 3. skew-adjoint derivative
dot(dds!(similar(x.a), x.a), v.a) + dot(x.a, dds!(similar(v.a), v.a)) # ≈ rounding
```

The test suite of the package performs the first five checks on the Kuramoto–Sivashinsky model.

### Drift directions

`ddi` lists the derivatives along the directions in which the orbit drifts with an unknown speed.
For a periodic orbit, leave it empty, `ddi=()`, and give `p = [log ω]`. For a relative periodic
orbit, give one derivative and one speed per drift direction.

Include only directions of translational symmetry along which the solution is free to drift. Each
drift speed adds a column $-\partial_i u$ to the Jacobian and a phase condition
$\langle \partial_i u, \delta u\rangle = 0$ to the Newton system. If $\partial_i u$ vanishes,
both vanish, the system is singular and the drift speed is undetermined. This happens in two cases.

- The field is restricted to a subspace that translations do not preserve. An example is a
  discretisation of the Kuramoto–Sivashinsky equation that keeps only the odd solutions,
  $u(-x) = -u(x)$, to exploit its reflection symmetry. A translated odd function is no longer
  odd, so the orbits in that subspace cannot drift. The derivative $\partial_x u$ is even, its
  projection on the odd subspace is zero, and $x$ must be left out of `ddi`.
- The solution does not depend on that coordinate, as a two-dimensional solution in a
  three-dimensional domain. A translation along the third direction does nothing, and
  $\partial_i u = 0$.

The page [Search by root finding](@ref) derives the phase conditions.

### Symmetric subspaces

Many systems have discrete symmetries besides the translations. The Kuramoto–Sivashinsky equation
is equivariant under the reflection $u(x) \to -u(-x)$, and plane Couette flow under rotations and
shift-reflections. The solutions invariant under such a symmetry $S$, with $S u = u$, form a
subspace that the dynamics never leaves. Searching for orbits there is often desirable. The problem
is smaller, the orbits are those of the symmetric subspace studied in the literature, and a
continuous family of translated copies, which would make the Newton system singular, may be
excluded. The keyword `project` restricts a search to such a subspace without redefining the field
type or the operators,

```julia
F = System(nl, lin, adj, dds!, x; linearise!, B, project=odd!)
```

where `odd!(a)` overwrites the field `a` with its projection on the subspace,
$P a = (a + S a)/2$, and returns it. The search then projects the initial orbit, every residual,
every gradient, every Jacobian action and every application of the preconditioner. The method
relies on four conditions.

1. $S$ is an isometry with $S^2 = I$, such as a reflection or a half-period shift. Then
   $P = (I + S)/2$ is an orthogonal projection, self-adjoint in the inner product of the fields.
   Check that `dot(project(copy(a)), b) ≈ dot(a, project(copy(b)))`.
2. The system is equivariant, $S\,N(u) = N(S u)$, and $S$ commutes with $\partial_s$. Then
   $N$, $L\{u\}$ and $L^+\{u\}$ with $u$ in the subspace, and $\partial_s$, map the subspace into
   itself. Residuals, gradients and Jacobian actions of orbits in the subspace stay in it in exact
   arithmetic. The projections only remove rounding errors, which would otherwise grow wherever
   the subspace is unstable. Check that the residual of a symmetric orbit is symmetric, computed
   without `project`.
3. The generators of the continuous symmetries are chosen accordingly. `dds!` is always passed.
   $S$ acts on space only, the time translations of a symmetric orbit are symmetric, and the
   log-frequency keeps its phase condition. In `ddi`, pass exactly the derivatives along the
   translations that commute with $S$, whose orbits stay in the subspace. Leave out those that
   anticommute with it, along which the symmetric orbits cannot drift. For odd
   Kuramoto–Sivashinsky fields $S\,\partial_x = -\partial_x S$. The derivative $\partial_x u$ is
   even, its projection vanishes, and the orbits are periodic, with `ddi=()` and `p = [log ω]`.
   For a flow with a spanwise reflection, the streamwise derivative stays in `ddi` and the
   spanwise one goes.
4. Ideally the preconditioner commutes with $S$. If it does not, its output is projected anyway
   and the search still stays in the subspace, but the metric of the preconditioner and the
   symmetry interfere.

The user provides only `project`, the orthogonal projection applied in place to a field. The
operators `nl`, `lin` and `adj`, the derivatives and the preconditioner are those of the full space
and need no change. The package applies the projection inside the functions of the
[`System`](@ref), so that both methods inherit it.

| quantity | where the projection is applied |
|---|---|
| initial orbit | at the start of [`solve!`](@ref), for both methods |
| residual $r$ | in `ReSolver.residual!` |
| gradient $\nabla R = \mathcal{J}^+(r, 0)$ | in `ReSolver.gradient!`, through the adjoint Jacobian action |
| Jacobian actions $\mathcal{J}\,\delta p$ and $\mathcal{J}^+ w$ | in `ReSolver.jacobian!` and `ReSolver.jacobian_adjoint!` |
| $\mathcal{B}^{-1} p$ and $\mathcal{B}^{-+} p$ | in every application of the preconditioner |

L-BFGS then uses projected gradients. Its directions combine projected gradients, preconditioned
vectors and curvature pairs, so they stay in the subspace. The hookstep builds its Krylov basis from
the projected residual with projected Jacobian actions. The parameters of the orbit, frequency and
drift speeds, are not affected. If the operators are equivariant, the projections only remove
rounding errors. If they are not, the search solves the projected equations $P r = 0$ on the
subspace, a Galerkin approximation whose solutions are not orbits of the full system.

The two searches for Kuramoto–Sivashinsky differ as follows.

| search | orbit parameters | `ddi` | `project` | Newton system |
|---|---|---|---|---|
| relative periodic orbit, full space | `[log ω, c]` | `(ddx!,)` | `demean!` | $N_u + 1$ unknowns, the mean removed |
| periodic orbit, odd subspace | `[log ω]` | `()` | `odd!` | $\dim V + 1$ unknowns, $\dim V \approx N_u/2$ |

`dds!` is passed in both. The section
[Symmetric subspaces](theory/root_finding.md#Symmetric-subspaces) of the theory derives the change
of the block system.

The search in the full space uses `project=demean!`, the projection on the fields of zero mean. It
is not the fixed space of a symmetry, but the residual and the Jacobian actions stay in it, since
the right-hand side of the equation has zero mean. The projected gradient is the gradient on the
subspace. The odd fields have zero mean, so `odd!` removes the mean too.

The projected search works with fields of the full size. It saves neither memory nor operations per
action, unlike a discretisation that represents only the symmetric fields, such as a sine series for
odd fields. It saves work in the solvers. The Krylov spaces contain only symmetric fields, the
neutral directions of the symmetries that the subspace excludes never enter the Newton system, and
the converged orbit is exactly symmetric. The example
[Kuramoto–Sivashinsky in the odd subspace](@ref) compares the projected search with searches in the
full space.

## Preconditioners

Both methods converge at a rate set by the spectrum of the linearised residual. For a discretised
PDE that spectrum spans many orders of magnitude. The time derivative grows with the number of
modes in rescaled time, and the dissipation with the spatial resolution. The Krylov solver of the
hookstep then needs a Krylov space nearly as large as the problem to solve each Newton system, and
L-BFGS behaves like steepest descent. A preconditioner $\mathcal{B}$ removes this dependence. It is
an operator on orbits that is cheap to invert and approximates the stiff part of the problem. It
is used in two places.

- In [`NewtonHookstep`](@ref) the Newton systems are solved with right preconditioning. GMRES runs
  on $\mathcal{J}\mathcal{B}^{-1}$ and the step is $\delta p = \mathcal{B}^{-1} z$, so that the
  Krylov space needed is small when $\mathcal{B}$ approximates $\mathcal{J}$. The trust region
  bounds $\lVert \mathcal{B}\,\delta p\rVert$.
- In [`LBFGS`](@ref) the initial inverse Hessian of the two-loop recursion is
  $\theta\, \mathcal{B}^{-1} \mathcal{B}^{-+}$. The search then works in the metric
  $\langle \mathcal{B} p, \mathcal{B} q\rangle$.

The examples show that, for the Kuramoto–Sivashinsky equation, the hookstep converges only with a
preconditioner. The page [Preconditioning](@ref) has the theory.

A preconditioner is any type for which two functions are extended. They apply the inverse of
$\mathcal{B}$ and its adjoint in the inner product of the orbits.

```julia
struct MyPreconditioner … end

ReSolver.precondition!(out::Orbit, B::MyPreconditioner, p::Orbit)         = …  # out = B⁻¹ p
ReSolver.precondition_adjoint!(out::Orbit, B::MyPreconditioner, p::Orbit) = …  # out = B⁻⁺ p
ReSolver.relinearise!(B::MyPreconditioner, x::Orbit)                     = …  # optional

F = System(nl, lin, adj, dds!, x; linearise!, ddi, B=MyPreconditioner(…))
```

A preconditioner built from the operators about an orbit, from its frequency, drift speeds and
field, can follow the search. [`NewtonHookstep`](@ref) calls `relinearise!(B, x)` with the current
orbit at the start of every Newton iteration, unless `relinearise=false`, and the method rebuilds
`B` in place. Without a method nothing happens, and the preconditioner stays as built. L-BFGS never
calls it, since its metric must stay the same for the whole run.

Both functions act on the field and on the parameters of the orbit. The default `B = I` applies no
preconditioning. For a real, positive operator diagonal in an orthogonal basis, such as the
`:linear` preconditioner of the Kuramoto–Sivashinsky example, $\mathcal{B}^{-+} = \mathcal{B}^{-1}$
and the second function can call the first. For a complex one, such as the `:jacobian`
preconditioners of the Lorenz and Kuramoto–Sivashinsky examples, diagonal or block diagonal in
Fourier, the second function applies the conjugate transpose of each block. In the diagonal case
this is the complex conjugate of each multiplier. The page [Preconditioning](@ref) discusses the
choice of $\mathcal{B}$, and the examples measure its effect.

A preconditioner may also be applied inexactly, by a few steps of an inner iterative solver, as the
factorised preconditioner of the [Kolmogorov flow](examples/kolmogorov_flow.md) example is. It is
then a slightly different operator at each application. Use it with
`NewtonHookstep(flexible=true)`, which forms the step from the preconditioned vectors and so stays
consistent with whatever the preconditioner did. Such a preconditioner has no exact adjoint, and it
does not suit L-BFGS, whose metric must be fixed. Its `precondition_adjoint!` may throw an error.

## Methods and options

[`LBFGS`](@ref) minimises $\tfrac12\lVert r\rVert^2$. It needs a system built with the adjoint
operator, and it has the following options.

| option | default | meaning |
|---|---|---|
| `memory` | `10` | number of curvature pairs. `0` gives gradient descent, steepest descent in the metric of the preconditioner scaled by Barzilai–Borwein (see [Search by optimisation](theory/optimisation.md)) |
| `maxiter` | `100` | largest number of iterations |
| `tol` | `1e-10` | stop when $\lVert r\rVert$ < `tol` |
| `verbose`, `io` | `true`, `stdout` | one line per iteration, and where to print it |
| `callback` | `info -> false` | called at the start and after every iteration, `true` stops |

[`NewtonHookstep`](@ref) solves $r = 0$, with the following options.

| option | default | meaning |
|---|---|---|
| `krylov_dim` | `50` | largest dimension of the Krylov space |
| `krylov_tol` | `1e-3` | relative residual of the Newton system that stops the Krylov space |
| `Δ`, `Δmax` | `1e-2`, `1` | initial and largest trust-region radius |
| `eta` | `1e-3` | smallest ratio of actual to predicted reduction of an accepted step |
| `maxiter` | `20` | largest number of Newton iterations |
| `tol` | `1e-10` | stop when $\lVert r\rVert$ < `tol` |
| `flexible` | `false` | keep the preconditioned Arnoldi vectors and form the step from them (flexible GMRES), for a preconditioner applied inexactly |
| `relinearise` | `true` | rebuild the preconditioner about the current orbit at every Newton iteration, by `relinearise!(B, x)` |
| `verbose`, `io` | `true`, `stdout` | one line per iteration, and where to print it |
| `callback` | `info -> false` | called at the start and after every iteration, `true` stops |

With `verbose=true` the hookstep prints one line per Newton iteration. It shows the residual norm,
the frequency, the trust-region radius after the update, the ratio of actual to predicted
reduction, the dimension of the Krylov space and the relative residual $\lVert g - Hy\rVert/\beta$
reached by the inner solve. The last two columns are the kind of the accepted step, `newton` for
the full Newton step inside the trust region or `hook` for a hookstep on its boundary, and the
number of trial steps rejected before it, each of which shrank the radius. For the search of
[A first example](@ref) the output reads

```
  iter      ‖r‖          ω          radius       ρ      krylov    linear    step   rejected
     0  7.8181e-02  3.141593e-01  1.000e-01       NaN       0  NaN
     1  4.0945e-02  3.031538e-01  2.000e-01     0.990      54  6.25e-04    hook         0
     ⋮
     5  1.4519e-02  3.085969e-01  2.000e-01     0.610      49  7.24e-04    hook         0
     6  1.1808e-03  3.065405e-01  2.000e-01     0.993      45  8.20e-04  newton         0
     7  1.4612e-06  3.064106e-01  2.000e-01     1.000      45  8.17e-04  newton         0
     8  1.1788e-09  3.064110e-01  2.000e-01     1.000      46  8.08e-04  newton         0
     9  1.1627e-12  3.064110e-01  2.000e-01     1.000      47  9.89e-04  newton         0
```

A sequence of hooksteps followed by Newton steps with a fast decrease of the residual is the typical
pattern. Far from the solution the trust region limits the step, and the linear model, solved only
in part, gives a good direction of descent. Close to the solution the full Newton step is taken.
The convergence is then quadratic, or linear with ratio `krylov_tol` if the Newton systems are
solved loosely. If every trial step is rejected until the radius falls to the level of rounding,
the hookstep stops and, with `verbose=true`, prints `trust region collapsed`. No step in the Krylov
space then reduces the residual. This happens when `krylov_dim` is too small for the Newton systems
to be solved to any useful accuracy (see [L-BFGS against the hookstep](@ref)).

The examples suggest some practical guidance.

- Start with L-BFGS from a poor guess, and switch to the hookstep once $\lVert r\rVert$ is small.
- With a good preconditioner a loose inner tolerance, `krylov_tol` between $10^{-1}$ and
  $10^{-4}$, minimises the total cost. Solving the Newton system exactly is wasteful.
- The radius `Δ` is measured in the metric of the preconditioner, so its scale changes with
  $\mathcal{B}$.
- The best preconditioner need not be the same for the two methods. A positive, self-adjoint
  $\mathcal{B}$ bounded below gives a balanced metric for L-BFGS. The hookstep gains most from a
  $\mathcal{B}$ close to the space-time operator, phase included. Build a second `System` with the
  same functions and a different `B` for the hookstep phase (see
  [Kuramoto–Sivashinsky equation](@ref)).

## Tracing a search

Both methods call a user function, the `callback` option, once before the first iteration and once
after every iteration. They pass it everything that describes the state of the search.

- [`LBFGS`](@ref) calls it after evaluating the residual and the gradient at the initial guess.
  Then it calls it after every iteration, once the line search has accepted a step and the gradient
  at the new point has been computed.
- [`NewtonHookstep`](@ref) calls it after evaluating the residual at the initial guess. Then it
  calls it after every Newton iteration, once the trust-region loop has accepted a step. By then
  the Krylov space has been built, the hookstep computed, and any rejected trial step shortened.

The argument is a named tuple `info` with the following fields.

| field | content |
|---|---|
| `iter` | the iteration number of the method, `0` before the first iteration |
| `x` | the current orbit, after the accepted step |
| `res` | the residual norm $\lVert r\rVert$ at `x` |
| `evaluations` | cumulative numbers of residuals, gradients, Jacobian actions, adjoint Jacobian actions and preconditioner applications of the system, as a named tuple `(residual, gradient, jacobian, adjoint, precondition)` |
| `krylov` | for the hookstep, the relative residual of the Newton system $\lVert g - Hy\rVert/\beta$ after each Arnoldi step of the iteration. Empty for L-BFGS and at `iter = 0` |
| `step` | for the hookstep, `"newton"` if the accepted step is the full Newton step, inside the trust region, or `"hook"` if it lies on its boundary. Empty for L-BFGS and at `iter = 0` |
| `trust` | for the hookstep, a named tuple `(radius, length, ratio, rejected)`. These are the trust-region radius with which the accepted step was computed (the initial radius at `iter = 0`), the length of that step in the metric of the preconditioner, the ratio of the actual to the predicted reduction of $\tfrac12\lVert r\rVert^2$, and the number of trial steps rejected before it. `NaN` and `0` for L-BFGS |

The callback returns `true` to stop the search and `false` to continue. `info.x` is the orbit being
solved for, and it is overwritten at the next iteration. Copy it to keep it.

### A custom callback

Any function of `info` can be passed. This one prints the period at every iteration and stops when
it leaves an interval.

```julia
function watch(info)
    T = 2π / ReSolver.frequency(info.x)
    println("iteration ", info.iter, ": ‖r‖ = ", info.res, ", T = ", T)
    return !(10 < T < 30)                   # true stops the search
end

solve!(x, F, NewtonHookstep(callback=watch, verbose=false))
```

This one keeps a copy of the orbit at every iteration.

```julia
orbits = Orbit[]
solve!(x, F, LBFGS(callback = info -> (push!(orbits, copy(info.x)); false)))
```

### Recording everything with `Trace`

A [`Trace`](@ref) is a callback that records all the fields of `info` but the orbit, of which it
keeps only the parameters, with the time elapsed since the first record. The field `trust.length`
of `info` is stored as `steplength`. The same trace can follow several calls to `solve!`, such as
L-BFGS followed by the hookstep.

```julia
trace = Trace()
solve!(x, F, LBFGS(maxiter=300, callback=trace))
solve!(x, F, NewtonHookstep(callback=trace))

trace.iter         # iteration numbers, restarting at 0 with each call
trace.res          # residual norms
trace.p            # parameters [log ω, c₁, …] at every record
trace.time         # seconds since the first record
trace.evaluations  # cumulative evaluations, e.g. trace.evaluations[end].jacobian
trace.krylov       # inner-solve histories, one vector per record
trace.step         # "newton", "hook", or "" for L-BFGS and the starts
trace.radius       # trust-region radius of each accepted step, NaN for L-BFGS
trace.steplength   # its length in the metric of the preconditioner
trace.ratio        # actual over predicted reduction of ½‖r‖²
trace.rejected     # trial steps rejected before it
```

`trace.krylov[j]` is the history of the inner solve of the iteration recorded at position `j`. Its
`n`-th entry is the relative residual of the Newton system after `n` Arnoldi steps. Its length is
the dimension of the Krylov space, and its last entry the accuracy to which the Newton system was
solved. The code below draws, with CairoMakie, the residual against the Jacobian actions and the
convergence of GMRES in every Newton iteration of a hookstep search.

```julia
using CairoMakie

trace = Trace()
solve!(x, F, NewtonHookstep(callback=trace))

fig = Figure(size=(900, 380))

# residual against the cost, hooksteps as open symbols
ax  = Axis(fig[1, 1]; xlabel="Jacobian actions", ylabel="‖r‖", yscale=log10)
jac = [e.jacobian for e in trace.evaluations]
scatterlines!(ax, jac, trace.res; color=:crimson,
              markercolor=[s == "hook" ? :white : :crimson for s in trace.step],
              strokecolor=:crimson, strokewidth=1)

# GMRES: one curve per Newton iteration
ax = Axis(fig[1, 2]; xlabel="Arnoldi step", ylabel="‖g - Hy‖ / β", yscale=log10)
for (j, k) in enumerate(trace.krylov[2:end])
    lines!(ax, 1:length(k), k; label="iteration $j")
end
axislegend(ax)

save("convergence.png", fig)
```

All the convergence figures of the examples are drawn from such traces.
