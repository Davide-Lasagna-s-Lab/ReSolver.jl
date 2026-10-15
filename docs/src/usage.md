# Usage

## Installation

ReSolver is not registered. From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ReSolver.jl")
```

The package depends only on the Julia standard library. The examples have their own environment,
`examples/Project.toml`, with FFTW and CairoMakie; from the root of the repository,

```
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/kuramoto_sivashinsky/example.jl
```

## The three objects

A search involves three objects.

- An [`Orbit`](@ref) holds the unknowns: the space-time field `a`, in any discretisation, and the
  parameter vector `p = [log ω, c₁, …, c_m]`, the logarithm of the frequency followed by one drift
  speed per drift direction. `ReSolver.frequency(x)` returns ``\omega``.
- A [`System`](@ref) holds the functions that define the problem: the nonlinear operator, its
  linearisation and adjoint, the derivatives, and optionally a preconditioner.
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

The search knows nothing about the system beyond the functions passed to [`System`](@ref):

```julia
System(nl, lin, adj, dds!, x; linearise!, ddi=(), B=I, project=identity)
```

With `a` the field of an `Orbit`:

| argument | call | does |
|---|---|---|
| `nl` | `nl(out, a)` | nonlinear operator ``N(a)`` |
| `lin` | `lin(out, b)` | linearised operator ``L\{u_0\}\, b``, about the point set by `linearise!` |
| `adj` | `adj(out, b)` | adjoint operator ``L^+\{u_0\}\, b``, about the same point |
| `linearise!` | `linearise!(op, a)` | sets the linearisation point of `op`, `lin` or `adj`, to `a` |
| `dds!` | `dds!(out, a)` | derivative ``\partial_s`` in rescaled time |
| `ddi` | `dd!(out, a)` for each `dd!` in the tuple | derivative ``\partial_i`` along each drift direction |
| `B` | `precondition!(out, B, p)`, `precondition_adjoint!(out, B, p)` | apply ``B^{-1}`` and ``B^{-+}`` to an orbit |
| `project` | `project(a)` | project a field, in place, on a subspace invariant under the operators (see [Symmetric subspaces](#Symmetric-subspaces)) |

All write into `out` and return it. `x` is an orbit of the right shape, used to allocate the
workspace; its parameter vector fixes the number of drift directions, which must equal
`length(ddi)`.

### Requirements on the field type

The field `a` of an orbit can be of any type, provided it satisfies the following requirements. The
first two are needed for the code to run; the last two for the gradient and the Jacobian to be
exact, and the methods to converge as expected.

1. **Vector-space operations in place.** The type broadcasts, with `out .= a`, `out .= a .+ b`,
   `out .= α .* a .+ b` and the like, writing into an existing field without allocating, and
   supports `similar(a)` and `copy(a)`.
2. **A real inner product.** `LinearAlgebra.dot(a, b)` returns a real number, the inner product of
   two fields, and `norm(a) = sqrt(dot(a, a))`. It defines every norm of the package: the residual,
   the stopping criteria, the trust region and the tolerances. A discretisation of the continuous
   inner product, e.g. the mean over the grid points, makes them independent of the resolution.
3. **Skew-adjoint derivatives.** `dds!` and every function in `ddi` are exactly skew-adjoint in
   that inner product, ``\langle \partial u, v\rangle = -\langle u, \partial v\rangle``, as the
   derivatives of the continuous problem are on periodic functions. The gradient of the frequency
   and of the drift speeds relies on it.
4. **An exact adjoint.** `adj` is the exact adjoint of `lin` in the same inner product,
   ``\langle w, L v\rangle = \langle L^+ w, v\rangle``, for all fields: the adjoint of the discrete
   operator, not the discretisation of the continuous adjoint. L-BFGS uses it for the gradient.

A plain `Array` satisfies 1 and 2; the examples define their own field types, `KSField` with
values on a space-time grid and `LorenzField` with Fourier coefficients in time, to keep the
transforms planned and the operators allocation-free. With spectral derivatives, use an odd number
of points or modes in every periodic direction: an unpaired Nyquist mode breaks requirement 3.

### Checking a system

Before a search, check the following, with random orbits `x`, `v`, `w` of the right shape.

1. **Linearisation.** `lin`, linearised about `x.a`, is the derivative of `nl`: the central
   difference of `nl` along `v.a` agrees with `lin` applied to `v.a`, to ``O(\varepsilon^2)``, and
   exactly for a quadratic operator.
2. **Adjoint.** ``\langle w, L v\rangle - \langle L^+ w, v\rangle`` is at the level of rounding.
3. **Skew-adjoint derivatives.** ``\langle \partial u, v\rangle + \langle u, \partial v\rangle``
   is at the level of rounding, for `dds!` and for every function in `ddi`.
4. **Gradient and Jacobian of the residual.** `ReSolver.gradient!` and `ReSolver.jacobian!`
   agree with finite differences of `ReSolver.objective` and `ReSolver.residual!`, including the
   components along the frequency and the drift speeds.
5. **Preconditioner.** `precondition_adjoint!` is the adjoint of `precondition!`:
   ``\langle B^{-1} p, q\rangle = \langle p, B^{-+} q\rangle``.
6. **No null space.** On a tiny grid, form ``\mathcal{J}`` column by column with `jacobian!`, as in
   the `preconditioners.jl` scripts of the examples, and check that its smallest singular value at a
   converged orbit is not zero: a zero reveals a family of equivalent solutions that the phase
   conditions do not remove, such as the Galilean invariance of the Kuramoto–Sivashinsky equation,
   which must be removed by pinning (see [Kuramoto–Sivashinsky equation](@ref)).

The first three, in code:

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
Leave it empty, `ddi=()`, and give `p = [log ω]` for a periodic orbit; give one derivative and one
speed per drift direction for a relative periodic orbit.

Include only directions of translational symmetry along which the solution is free to drift. Each
drift speed adds a column ``-\partial_i u`` to the Jacobian and a phase condition
``\langle \partial_i u, \delta u\rangle = 0`` to the Newton system; if ``\partial_i u`` vanishes,
both vanish and the system is singular, with the drift speed left undetermined. This happens in two
cases:

- **the field is restricted to a subspace that translations do not preserve.** For example, a
  discretisation of the Kuramoto–Sivashinsky equation that keeps only the odd solutions,
  ``u(-x) = -u(x)``, to exploit its reflection symmetry: a translated odd function is no longer odd,
  so the orbits in that subspace cannot drift; ``\partial_x u`` is even, its projection on the odd
  subspace is zero, and ``x`` must be left out of `ddi`;
- **the solution does not depend on that coordinate**, e.g. a two-dimensional solution in a
  three-dimensional domain: a translation along the third direction does nothing, and
  ``\partial_i u = 0``.

See [Search by root finding](@ref) for the phase conditions.

### Symmetric subspaces

Many systems have discrete symmetries besides the translations: the Kuramoto–Sivashinsky equation
is equivariant under the reflection ``u(x) \to -u(-x)``, plane Couette flow under rotations and
shift-reflections. The solutions invariant under such a symmetry ``S``, ``S u = u``, form a
subspace that the dynamics never leaves, and searching for orbits there is often desirable: the
problem is smaller, the orbits are those of the symmetric subspace studied in the literature, and
a continuous family of translated copies, which would make the Newton system singular, may be
excluded. The keyword `project` restricts a search to such a subspace without redefining the field
type or the operators:

```julia
F = System(nl, lin, adj, dds!, x; linearise!, B, project=odd!)
```

where `odd!(a)` overwrites the field `a` with its projection on the subspace,
``P a = (a + S a)/2``, and returns it. The search then projects the initial orbit, every residual,
every gradient, every Jacobian action and every application of the preconditioner. The method
relies on four conditions.

1. **``S`` is an isometry with ``S^2 = I``**, a reflection or a half-period shift for instance, so
   that ``P = (I + S)/2`` is an orthogonal projection, self-adjoint in the inner product of the
   fields. Check that `dot(project(copy(a)), b) ≈ dot(a, project(copy(b)))`.
2. **The system is equivariant**, ``S\,N(u) = N(S u)``, and ``S`` commutes with ``\partial_s``.
   Then ``N``, ``L\{u\}`` and ``L^+\{u\}`` with ``u`` in the subspace, and ``\partial_s``, map the
   subspace into itself: residuals, gradients and Jacobian actions of orbits in the subspace stay
   in it, in exact arithmetic. The projections only remove rounding errors, which would otherwise
   grow wherever the subspace is unstable. Check that the residual of a symmetric orbit is
   symmetric, computed without `project`.
3. **The generators of the continuous symmetries are chosen accordingly.** `dds!` is always
   passed: ``S`` acts on space only, the time translations of a symmetric orbit are symmetric, and
   the log-frequency keeps its phase condition. In `ddi` pass exactly the derivatives along the
   translations that commute with ``S``, whose orbits stay in the subspace; leave out those that
   anticommute with it, along which the symmetric orbits cannot drift. For odd
   Kuramoto–Sivashinsky fields, ``S\,\partial_x = -\partial_x S``: ``\partial_x u`` is even, its
   projection vanishes, and the orbits are periodic, `ddi=()` with `p = [log ω]`. For a flow with a
   spanwise reflection, the streamwise derivative stays in `ddi` and the spanwise one goes.
4. **The preconditioner commutes with ``S``**, ideally. If it does not, its output is projected
   anyway, and the search still stays in the subspace, but the metric of the preconditioner and the
   symmetry interfere.

**What the package projects, and what the user provides.** The user provides only `project`, the
orthogonal projection, applied in place to a field; the operators `nl`, `lin`, `adj`, the
derivatives and the preconditioner are those of the full space and need no change. The package
applies the projection inside the functions of the [`System`](@ref), so that both methods inherit it:

| quantity | where the projection is applied |
|---|---|
| initial orbit | at the start of [`solve!`](@ref), for both methods |
| residual ``r`` | in `ReSolver.residual!` |
| gradient ``\nabla R = \mathcal{J}^+(r, 0)`` | in `ReSolver.gradient!`, through `ReSolver.jacobian_adjoint!` |
| Jacobian actions ``\mathcal{J}\,\delta p`` and ``\mathcal{J}^+ w`` | in `ReSolver.jacobian!` and `ReSolver.jacobian_adjoint!` |
| ``B^{-1} p`` and ``B^{-+} p`` | in every application of the preconditioner |

L-BFGS then uses projected gradients, and its directions, combinations of projected gradients,
preconditioned vectors and curvature pairs, stay in the subspace; the hookstep builds its Krylov
basis from the projected residual with projected Jacobian actions. The parameters of the orbit,
frequency and drift speeds, are not affected. If the operators are equivariant, the projections only
remove rounding errors. If they are not, the search solves the projected equations ``P r = 0`` on
the subspace, a Galerkin approximation, whose solutions are not orbits of the full system.

In summary, for Kuramoto–Sivashinsky:

| search | orbit parameters | `ddi` | `project` | Newton system |
|---|---|---|---|---|
| relative periodic orbit, full space | `[log ω, c]` | `(ddx!,)` | `identity` | ``N_u + 2`` unknowns |
| periodic orbit, odd subspace | `[log ω]` | `()` | `odd!` | ``\dim V + 1`` unknowns, ``\dim V \approx N_u/2`` |

with `dds!` passed in both; the change of the block system is derived in
[Symmetric subspaces](theory/root_finding.md#Symmetric-subspaces) of the theory.

The projected search works with fields of the full size: it saves neither memory nor operations
per action, unlike a discretisation that represents only the symmetric fields, a sine series for
odd fields for instance. What it saves is the work of the solvers. The Krylov spaces contain only
symmetric fields, the neutral directions of the symmetries that the subspace excludes never enter
the Newton system, and the converged orbit is exactly symmetric. The example
[Kuramoto–Sivashinsky in the odd subspace](@ref) compares the projected search with searches in
the full space.

## Preconditioners

Both methods converge at a rate set by the spectrum of the linearised residual, and for a
discretised PDE that spectrum spans many orders of magnitude: the time derivative grows with the
number of modes in rescaled time, the dissipation with the spatial resolution. The Krylov solver of
the hookstep then needs a Krylov space nearly as large as the problem to solve each Newton system,
and L-BFGS behaves like steepest descent. A preconditioner ``B``, an operator on orbits that is
cheap to invert and approximates the stiff part of the problem, removes this dependence. It is used
in two places:

- in [`NewtonHookstep`](@ref), the Newton systems are solved with right preconditioning: GMRES runs
  on ``\mathcal{J}B^{-1}`` and the step is ``\delta p = B^{-1} z``, so that the Krylov space needed
  is small when ``B`` approximates ``\mathcal{J}``; the trust region bounds ``\lVert B\,\delta
  p\rVert``;
- in [`LBFGS`](@ref), the initial inverse Hessian of the two-loop recursion is ``\theta\, B^{-1}
  B^{-+}``, i.e. the search works in the metric ``\langle B p, B q\rangle``.

The examples show that, for the Kuramoto–Sivashinsky equation, the hookstep converges only with a
preconditioner. The theory is in [Preconditioning](@ref).

A preconditioner is any type for which two functions are extended, applying the inverse of ``B``
and its adjoint in the inner product of the orbits:

```julia
struct MyPreconditioner … end

ReSolver.precondition!(out::Orbit, B::MyPreconditioner, p::Orbit)         = …  # out = B⁻¹ p
ReSolver.precondition_adjoint!(out::Orbit, B::MyPreconditioner, p::Orbit) = …  # out = B⁻⁺ p

F = System(nl, lin, adj, dds!, x; linearise!, ddi, B=MyPreconditioner(…))
```

Both act on the field and on the parameters of the orbit. The default `B = I` applies no
preconditioning. For a real, positive operator diagonal in an orthogonal basis, as the `:linear`
preconditioner of the Kuramoto–Sivashinsky example, ``B^{-+} = B^{-1}`` and the second function can
call the first; for a complex one, as the `:jacobian` preconditioners of both examples, diagonal
or block diagonal in Fourier, the second applies the conjugate transpose of each block, the complex
conjugate of each multiplier in the diagonal case. The choice of ``B``
and its effect are discussed in [Preconditioning](@ref) and measured in the examples.

## Methods and options

[`LBFGS`](@ref) minimises ``\tfrac12\lVert r\rVert^2``:

| option | default | meaning |
|---|---|---|
| `memory` | `10` | number of curvature pairs |
| `maxiter` | `100` | largest number of iterations |
| `tol` | `1e-10` | stop when ``\lVert r\rVert`` < `tol` |
| `verbose`, `io` | `true`, `stdout` | one line per iteration, and where to print it |
| `callback` | `info -> false` | called at the start and after every iteration; `true` stops |

[`NewtonHookstep`](@ref) solves ``r = 0``:

| option | default | meaning |
|---|---|---|
| `krylov_dim` | `50` | largest dimension of the Krylov space |
| `krylov_tol` | `1e-3` | relative residual of the Newton system that stops the Krylov space |
| `Δ`, `Δmax` | `1e-2`, `1` | initial and largest trust-region radius |
| `eta` | `1e-3` | smallest ratio of actual to predicted reduction of an accepted step |
| `maxiter` | `20` | largest number of Newton iterations |
| `tol` | `1e-10` | stop when ``\lVert r\rVert`` < `tol` |
| `verbose`, `io` | `true`, `stdout` | one line per iteration, and where to print it |
| `callback` | `info -> false` | called at the start and after every iteration; `true` stops |

With `verbose=true` the hookstep prints, for every Newton iteration, the residual norm, the
frequency, the trust-region radius, the ratio of actual to predicted reduction, the dimension of the
Krylov space, the relative residual ``\lVert g - Hy\rVert/\beta`` reached by the inner solve, the
kind of the accepted step, `newton` for the full Newton step inside the trust region or `hook` for a
hookstep on its boundary, and the number of trial steps rejected before it, each of which shrank the
radius:

```
  iter      ‖r‖          ω          radius       ρ      krylov    linear    step   rejected
     0  7.8181e-02  3.141593e-01  1.000e-01       NaN       0  NaN
     1  3.6409e-02  3.001349e-01  2.000e-01     0.993     150  4.60e-01    hook         0
     ⋮
     6  3.1069e-03  3.067173e-01  2.000e-01     0.958      46  9.65e-04  newton         0
     7  8.5112e-06  3.064105e-01  2.000e-01     1.000      45  9.07e-04  newton         0
```

A sequence of hooksteps followed by Newton steps with a fast decrease of the residual is the typical
pattern: far from the solution the trust region limits the step, and the linear model, solved only
in part, is used to choose a good direction of descent; close to it the full Newton step is taken
and the convergence becomes quadratic, or linear with ratio `krylov_tol` if the Newton systems are
solved loosely. If every trial step is rejected until the radius falls to the level of rounding, the
hookstep stops and, with `verbose=true`, prints `trust region collapsed`: no step in the Krylov
space reduces the residual. This happens when `krylov_dim` is too small for the Newton systems to
be solved to any useful accuracy (see [L-BFGS against the hookstep](@ref)).

Some practical guidance, from the examples:

- start with L-BFGS from a poor guess, and switch to the hookstep once ``\lVert r\rVert`` is small;
- with a good preconditioner a loose inner tolerance, `krylov_tol` between ``10^{-1}`` and
  ``10^{-4}``, minimises the total cost; solving the Newton system exactly is wasteful;
- the radius `Δ` is measured in the metric of the preconditioner, so its scale changes with ``B``;
- the best preconditioner need not be the same for the two methods: a positive, self-adjoint ``B``
  bounded below gives a balanced metric for L-BFGS, while the hookstep gains most from a ``B`` close
  to the space-time operator, phase included. Build a second `System` with the same functions and a
  different `B` for the hookstep phase (see [Kuramoto–Sivashinsky equation](@ref)).

## Tracing a search

Both methods are fully traceable: they call a user function, the `callback` option, once before the
first iteration and once after every iteration, and pass it everything that describes the state of
the search.

- [`LBFGS`](@ref) calls it after evaluating the residual and the gradient at the initial guess, and
  then after every iteration, once the line search has accepted a step and the gradient at the new
  point has been computed.
- [`NewtonHookstep`](@ref) calls it after evaluating the residual at the initial guess, and then
  after every Newton iteration, once the trust-region loop has accepted a step, i.e. after the
  Krylov space has been built, the hookstep computed, and any rejected trial steps shortened.

The argument is a named tuple `info` with:

| field | content |
|---|---|
| `iter` | the iteration number of the method, `0` before the first iteration |
| `x` | the current orbit, after the accepted step |
| `res` | the residual norm ``\lVert r\rVert`` at `x` |
| `evaluations` | cumulative numbers of residuals, gradients, Jacobian actions, adjoint Jacobian actions and preconditioner applications of the system, as a named tuple `(residual, gradient, jacobian, adjoint, precondition)` |
| `krylov` | for the hookstep, the relative residual of the Newton system ``\lVert g - Hy\rVert/\beta`` after each Arnoldi step of the iteration; empty for L-BFGS and at `iter = 0` |
| `step` | for the hookstep, `"newton"` if the accepted step is the full Newton step, inside the trust region, `"hook"` if it lies on its boundary; empty for L-BFGS and at `iter = 0` |

The callback returns `true` to stop the search, `false` to continue. `info.x` is the orbit being
solved for, overwritten at the next iteration: copy it to keep it.

### A custom callback

Any function of `info` can be passed. For example, to print the period at every iteration and stop
when it leaves an interval:

```julia
function watch(info)
    T = 2π / ReSolver.frequency(info.x)
    println("iteration ", info.iter, ": ‖r‖ = ", info.res, ", T = ", T)
    return !(10 < T < 30)                   # true stops the search
end

solve!(x, F, NewtonHookstep(callback=watch, verbose=false))
```

or to keep a copy of the orbit at every iteration:

```julia
orbits = Orbit[]
solve!(x, F, LBFGS(callback = info -> (push!(orbits, copy(info.x)); false)))
```

### Recording everything: `Trace`

A [`Trace`](@ref) is a callback that records all the fields of `info` but the orbit, keeping only its
parameters, with the time elapsed since the first record. The same trace can follow several calls to
`solve!`, e.g. L-BFGS followed by the hookstep:

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
```

`trace.krylov[j]` is the history of the inner solve of the iteration recorded at position `j`: its
`n`-th entry is the relative residual of the Newton system after `n` Arnoldi steps, so its length
is the dimension of the Krylov space, and its last entry the accuracy to which the Newton system was
solved. For example, with CairoMakie, the residual against the Jacobian actions, and the convergence
of GMRES in every Newton iteration of a hookstep search:

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
