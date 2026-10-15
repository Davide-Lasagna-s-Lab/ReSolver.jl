# Kuramoto–Sivashinsky in the odd subspace

On domains larger than $L = 22$ the Kuramoto–Sivashinsky equation is often studied in the
subspace of odd solutions, $u(-x) = -u(x)$. There the dynamics is still chaotic but the continuous
translation symmetry is absent. Lan & Cvitanović (2008) computed periodic orbits in this subspace
on $L = 38.5$, and Lasagna (2018) large sets of them on $L = 39$. This page searches for a periodic
orbit in the odd subspace on $L = 39$ with the keyword `project` of [`System`](@ref), and explains
why the search then needs no drift speed. The script is
`examples/kuramoto_sivashinsky/scripts/symmetric.jl`, and the model is that of the
[Kuramoto–Sivashinsky equation](@ref) page.

## The odd subspace

The equation is equivariant under the reflection $S u(x) = -u(-x)$. If $u$ is a solution, so is
$S u$. The fields fixed by the reflection, $S u = u$, are the odd fields. They form a subspace $V$
that the dynamics never leaves, since for odd $u$ the terms $u\,\partial_x u$, $\partial_x^2 u$ and
$\partial_x^4 u$ are odd, and so is the right-hand side. Odd fields are sine series, with purely
imaginary coefficients $\hat u_k$. They have zero mean. The projection on the odd fields therefore
removes the mean too, and takes the place of `demean!`, which removes the mean in the full space.

The grid used below has $N_x = 65$ points in space and $N_s = 73$ in rescaled time. An odd field on
it is determined by $(N_x - 1)/2 = 32$ values per time level. The search in the odd subspace
therefore has $32 \times 73 + 1 = 2\,337$ real unknowns, the field and the log-frequency, against
$65 \times 73 + 1 = 4\,746$ for a relative periodic orbit in the full space, without the mean.

## The complete example

The projection on the odd fields, $P u(x) = \big(u(x) - u(-x)\big)/2$, is computed on the grid,
which the reflection maps onto itself. The point $x_j = (j - 1) L/N_x$ goes to the point
$j' = N_x - j + 2$, modulo $N_x$, and $x = 0$ is fixed. The projection is the function `odd!` of
`ks.jl`.

```julia
function odd!(u::KSField)
    Nx = u.g.Nx

    for l in axes(u.data, 2), j in 1:(Nx + 1) ÷ 2
        j′ = mod(Nx - j + 1, Nx) + 1
        a  = u.data[j, l]
        b  = u.data[j′, l]

        u.data[j, l]  = (a - b) / 2
        u.data[j′, l] = (b - a) / 2
    end

    return u
end
```

The whole search, from the trajectory to the converged orbit, reads as follows.

```julia
using Random
using ReSolver
include("examples/kuramoto_sivashinsky/ks.jl")

Random.seed!(2)
L  = 39
Δt = 0.25

# ---- a chaotic trajectory in the odd subspace, from a random sine series ----
gt = KSGrid(L, 65, 97)
xs = xpoints(gt)
u₀ = sum(0.1 * randn() .* sin.(2π * k .* xs ./ L) for k in 1:6)
U  = integrate(u₀, gt, 0.05, 4000; every=4000, odd=true)       # transient
U  = integrate(U[:, end], gt, 0.05, 16000; every=5, odd=true)   # 800 time units

# ---- initial guess: its best near recurrence with period in [20, 30], without shift ----
e, i, m, _ = recurrence(U, gt, Δt, (20, 30); drift=false)

g = KSGrid(L, 65, 73)
x = initial_orbit(U, g, Δt, i, m, 0.0; drift=false)           # Orbit(a, [log ω])

# ---- the system on the odd subspace: no drift directions, the projection odd!, which also
#      removes the mean ----
F = System(KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!, x;
           linearise!, B=KSPreconditioner(x; kind=:jacobian), project=odd!)

solve!(x, F, NewtonHookstep(maxiter=30, krylov_dim=200, Δ=0.1, Δmax=10, tol=1e-10))
```

The search differs in three ways from the one on $L = 22$. The orbit carries the log-frequency
only, `p = [log ω]`. The keyword `ddi` is not given, so it is empty, `ddi=()`. The keyword
`project=odd!` is passed. The operators, the derivative `dds!` and the preconditioner are those of
the full space, unchanged. The trajectory is integrated with `odd=true`, which removes the cosine
part of the Fourier coefficients at every time step. The odd subspace is unstable in the full space,
and rounding errors would otherwise grow, approximately as $e^{0.09 t}$, until the trajectory leaves
it after a few hundred time units.

## Why no drift speed is needed

### In the full space

Seen in the full space, an odd orbit is a relative periodic orbit with zero drift. The equation is
equivariant under translations, so every translated copy $u(x + \ell, s)$ is an orbit too.
Differentiating with respect to $\ell$ shows that $\partial_x u$ is a null vector of
$\mathcal{J}_u$, the derivative of the residual with respect to the field. It is the neutral
direction of translations. In the full space this direction must be removed by a drift speed with
its phase condition, `ddi=(ddx!,)`, which gives the Newton system

```math
\begin{bmatrix}
\mathcal{J}_u                      & \omega\,\partial_s u & -\partial_x u \\
\langle \partial_s u, \cdot\,\rangle & 0                    & 0             \\
\langle \partial_x u, \cdot\,\rangle & 0                    & 0
\end{bmatrix}
\begin{bmatrix} \delta u \\ \delta\rho \\ \delta c \end{bmatrix}
= -\begin{bmatrix} r \\ 0 \\ 0 \end{bmatrix} .
```

### In the odd subspace

A translated odd field is not odd, except for $\ell = 0$ and $L/2$. The continuous family of
translated copies leaves $V$ at once. Its direction, $\partial_x u$, is an *even* field, orthogonal
to every odd field, so $P\,\partial_x u = 0$. Restricted to $V$, where $\delta u = P\,\delta u$,
the drift column $-\partial_x u$ and the phase condition $\langle \partial_x u, \delta u\rangle$
vanish identically, and the translation is no longer a null vector. The Newton system is that of a
periodic orbit,

```math
\begin{bmatrix}
P \mathcal{J}_u P                  & \omega\,\partial_s u \\
\langle \partial_s u, \cdot\,\rangle & 0
\end{bmatrix}
\begin{bmatrix} \delta u \\ \delta\rho \end{bmatrix}
= -\begin{bmatrix} r \\ 0 \end{bmatrix} , \qquad \delta u \in V ,
```

nonsingular without any condition on translations. This is why `ddi` is left empty. Passing
`ddi=(ddx!,)` together with `project=odd!` would add a column and a row that vanish on $V$. The
Newton system would then be singular, with the drift speed undetermined.

### Time translations stay

The reflection acts on space only and commutes with $\partial_s$. A time-shifted odd orbit
$u(x, s + \sigma)$ is odd, so $\partial_s u$ lies in $V$, and it is still a neutral direction of
the restricted problem. The log-frequency and its phase condition are kept, and `dds!` is passed as
always. In general, `ddi` keeps the derivatives along the translations that commute with the
symmetry. It drops those that anticommute with it, as $\partial_x$ does here,
$S\,\partial_x = -\partial_x S$ (see
[Symmetric subspaces](../theory/root_finding.md#Symmetric-subspaces)).

## Why the projected search works

The search never forms $P \mathcal{J}_u P$. It works because $V$ is invariant. For an odd orbit
$u$ the residual $r$ is odd, and $N$, $L\{u\}$, $L^+\{u\}$ and $\partial_s$ map odd fields to odd
fields. The Krylov space built from the odd right-hand side $-(r, 0)$ by Jacobian actions on odd
fields therefore contains odd fields only. So do the Newton step, the gradient of L-BFGS and the
next orbit. In exact arithmetic the search stays in $V$ with no projection at all. The projection
removes what rounding errors put outside $V$, after every residual, gradient, Jacobian action and
application of the preconditioner (see [Symmetric subspaces](../usage.md#Symmetric-subspaces)).

For this, $P$ must be an *orthogonal projection* in the inner product of the fields, which means
two things.

- $P$ is a projection, $P^2 = P$. An odd field is left unchanged, so projecting twice is the
  same as projecting once.
- $P$ is self-adjoint, $\langle P u, v\rangle = \langle u, P v\rangle$ for all fields. The part
  removed, $u - P u$, the even part of $u$, is orthogonal to every odd field, and $P u$ is the
  odd field closest to $u$ in the norm.

Both hold here because the reflection $u(x) \to u(-x)$ maps the grid onto itself and preserves the
mean over the grid points. They are what the methods need. The projected gradient is the gradient
on $V$. Projecting changes neither the inner products between odd fields, used by the Arnoldi
iteration and by the two-loop recursion, nor the norms of the trust region. The test suite checks
that `odd!` is idempotent and self-adjoint, that the residual of an odd orbit is odd without any
projection, and that a projected search stays odd.

## The orbit

The best near recurrence of the trajectory has relative error 0.013, with $T_0 = 25.50$. From it,
the hookstep with the jacobian preconditioner, Krylov spaces of at most 200 vectors and
$\tau = 10^{-3}$, converges in 4 Newton iterations and 214 Jacobian actions, to
$\lVert r\rVert = 3.3 \times 10^{-13}$. The result is a periodic orbit of period $T = 25.370562$,
odd to the last bit at every iteration.

![Periodic orbit of Kuramoto–Sivashinsky in the odd subspace, L = 39](../assets/kuramoto_sivashinsky/symmetric.png)

*Top, the initial guess and the converged orbit over one period. The dashed line marks $x = L/2$,
about which odd periodic fields are also antisymmetric. Bottom, the residual against the Newton
iterations, all full Newton steps, and the relative residual of the GMRES solution of the Newton
system against the Arnoldi steps, one curve per Newton iteration, with the tolerance $10^{-3}$
dashed.*

## A check of the argument with searches in the full space

The argument above says that the translation is a null direction of the full-space problem, which
the projection removes. To see its effect, the script repeats the search from the same, exactly odd,
initial guess in the full space, in the two ways available there. Both keep the even fields and
remove only the mean, with `project=demean!`, as for the orbit on $L = 22$.

- A periodic orbit in the full space, with `ddi=()` and `project=demean!`. The translation is then
  a null vector of the Newton system that nothing removes. In exact arithmetic the search would be
  identical to the projected one, since every operator preserves odd fields. In floating point it
  shows what rounding errors do along that null direction.
- A relative periodic orbit in the full space, with `ddi=(ddx!,)`, the drift speed as an unknown
  starting from zero, and `project=demean!`. This is the search one would run without exploiting
  the symmetry, as for the orbit on $L = 22$.

| search | `p` | `ddi` | `project` | Newton iterations | Jacobian actions | Arnoldi steps per iteration | $\lVert (I - P)u\rVert / \lVert u\rVert$ at convergence |
|---|---|---|---|---|---|---|---|
| odd subspace | `[log ω]` | `()` | `odd!` | 4 | 214 | 56, 51, 52, 55 | 0 |
| periodic orbit, full space | `[log ω]` | `()` | `demean!` | 4 | 280 | 61, 64, 76, 79 | $1.4 \times 10^{-10}$ |
| relative periodic orbit, full space | `[log ω, c]` | `(ddx!,)` | `demean!` | 4 | 307 | 63, 73, 81, 90 | $1.1 \times 10^{-7}$ |

*The last column is the distance of the converged orbit from the odd subspace.*

All three converge to the same orbit and period. The full-space searches cost 31% and 43% more, and
their inner solves grow from one Newton iteration to the next. Rounding errors put even components
into the Krylov basis, and the full-space Jacobian amplifies them along its nearly singular even
direction, the translation $\partial_x u$. Without a drift speed nothing removes that direction.
The Newton steps contain small arbitrary translations, and the orbit leaves the subspace by
$10^{-7}$ after the first iteration. With a drift speed the phase condition removes it. The drift
speed, though, fluctuates around zero during the iterations, at $-2 \times 10^{-14}$ at
convergence, and translates the orbit by $10^{-7}$ to $10^{-6}$. In both cases GMRES spends Arnoldi
steps on even components that play no role in the solution. The projected search avoids these
effects. Its restricted Jacobian is nonsingular, the even components are removed as they appear,
and the orbit is exactly odd at every iteration. It does not reduce the memory or the cost of an
operator, since the fields keep their full size. A discretisation in sine series would halve both,
at the price of a different field type and different operators.

## References

- Y. Lan and P. Cvitanović, *Unstable recurrent patterns in Kuramoto–Sivashinsky dynamics*,
  Phys. Rev. E 78, 026208 (2008).
- D. Lasagna, *Sensitivity analysis of chaotic systems using unstable periodic orbits*, SIAM J. Appl.
  Dyn. Syst. 17, 547–580 (2018).
