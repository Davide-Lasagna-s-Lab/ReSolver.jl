# Search by root finding

Close to a solution, Newton's method converges far faster than any minimisation of the residual.
The second method, [`NewtonHookstep`](@ref), solves $r = 0$ with a Newton–Krylov iteration
globalised by the hookstep trust region of Viswanath (2007). Equations of the
[Space-time formulation](@ref) are cited as (F*n*).

## Newton step

Newton's method replaces the nonlinear equation $r(p) = 0$ by a sequence of linear ones. At the
current orbit $p$, the residual at a nearby orbit $p + \delta p$, with
$\delta p = (\delta u, \delta\rho, \delta c)$, is expanded to first order,

```math
r(p + \delta p) = r(p) + \mathcal{J}_r\,\delta p + O(\lVert \delta p\rVert^2) , \tag{1a}
```

where $\mathcal{J}_r$ is the Jacobian of the residual. It is the linear operator that maps a
variation of the orbit to the first-order variation $\delta r$ of the residual. The Newton step is
the $\delta p$ that cancels the right-hand side truncated to first order,

```math
\mathcal{J}_r\,\delta p = -r(p) , \tag{1b}
```

and the next orbit is $p + \delta p$. If $r$ were linear, one step would land on the solution.
Close to a solution the neglected term is quadratic in $\delta p$, and the error is squared at
every step.

### The Jacobian

The first-order variation of the residual was computed in [Search by optimisation](@ref), in its
equation (3). It follows from perturbing $u$, $\rho = \log\omega$ and the $c_i$ in (F6),
expanding $\omega\, e^{\delta\rho} \approx \omega(1 + \delta\rho)$ and
$N(u + \delta u) \approx N(u) + L\,\delta u$, and keeping the terms linear in the variations.
Grouped by unknown,

```math
\mathcal{J}_r\,\delta p
= \underbrace{\mathcal{J}_u\,\delta u}_{\text{field}}
+ \underbrace{\omega\,\partial_s u\;\delta\rho}_{\text{frequency}}
- \underbrace{\sum_{i=1}^{m} \partial_i u\;\delta c_i}_{\text{drift speeds}} ,
\qquad
\mathcal{J}_u = \omega\,\partial_s - \sum_i c_i\,\partial_i - L , \tag{1}
```

with $\mathcal{J}_u$ the derivative of the residual with respect to the field, the linearised
space-time operator, $L = L\{u\}$ the linearised operator about the current field, and $m$ the
number of drift directions. Each unknown contributes one block. The field contributes through the
operator $\mathcal{J}_u$, which acts on the whole space-time field $\delta u$. The log-frequency
contributes through the single field $\omega\,\partial_s u$, the derivative of the residual with
respect to $\rho$, and each drift speed through the single field $-\partial_i u$. In matrix form,
with the field and the parameters stacked,

```math
\mathcal{J}_r = \begin{bmatrix} \mathcal{J}_u & \omega\,\partial_s u & -\partial_1 u & \cdots & -\partial_m u
\end{bmatrix} ,
```

one block row with as many rows as field unknowns, and $1 + m$ more columns. Computing
$\mathcal{J}_r\,\delta p$ costs one linearised operator $L\,\delta u$, the derivatives
$\partial_s\delta u$ and $\partial_i\delta u$, and the derivatives $\partial_s u$ and
$\partial_i u$ of the current orbit, which do not change during a Newton iteration.

## Two defects and the phase conditions

The Newton equation (1b), $\mathcal{J}_r\,\delta p = -r$, cannot be solved as it stands, for two
reasons.

1. It has too many unknowns. It has as many equations as field unknowns $\delta u$, but $1 + m$
   more unknowns, $\delta\rho$ and the $\delta c_i$.
2. Its operator is singular. By the symmetries, a solution $u$ shifted in rescaled time,
   $u(x, s + \sigma)$, or along a drift direction, $u(x + \eta\, e_i, s)$, is again a solution
   with the same $\omega$ and $c$. Differentiating $r = 0$ with respect to $\sigma$ and
   $\eta$ at zero gives

```math
\mathcal{J}_u\,\partial_s u = 0, \qquad \mathcal{J}_u\,\partial_i u = 0 . \tag{2}
```

   At a solution $\mathcal{J}_u$ is therefore singular, with null space spanned by the *neutral
   directions* $\partial_s u$ and $\partial_i u$. Close to a solution $\mathcal{J}_u$ is nearly
   singular, and a step can wander along these directions without reducing the residual.

Both defects are removed by requiring the step to be orthogonal to the neutral directions. These
are the *phase conditions*

```math
\langle \partial_s u, \delta u\rangle = 0, \qquad \langle \partial_i u, \delta u\rangle = 0,
\quad i = 1, \dots, m . \tag{3}
```

They are $1 + m$ equations, as many as the missing ones. They fix the origin of $s$ and the
position of the orbit along the drift directions, which the residual cannot determine.

## Block system

Together, the Newton equation (1b) and the phase conditions form the Newton system, whose first
block row is $\mathcal{J}_r$,

```math
\underbrace{\begin{bmatrix}
\mathcal{J}_u                      & \omega\,\partial_s u & -\partial_1 u & \cdots & -\partial_m u \\
\langle \partial_s u, \cdot\,\rangle & 0                    & 0             & \cdots & 0             \\
\langle \partial_1 u, \cdot\,\rangle & 0                    & 0             & \cdots & 0             \\
\vdots                             & \vdots               & \vdots        & \ddots & \vdots        \\
\langle \partial_m u, \cdot\,\rangle & 0                    & 0             & \cdots & 0
\end{bmatrix}}_{\mathcal{J}}
\begin{bmatrix}
\delta u \\ \delta\rho \\ \delta c_1 \\ \vdots \\ \delta c_m
\end{bmatrix}
=
-\begin{bmatrix}
r \\ 0 \\ 0 \\ \vdots \\ 0
\end{bmatrix} . \tag{4}
```

The first block row is the linearised residual, the others the phase conditions. The right-hand side
is minus the augmented residual $(r, 0, \dots, 0)$ returned by `residual!`, whose parameter
components, the phase conditions, vanish at the current orbit. The operator $\mathcal{J}$ maps an
orbit to an orbit, the space of the unknowns, so the system is square. `jacobian!` computes its
action on $\delta p$. The singular $\mathcal{J}_u$ is bordered by its own null vectors. At an
isolated relative periodic orbit, whose only neutral directions are those of the symmetries,
$\mathcal{J}$ is nonsingular, and Newton's method converges quadratically.

### Which drift directions

Every drift direction adds a row and a column to $\mathcal{J}$, and must be a genuine neutral
direction of the problem. If the fields are restricted to a symmetric subspace that a translation
would leave, as the odd solutions $u(-x) = -u(x)$ of the Kuramoto–Sivashinsky equation are, then
$\partial_i u$ has no component in that subspace. Its row and column of $\mathcal{J}$ vanish and
the system is singular. Such directions are left out of `ddi`.

## Symmetric subspaces

Let $S$ be a discrete symmetry of the problem, such as a reflection or a half-period shift. It is
an isometry with $S^2 = I$, acting on space only, under which the equation is equivariant,
$S\,N(u) = N(S u)$. The orbits whose field is fixed by $S$ form a subspace $V$, and
$P = (I + S)/2$ is the orthogonal projection on its fields. If the current orbit lies in $V$,
then $L\{u\}$ commutes with $S$, and $\mathcal{J}_u$ maps the fields of $V$ into themselves.

### Which generators survive

The columns and phase conditions of (4) involve the generators of the continuous symmetries,
$\partial_s u$ and $\partial_i u$. Whether these lie in $V$ decides the shape of the restricted
system.

- $S$ acts on space only, so it commutes with $\partial_s$ and $\partial_s u$ lies in $V$. The
  time translations of a symmetric orbit are symmetric orbits, and $\partial_s u$ is still a
  neutral direction of the restricted problem. The log-frequency and its phase condition stay, and
  `dds!` is always needed.
- A translation that commutes with $S$, $S\,\partial_i = \partial_i S$, such as a streamwise
  translation under a spanwise reflection, has $\partial_i u$ in $V$. The translated orbits stay
  in $V$, and the drift speed with its phase condition stays. Then $\partial_i$ belongs in `ddi`.
- A translation that anticommutes with $S$, $S\,\partial_i = -\partial_i S$, such as the
  translation along the axis of the reflection itself, has $\partial_i u$ in the orthogonal
  complement of $V$, so $P\,\partial_i u = 0$. The translated orbits leave $V$, and the symmetric
  orbits cannot drift in that direction. Then $\partial_i$ must be left out of `ddi`.

### The restricted system

For the odd Kuramoto–Sivashinsky orbits, $S u(x) = -u(-x)$ anticommutes with $\partial_x$. In the
full space the Newton system of a relative periodic orbit is

```math
\begin{bmatrix}
\mathcal{J}_u                      & \omega\,\partial_s u & -\partial_x u \\
\langle \partial_s u, \cdot\,\rangle & 0                    & 0             \\
\langle \partial_x u, \cdot\,\rangle & 0                    & 0
\end{bmatrix}
\begin{bmatrix} \delta u \\ \delta\rho \\ \delta c \end{bmatrix}
= -\begin{bmatrix} r \\ 0 \\ 0 \end{bmatrix} ,
```

of size $N_u + 2$. Restricted to the odd fields, with $\delta u = P\,\delta u$, the column
$-\partial_x u$ and the row $\langle \partial_x u, \cdot\,\rangle$ vanish, since $\partial_x u$ is
even. The system becomes that of a periodic orbit,

```math
\begin{bmatrix}
P \mathcal{J}_u P                  & \omega\,\partial_s u \\
\langle \partial_s u, \cdot\,\rangle & 0
\end{bmatrix}
\begin{bmatrix} \delta u \\ \delta\rho \end{bmatrix}
= -\begin{bmatrix} r \\ 0 \end{bmatrix} , \qquad \delta u \in V ,
```

of size $\dim V + 1$, about half. It is nonsingular although $\mathcal{J}_u$ is singular on the
full space. Its null vector $\partial_x u$ is not in $V$, so the translation needs neither a
drift speed nor a phase condition. In the code this is `System(...; ddi=(), project=odd!)` with an
orbit carrying the log-frequency only, against `ddi=(ddx!,)` and a drift speed in the full space.

### How the code restricts the system

The right-hand side $-(r, 0)$ lies in $V$, and so does the image of $V$ under every block. The
Krylov spaces built from it, and the Newton step, therefore never leave $V$, and GMRES solves the
restricted system without ever forming $P \mathcal{J}_u P$. In floating point, rounding errors put
components outside $V$ into the Krylov basis. The nearly singular directions of $\mathcal{J}_u$
outside $V$, such as the translation above, amplify them. The orbit then drifts out of the
subspace, and the Krylov spaces grow to resolve directions that play no role. The orthogonal
projection on $V$, applied after every operator (the keyword `project` of [`System`](@ref)),
removes these components as they appear.

## Krylov space

### Matrix-free solution

The operator $\mathcal{J}$ acts on orbits, whose dimension is the number of field unknowns plus
$1 + m$. For a PDE this is far too large for $\mathcal{J}$ to be formed or factorised. What is
available is its action on an orbit, $\delta p \mapsto \mathcal{J}\,\delta p$, at the cost of one
linearised operator and a few derivatives. The Newton system is therefore solved in a Krylov space,
built from these actions alone (Saad & Schultz 1986; Knoll & Keyes 2004). The linearised operator
itself can be replaced by a finite difference of $N$, as in Jacobian-free Newton–Krylov methods
(Knoll & Keyes 2004). The page [Usage](@ref) describes this option of [`System`](@ref).

### Arnoldi iteration

The Arnoldi iteration starts from the right-hand side $b = -(r, 0, \dots, 0)$, of norm
$\beta = \lVert b\rVert$. It builds an orthonormal basis $Q_n = [q_1, \dots, q_n]$ of the Krylov
space

```math
\mathcal{K}_n = \mathrm{span}\{\, b,\ \mathcal{J} b,\ \dots,\ \mathcal{J}^{\,n-1} b \,\} ,
\qquad q_1 = b/\beta , \tag{5}
```

one vector per Jacobian action. At step $n$ it applies $\mathcal{J}$ to the last vector,
$v = \mathcal{J} q_n$, removes its components $h_{jn} = \langle q_j, v\rangle$ along the basis,
and normalises what remains, $q_{n+1} = v / h_{n+1,n}$ with $h_{n+1,n}$ its norm. The
coefficients $h_{jn}$ fill an $(n+1) \times n$ upper Hessenberg matrix $H$, and the construction
is summarised by the Arnoldi relation

```math
\mathcal{J}\, Q_n = Q_{n+1}\, H . \tag{6}
```

The orthogonalisation is modified Gram–Schmidt, done twice. A single pass loses orthogonality once
the new vector lies almost in the span of the basis, as it does when the Krylov space captures the
dominant action of $\mathcal{J}$. A second pass restores it to rounding ("twice is enough",
Giraud, Langou & Rozložník 2005). If $h_{n+1,n}$ vanishes the Krylov space is invariant
(breakdown) and stops growing. The least-squares solution below is then exact.

### Reduction to a small problem

Look for the step in the Krylov space, $\delta p = Q_n y$ with coordinates $y \in \mathbb{R}^n$,
where $Q_n y = \sum_{j=1}^{n} y_j\, q_j$. Both terms of the residual of the linear model are
combinations of the first $n + 1$ Arnoldi vectors. The right-hand side is the first of them,
$b = \beta\, q_1 = Q_{n+1}\, g$, with $g = (\beta, 0, \dots, 0) \in \mathbb{R}^{n+1}$, and by the
Arnoldi relation $\mathcal{J}\,\delta p = \mathcal{J}\, Q_n y = Q_{n+1}\, H y$. Hence

```math
b - \mathcal{J}\,\delta p = Q_{n+1}\,(g - H y) . \tag{7}
```

The norm of an orbit written on an orthonormal basis is the Euclidean norm of its coordinates. For
any $z \in \mathbb{R}^{n+1}$,

```math
\lVert Q_{n+1} z\rVert^2
= \sum_{j,k} z_j\, z_k\, \langle q_j, q_k\rangle
= \sum_j z_j^2
= \lVert z\rVert^2 , \tag{8}
```

since $\langle q_j, q_k\rangle = \delta_{jk}$. With $z = g - H y$, and with $z = y$ on the first
$n$ vectors,

```math
\lVert b - \mathcal{J}\,\delta p \rVert = \lVert g - H y \rVert ,
\qquad \lVert \delta p\rVert = \lVert y\rVert . \tag{9}
```

Solving the Newton system in $\mathcal{K}_n$ is the $(n+1) \times n$ least-squares problem
$\min_y \lVert g - H y\rVert$, whose solution is the GMRES step. The Krylov dimension is tens to a
few hundred, orders of magnitude below the number of unknowns of a PDE.

## Hookstep

### Why a trust region

The Newton step solves the linear model exactly, but the model is accurate only near the current
orbit. Far from a solution, and with $\mathcal{J}$ nearly singular along directions that the phase
conditions do not remove, the GMRES step can be very long and move to a point where the residual is
larger. The step is therefore restricted to a trust region $\lVert \delta p\rVert \le \Delta$,
which by (9) reads, in the Krylov space,

```math
\min_y \lVert g - H y\rVert \quad\text{subject to}\quad \lVert y\rVert \le \Delta . \tag{10}
```

### Solution of the constrained problem

The minimiser satisfies $(H^\top H + \mu I)\, y = H^\top g$ for a Lagrange multiplier
$\mu \ge 0$, with $\mu = 0$ if the constraint is inactive and $\lVert y\rVert = \Delta$
otherwise. With the singular value decomposition $H = U \Sigma V^\top$, singular values
$\sigma_1, \dots, \sigma_n$, and $\hat g = U^\top g \in \mathbb{R}^n$, the problem decouples along
the singular directions,

```math
y(\mu) = V\,\hat y(\mu), \qquad
\hat y_k(\mu) = \frac{\sigma_k\, \hat g_k}{\sigma_k^2 + \mu}, \qquad
\lVert y(\mu)\rVert^2 = \sum_{k=1}^{n} \frac{\sigma_k^2\, \hat g_k^2}{(\sigma_k^2 + \mu)^2} . \tag{11}
```

If the GMRES step $y(0)$, with $\hat y_k = \hat g_k / \sigma_k$, lies inside the region, it is
the step. Otherwise $\mu$ is the root of $\lVert y(\mu)\rVert = \Delta$. The norm decreases
monotonically in $\mu$ and is bounded by
$\lVert y(\mu)\rVert \le \max_k \sigma_k\, \lVert \hat g\rVert / \mu$. The root is therefore
unique and lies in $[0,\ \max_k \sigma_k\, \lVert \hat g\rVert / \Delta]$, where it is found by
bisection.

### Why the step is called a hook

As $\mu$ grows from zero, $y(\mu)$ moves along a curved path from the Newton (GMRES) step at
$\mu = 0$ towards the steepest-descent direction of $\lVert g - H y\rVert^2$, with
$y \approx H^\top g / \mu$ for large $\mu$. The multiplier damps the components along small
singular values, $\hat y_k \approx \sigma_k \hat g_k / \mu$ when $\sigma_k^2 \ll \mu$. These are
the nearly singular directions of $\mathcal{J}$, where the Newton step is long and least reliable.
A short trust region keeps the well-determined part of the step and discards the rest.

## Trust-region update

The model predicts the reduction $\tfrac12\big(\beta^2 - \lVert g - Hy\rVert^2\big)$ of
$R = \tfrac12\lVert r\rVert^2$. The trial point $p + \delta p$ gives the actual one, at the cost
of one nonlinear operator. Their ratio $\gamma$ drives the radius. The hookstep prints it as
`ρ`, and a [`Trace`](@ref) stores it as `ratio`. The threshold $\eta$ is the option `eta`.

| ratio | meaning | action |
|---|---|---|
| $\gamma < 1/4$, or not a number | poor model | $\Delta \leftarrow \Delta/4$ |
| $\gamma > 3/4$, step on the boundary | good model, step limited by $\Delta$ | $\Delta \leftarrow \min(2\Delta, \Delta_{\max})$ |
| $\gamma > \eta$ | sufficient reduction | accept the step |

A rejected step is recomputed with the smaller radius in the same Krylov space, without new
Jacobian actions. If the radius falls below $\epsilon\,\lVert p\rVert$, with $\epsilon$ the machine
precision, the search stops. Close to a solution the steps fall inside the region and the hookstep
is the Newton step.

## Inexact Newton

The Krylov space grows until the GMRES step, the unconstrained least-squares solution
$y_0 = \arg\min_y \lVert g - Hy\rVert$, solves the Newton system to a fraction of the right-hand
side,

```math
\lVert g - H y_0\rVert \le \tau\, \beta , \tag{12}
```

with $\tau$ the option `krylov_tol`, or until it reaches `krylov_dim` vectors. The hookstep (10)
is then computed in that space. The criterion is on the unconstrained problem, not on the hookstep,
on purpose. When the trust region is active the hookstep cannot reduce the residual of the model
below the level reached by a step of length $\Delta$. If the Newton step is ten times longer than
the radius, $\lVert g - Hy\rVert$ stalls close to $\beta$ however large the Krylov space. A
criterion on it would build $\texttt{krylov\_dim}$ vectors at every step limited by the trust
region, with no benefit. With (12), the space is the one in which the Newton step is known to
accuracy $\tau$, and the hookstep chooses the best part of that step that the trust region allows.
For the first Kuramoto–Sivashinsky example this halves the cost of the search.

This is an *inexact Newton* method (Dembo, Eisenstat & Steihaug 1982). Each step reduces the
residual of the linear model by the factor $\tau$ only, and close to the solution the Newton
iteration converges linearly with a rate of about $\tau$, rather than quadratically. Solving the
Newton system exactly, $\tau \to 0$, restores quadratic convergence but spends Arnoldi steps on
accuracy that the next Newton iteration does not need. A large $\tau$ makes each step cheap but
needs more Newton iterations. The best $\tau$ balances the two. Adaptive choices that tighten
$\tau$ as the residual decreases are possible (Eisenstat & Walker 1996). The examples measure the
cost of convergence against $\tau$.

## Cost

Each Newton iteration costs one linearisation and one nonlinear operator for the right-hand side.
It then costs one linearised operator per Krylov vector, one nonlinear operator per trial step, and
the orthogonalisation of each new vector against the basis, $O(n N)$ operations for $n$ vectors of
$N$ unknowns. The cost of a search is dominated by the number of Jacobian actions. This number
depends on how fast the Krylov solver converges, which depends on the spectrum of $\mathcal{J}$ and
so on [Preconditioning](@ref).
