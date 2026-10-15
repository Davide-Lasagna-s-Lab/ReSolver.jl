# Search by optimisation

Far from a solution, a Newton step built on a linear model is unreliable. The first method instead
minimises the size of the residual, a smooth functional of the orbit, with a quasi-Newton method,
[`LBFGS`](@ref). Equations of the [Space-time formulation](@ref) are cited as (F*n*).

## Objective

The search minimises

```math
R(p) = \tfrac12 \lVert r(p) \rVert^2 , \tag{1}
```

with $r$ the residual (F6) and $\lVert\cdot\rVert$ the norm of the fields. The global minimum
$R = 0$ is attained exactly at the orbits. Local minima with $R > 0$ also exist, and a search
that stalls on one needs a different initial guess.

## Gradient

The gradient of $R$ is the orbit $\nabla R$ given by the Riesz representation theorem in the
inner product (F10). For every variation $\delta p = (\delta u, \delta\rho, \delta c)$,

```math
R(p + \delta p) = R(p) + \langle \nabla R, \delta p\rangle + o(\lVert \delta p\rVert) . \tag{2}
```

### Variation of the residual

Perturb every unknown, $u \to u + \delta u$, $\rho \to \rho + \delta\rho$ and
$c_i \to c_i + \delta c_i$. Since $\omega = e^{\rho}$, the frequency becomes
$\omega\, e^{\delta\rho} = \omega\,(1 + \delta\rho) + O(\delta\rho^2)$. The nonlinear operator
becomes $N(u + \delta u) = N(u) + L\,\delta u + O(\lVert\delta u\rVert^2)$, with $L = L\{u\}$ the
linearised operator about $u$. Substituting in the residual (F6) gives

```math
\begin{aligned}
r(p + \delta p)
&= \omega\,(1 + \delta\rho)\,\partial_s (u + \delta u)
 - \sum_i (c_i + \delta c_i)\,\partial_i (u + \delta u)
 - N(u) - L\,\delta u + \dots \\
&= r(p) + \delta r + O(\lVert \delta p\rVert^2) ,
\end{aligned}
```

where $\delta r$ collects the terms of first order, one for each kind of variation,

```math
\delta r = \underbrace{\omega\,\partial_s\delta u - \sum_i c_i\,\partial_i\delta u - L\,\delta u}_{\text{field}}
          + \underbrace{\omega\,\partial_s u\;\delta\rho}_{\text{frequency}}
          - \underbrace{\sum_i \partial_i u\;\delta c_i}_{\text{drift speeds}} . \tag{3}
```

The products of two variations, such as $\omega\,\delta\rho\,\partial_s\delta u$ or
$\delta c_i\,\partial_i\delta u$, are of second order and are dropped.

### Variation of the objective

With this expansion,

```math
R(p + \delta p) = \tfrac12 \langle r + \delta r,\ r + \delta r\rangle + O(\lVert \delta p\rVert^2)
= R(p) + \langle r, \delta r\rangle + O(\lVert \delta p\rVert^2) , \tag{3a}
```

since $\tfrac12\lVert\delta r\rVert^2$ is of second order. To first order $R$ changes by
$\langle r, \delta r\rangle$, in the inner product of the fields. The gradient follows by writing
this change in the form $\langle \nabla R, \delta p\rangle$ of (2).

### Moving the operators onto the residual

The field terms of $\langle r, \delta r\rangle$ contain $\delta u$ under a derivative or under
$L$. The skew-adjointness (F12) of the derivatives and the definition of the adjoint $L^+$ move
each operator from $\delta u$ onto $r$,

```math
\langle r, \omega\,\partial_s \delta u\rangle = -\langle \omega\,\partial_s r, \delta u\rangle , \qquad
\langle r, -c_i\,\partial_i \delta u\rangle = \langle c_i\,\partial_i r, \delta u\rangle , \qquad
\langle r, -L\,\delta u\rangle = -\langle L^+ r, \delta u\rangle . \tag{3b}
```

The parameter terms are already products of a number, $\delta\rho$ or $\delta c_i$, with an
inner product of two known fields. Altogether,

```math
\langle r, \delta r\rangle
= \Big\langle -\omega\,\partial_s r + \sum_i c_i\,\partial_i r - L^{+} r,\ \delta u \Big\rangle
+ \omega\,\langle \partial_s u, r\rangle\,\delta\rho
- \sum_i \langle \partial_i u, r\rangle\,\delta c_i . \tag{4}
```

Matching this with $\langle \nabla R, \delta p\rangle$ in the inner product (F10) gives the
components of $\nabla R = (g_u, g_\rho, g_c)$,

```math
g_u = -\omega\,\partial_s r + \sum_i c_i\,\partial_i r - L^{+} r, \qquad
g_\rho = \omega\,\langle \partial_s u, r\rangle, \qquad
g_{c_i} = -\langle \partial_i u, r\rangle . \tag{5}
```

A comparison of (4) with the Jacobian of root finding shows that the gradient is the adjoint of
that Jacobian applied to the residual with zero phase conditions, $\nabla R = \mathcal{J}^+ (r, 0)$.
The code computes it so, with the same function as `ReSolver.jacobian_adjoint!`, the action of
$\mathcal{J}^+$ that Krylov solvers of least-squares problems, such as LSQR, also need. One
evaluation of $R$ costs one nonlinear operator. One gradient costs one more adjoint operator, about
the current orbit. The gradient is exact for the discretised problem when the discrete derivatives
are exactly skew-adjoint and the adjoint operator is the exact adjoint of the discrete linearised
operator. The examples check both to rounding.

## Limited-memory BFGS

$R$ is minimised by limited-memory BFGS (Nocedal & Wright 2006, ch. 7). The search direction is

```math
d = -H g , \tag{6}
```

with $g = \nabla R$ and $H$ an approximation of the inverse Hessian built from the last
`memory` pairs

```math
s_k = p_{k+1} - p_k, \qquad y_k = g_{k+1} - g_k .
```

$H$ is applied by the two-loop recursion. Without a preconditioner the recursion starts from the
scaled identity $\theta I$, with $\theta = \langle y, s\rangle / \langle y, y\rangle$ of the newest
pair. With a preconditioner it starts from $\theta M^{-1}$, with $M$ the metric that the
preconditioner defines (see [Preconditioning](@ref)). In both cases $\theta$ is clamped to
$[10^{-8}, 10^{8}]$. A pair is kept only if its curvature is positive enough (cautious update),

```math
\langle y_k, s_k\rangle > 10^{-10}\, \lVert s_k\rVert\, \lVert y_k\rVert , \tag{7}
```

so that $H$ stays positive definite. If $d$ is not a descent direction, the steepest-descent
direction $-g$ is used instead. The step $\alpha$ along $d$ is found by backtracking from
$\alpha = 1$, halving it until the Armijo condition

```math
R(p + \alpha d) \le R(p) + 10^{-4}\, \alpha\, \langle g, d\rangle \tag{8}
```

holds. After 25 halvings without success the last trial point is accepted. L-BFGS is robust from a
poor initial guess, but converges only linearly. Its rate is set by the spread of the spectrum of
the Hessian of $R$, which in the Gauss–Newton approximation is $\mathcal{J}_r^+ \mathcal{J}_r$,
with $\mathcal{J}_r$ the Jacobian of the residual (3). For a stiff problem this spread is large,
and L-BFGS needs a preconditioner (see [Preconditioning](@ref)).

## Gradient descent with memory 0

With `LBFGS(memory=0)` the recursion has no pairs, and the method is gradient descent. It is
steepest descent in the metric of the preconditioner, with the step scaled by the newest change of
the iterate and of the gradient, and the same line search as L-BFGS.

### Direction

The two-loop recursion over the newest $m$ pairs applies to $g$ the inverse Hessian built from the
initial matrix

```math
H_0 = \theta\, M^{-1} , \qquad \theta = \frac{\langle y, s\rangle}{\langle y, M^{-1} y\rangle} , \tag{9}
```

with $(s, y)$ the newest pair that passed the test (7), and $M = I$ without preconditioner. The
first loop runs over the pairs from the newest to the oldest, the second from the oldest to the
newest. With $m = 0$ both are empty, and the direction is

```math
d = -\theta\, M^{-1} g . \tag{10}
```

The newest pair is kept for $\theta$ even with no memory. The memory holds $\max(m, 1)$ pairs, and
the recursion uses the newest $m$ of them. At the first iteration, with no pair yet,
$d = -M^{-1} g$, as for every memory.

### Step length

The line search (8) starts from $\alpha = 1$. The step tried first is therefore $\theta M^{-1} g$,
and the step taken is $\alpha\theta M^{-1} g$. Here $\theta$ is the second step length of
Barzilai & Borwein (1988), in the metric $M$. It is the inverse of the curvature of $R$ along the
last step, measured by the change of the gradient. It gives the trial step the right size, so that
the line search seldom backtracks. Without it the unit step along $-M^{-1}g$ would be too long,
which wastes residuals in the line search, or too short, which wastes iterations. With (9) and a
monotone line search, memory 0 is the spectral gradient method of Raydan (1997), whose line search
is not monotone.

### One pair

One pair is not gradient descent. With $m = 1$ the recursion makes one BFGS update of
$\theta M^{-1}$ with the newest pair. This is the memoryless BFGS method, whose directions with an
exact line search are those of the nonlinear conjugate gradient method of Hestenes and Stiefel
(Shanno 1978; Nocedal & Wright 2006, §7.2). It carries the information of one previous step, as
conjugate gradients do.

### Use

Gradient descent and L-BFGS cost the same per iteration, apart from the recursion. They are
compared with the same metric, line search and scaling, so that the comparison measures the value
of the curvature pairs alone. In tests on Kolmogorov flow, gradient descent was as good as L-BFGS
as a first stage before the hookstep, down to $\lVert r\rVert = 10^{-3}$. It needed up to 2.6 times
as many iterations, but the search took the same time. For long runs the pairs pay. On an
under-resolved grid gradient descent ended twenty times higher than L-BFGS with 10 pairs after
300 s, and on a resolved grid a factor of two higher.

## Cost

Each iteration costs one gradient, which is one nonlinear and one adjoint operator, and one
nonlinear operator for each trial point of the line search, at least one. With a preconditioner,
the recursion applies $M^{-1} = \mathcal{B}^{-1}\mathcal{B}^{-+}$ twice, to the newest $y$ for the
scaling $\theta$ and to the vector left by the first loop. This makes four applications of
$\mathcal{B}^{-1}$ or $\mathcal{B}^{-+}$ per iteration, and two at the first iteration. The
recursion over $m$ pairs adds about $4m$ inner products and updates of the orbit, which match the
cost of the operators when $m$ is large. For Kolmogorov flow on 65 instants an iteration takes
1.6 ms with $m = 0$, 2.7 ms with $m = 10$ and 6.9 ms with $m = 50$, on an NVIDIA H200.

## References

- Barzilai, J. & Borwein, J. M. 1988 Two-point step size gradient methods. *IMA J. Numer. Anal.* 8,
  141–148.
- Nocedal, J. & Wright, S. J. 2006 *Numerical Optimization*, 2nd edn. Springer.
- Raydan, M. 1997 The Barzilai and Borwein gradient method for the large scale unconstrained
  minimization problem. *SIAM J. Optim.* 7, 26–33.
- Shanno, D. F. 1978 Conjugate gradient methods with inexact searches. *Math. Oper. Res.* 3,
  244–256.
