# Search by optimisation

Far from a solution, a Newton step built on a linear model is unreliable. The first method instead
minimises the size of the residual, a smooth functional of the orbit, with a quasi-Newton method:
[`LBFGS`](@ref). Equations of the [Space-time formulation](@ref) are cited as (F*n*).

## Objective

The search minimises

```math
R(p) = \tfrac12 \lVert r(p) \rVert^2 , \tag{1}
```

with ``r`` the residual (F6) and ``\lVert\cdot\rVert`` the norm of the fields. The global minimum
``R = 0`` is attained exactly at the orbits; local minima with ``R > 0`` also exist, and a search
that stalls on one needs a different initial guess.

## Gradient

The gradient of ``R`` is the orbit ``\nabla R`` given by the Riesz representation theorem in the
inner product (F10): for every variation ``\delta p = (\delta u, \delta\rho, \delta c)``,

```math
R(p + \delta p) = R(p) + \langle \nabla R, \delta p\rangle + o(\lVert \delta p\rVert) . \tag{2}
```

**Variation of the residual.** Perturb every unknown, ``u \to u + \delta u``,
``\rho \to \rho + \delta\rho`` and ``c_i \to c_i + \delta c_i``. Since ``\omega = e^{\rho}``, the
frequency becomes ``\omega\, e^{\delta\rho} = \omega\,(1 + \delta\rho) + O(\delta\rho^2)``, and the
nonlinear operator ``N(u + \delta u) = N(u) + L\,\delta u + O(\lVert\delta u\rVert^2)``, with
``L = L\{u\}`` the linearised operator about ``u``. Substituting in the residual (F6),

```math
\begin{aligned}
r(p + \delta p)
&= \omega\,(1 + \delta\rho)\,\partial_s (u + \delta u)
 - \sum_i (c_i + \delta c_i)\,\partial_i (u + \delta u)
 - N(u) - L\,\delta u + \dots \\
&= r(p) + \delta r + O(\lVert \delta p\rVert^2) ,
\end{aligned}
```

where ``\delta r`` collects the terms of first order, one for each kind of variation:

```math
\delta r = \underbrace{\omega\,\partial_s\delta u - \sum_i c_i\,\partial_i\delta u - L\,\delta u}_{\text{field}}
          + \underbrace{\omega\,\partial_s u\;\delta\rho}_{\text{frequency}}
          - \underbrace{\sum_i \partial_i u\;\delta c_i}_{\text{drift speeds}} . \tag{3}
```

The products of two variations, such as ``\omega\,\delta\rho\,\partial_s\delta u`` or
``\delta c_i\,\partial_i\delta u``, are of second order and are dropped.

**Variation of the objective.** With this expansion,

```math
R(p + \delta p) = \tfrac12 \langle r + \delta r,\ r + \delta r\rangle + O(\lVert \delta p\rVert^2)
= R(p) + \langle r, \delta r\rangle + O(\lVert \delta p\rVert^2) , \tag{3a}
```

since ``\tfrac12\lVert\delta r\rVert^2`` is of second order. So ``R`` changes, to first order, by
``\langle r, \delta r\rangle``, in the inner product of the fields, and the gradient follows by
writing this change in the form ``\langle \nabla R, \delta p\rangle`` of (2).

**Moving the operators onto ``r``.** The field terms of ``\langle r, \delta r\rangle`` contain
``\delta u`` under a derivative or under ``L``. The skew-adjointness (F12) of the derivatives and the
definition of the adjoint ``L^+`` move each operator from ``\delta u`` onto ``r``:

```math
\langle r, \omega\,\partial_s \delta u\rangle = -\langle \omega\,\partial_s r, \delta u\rangle , \qquad
\langle r, -c_i\,\partial_i \delta u\rangle = \langle c_i\,\partial_i r, \delta u\rangle , \qquad
\langle r, -L\,\delta u\rangle = -\langle L^+ r, \delta u\rangle . \tag{3b}
```

The parameter terms are already products of a number, ``\delta\rho`` or ``\delta c_i``, with an
inner product of two known fields. Altogether,

```math
\langle r, \delta r\rangle
= \Big\langle -\omega\,\partial_s r + \sum_i c_i\,\partial_i r - L^{+} r,\ \delta u \Big\rangle
+ \omega\,\langle \partial_s u, r\rangle\,\delta\rho
- \sum_i \langle \partial_i u, r\rangle\,\delta c_i . \tag{4}
```

Matching this with ``\langle \nabla R, \delta p\rangle`` in the inner product (F10) gives the
components of ``\nabla R = (g_u, g_\rho, g_c)``:

```math
g_u = -\omega\,\partial_s r + \sum_i c_i\,\partial_i r - L^{+} r, \qquad
g_\rho = \omega\,\langle \partial_s u, r\rangle, \qquad
g_{c_i} = -\langle \partial_i u, r\rangle . \tag{5}
```

Comparing (4) with the Jacobian of root finding, the gradient is the adjoint of that Jacobian
applied to the residual with zero phase conditions, ``\nabla R = \mathcal{J}^+ (r, 0)``, and the code
computes it so, with `ReSolver.jacobian_adjoint!`, the action of ``\mathcal{J}^+`` that Krylov
solvers of least-squares problems, such as LSQR, also need. One evaluation of ``R`` costs one
nonlinear operator; one gradient costs in addition one adjoint operator, about the current orbit. The gradient is exact for the discretised problem when the
discrete derivatives are exactly skew-adjoint and the adjoint operator is the exact adjoint of the
discrete linearised operator; the examples check both to rounding.

## Limited-memory BFGS

``R`` is minimised by limited-memory BFGS (Nocedal & Wright 2006, ch. 7). The search direction is

```math
d = -H g , \tag{6}
```

with ``g = \nabla R`` and ``H`` an approximation of the inverse Hessian built from the last
`memory` pairs

```math
s_k = p_{k+1} - p_k, \qquad y_k = g_{k+1} - g_k ,
```

applied by the two-loop recursion, starting from the scaled identity ``\theta I``, with
``\theta = \langle y, s\rangle / \langle y, y\rangle`` of the newest pair. A pair is kept only if its
curvature is positive enough (cautious update),

```math
\langle y_k, s_k\rangle > 10^{-10}\, \lVert s_k\rVert\, \lVert y_k\rVert , \tag{7}
```

so that ``H`` stays positive definite; if ``d`` is not a descent direction, steepest descent is used
instead. The step ``\alpha`` along ``d`` is found by backtracking from ``\alpha = 1``, halving it until
the Armijo condition

```math
R(p + \alpha d) \le R(p) + 10^{-4}\, \alpha\, \langle g, d\rangle \tag{8}
```

holds. L-BFGS is robust from a poor initial guess, but converges only linearly, at a rate set by the
spread of the spectrum of the Hessian of ``R``, which in the Gauss–Newton approximation is
``\mathcal{J}_r^+ \mathcal{J}_r``, with ``\mathcal{J}_r`` the Jacobian of the residual (3). For a
stiff problem this spread is large, and L-BFGS needs a preconditioner (see
[Preconditioning](@ref)).

## Cost

Each iteration costs one gradient, that is one nonlinear and one adjoint operator, plus one
nonlinear operator per rejected line-search trial; with a preconditioner, two applications of
``B^{-1}`` or ``B^{-+}`` for the initial matrix of the recursion.
