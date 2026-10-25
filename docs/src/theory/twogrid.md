# Two-grid preconditioning

The diagonal preconditioners of the examples invert the linear part of the space-time operator,
the time derivative and the dissipation, exactly. They leave the part that the nonlinear terms
couple: large scales at low temporal frequencies. That part contains the slow directions of the
Jacobian, the rungs of the Floquet ladders near the origin, whose number grows with the period
$T$. GMRES spends a number of iterations proportional to $T$ on them before the residual falls,
and the cost of a Newton step grows like $T^2$ in memory and $T^3$ in orthogonalisation.

[`TwoGrid`](@ref) removes that part with a coarse-grid correction, in the sense of multigrid
methods (Hackbusch 1985; Briggs, Henson & McCormick 2000; Trottenberg, Oosterlee & Schüller 2001).

## The cycle

The preconditioner $B$ given by the user becomes the *smoother*: one smoothing step is
$z \leftarrow z + \theta B^{-1}(b - \mathcal{J} z)$, a preconditioned Richardson iteration. It
damps the error at small scales and high frequencies, where $B \approx \mathcal{J}$, and leaves the
rest. The *coarse-grid correction* $z \leftarrow z + P K^{-1} R\,(b - \mathcal{J} z)$ solves for
that rest in a coarse space, with prolongation $P$, restriction $R = P^{\mathsf T}$ and coarse
operator $K \approx R\,\mathcal{J} P$. The default cycle is two smoothing steps, the coarse-grid
correction and two smoothing steps, which costs four Jacobian actions on top of the one of GMRES.
Smoothing on both sides of the coarse-grid correction matters: one-sided cycles stall.
The cycle is a fixed linear operator, so plain right-preconditioned GMRES applies.

## The coarse space

At each time slice the coarse space keeps the first `r` spatial coordinates of the field, given by
the user as an isometry `coords`/`field!` ordered from large to small scales: spectral truncation in
space. With `neutral=true` it also keeps, at each slice, the profiles of the neutral directions,
$\partial_s a$ and the drift derivatives. In time it keeps every slice (`tc=1`) or periodic cubic
B-splines on a grid `tc` times coarser.

The coarse operator is the Galerkin projection of the Jacobian. Its spatial blocks are obtained by
probing: one application of the linearised operator to the field whose coordinates are $e_\alpha$
at every slice gives column $\alpha$ of the blocks of all slices at once. Building $K$ therefore
costs `r` applications per Newton iteration, independently of $T$. The time derivative is exact on
the B-splines; with `tc=1` it is replaced by centred differences of order `2h`. Either way $K$ is
cyclic block-banded, and its direct factorisation, by sparse LU or, with `using BorderedMatrices`,
by banded block elimination (`solver=:banded`), costs time and memory proportional to $T$.

## Use

For the Kuramoto–Sivashinsky example, in the odd subspace,

```julia
coords, field! = odd_sine_coordinates(g)
B = TwoGrid(KSPreconditioner(x; kind=:jacobian), coords, field!, 16; tc=2, solver=:banded)  # V(2,2)
F = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, B, project=odd!)
solve!(x, F, NewtonHookstep(krylov_dim=60))
```

The preconditioner applies the Jacobian, so it serves the hookstep only. The study in
`examples/kuramoto_sivashinsky/twogrid` measures its cost against the diagonal preconditioner
alone for periods up to 64 times that of the shortest orbit, and the effect of `r` and `tc`.

## References

- W. Hackbusch, *Multi-Grid Methods and Applications*, Springer (1985).
- W. L. Briggs, V. E. Henson and S. F. McCormick, *A Multigrid Tutorial*, 2nd ed., SIAM (2000).
- U. Trottenberg, C. W. Oosterlee and A. Schüller, *Multigrid*, Academic Press (2001).
- T. A. Zang, Y. S. Wong and M. Y. Hussaini, Spectral multigrid methods for elliptic equations,
  *J. Comput. Phys.* 48, 485–501 (1982).
