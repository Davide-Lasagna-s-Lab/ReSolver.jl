# PeriodicBSplines.jl

Periodic cubic B-splines on a uniform grid of the circle ``[0, 2\pi)``: the basis functions, their
values and derivatives on a finer uniform grid, their exact Gram matrices, interpolation, and the
Fourier symbols that say how well they resolve and differentiate a harmonic.

The package is small on purpose. It exists so that one numerical ingredient of
[ReSolver](https://github.com/Davide-Lasagna-s-Lab/ReSolver.jl) is defined, tested and documented
in one place: the coarse time grid of the two-grid preconditioner `TwoGrid`, where a space-time
field that is periodic in time is represented, on a grid ``t_c`` times coarser than its own, by
periodic cubic B-splines. The [Theory](@ref) page derives everything the package computes; the
[API](@ref) page lists the functions.

![The cardinal cubic B-spline and its derivatives, and the eight periodic splines of a grid with
Nc = 8, which sum to one.](assets/basis.png)

## Quick start

```julia
using PeriodicBSplines

Nc, Ns = 16, 48                         # 16 splines, values at 48 equispaced points
Φ, Φ′ = collocation(Ns, Nc)             # sparse 48 × 16: φ_m(s_l) and φ_m′(s_l)
M, D  = gram(Nc), derivative_gram(Nc)   # exact ∫ φ_m φ_m′ ds and ∫ φ_m φ_m′′ ds

f(s) = exp(sin(s))
c = interpolate(f.(2π .* (0:Nc - 1) ./ Nc))   # the spline through f at the centres
evaluate(c, 1.0) - f(1.0)                      # ≈ 3e-4: fourth order in 1/Nc
```

## What it is for

In a space-time Newton–Krylov search for a periodic orbit of period ``T``, a field ``a(x, s)`` is
sampled at ``N_s`` equispaced times ``s_l = 2\pi (l-1)/N_s``, with ``s = 2\pi t / T``. The coarse
level of ReSolver's two-grid preconditioner represents slowly varying corrections as

```math
c(s_l) = \sum_{m=1}^{N_c} \phi_m(s_l)\, y_m, \qquad N_c = N_s / t_c ,
```

that is ``c = \Phi y``: the prolongation in time is the collocation matrix ``\Phi``, and the
restriction, its transpose. The coarse operator needs the time derivative of the coarse fields,
which on splines is exact: ``\partial_s c(s_l) = (\Phi' y)_l``. Its blocks are the discrete
products ``\Phi^{\mathsf T}\Phi`` and ``\Phi^{\mathsf T}\Phi'``, which approximate the exact Gram
matrices ``M`` and ``D`` (to 0.15% with ``t_c = 2``, 0.01% with ``t_c = 4``).

How coarse can the time grid be? The figure below answers it for a harmonic ``\cos ns``: about 5
splines per period interpolate it to 1%, and the Galerkin derivative ``M^{-1}D`` differentiates it
to 0.01% with the same number of points, far better than centred differences of order 8.

![Fourth-order convergence of interpolation; interpolation error of a harmonic against the number
of splines per period; modified wavenumber of the Galerkin derivative on splines, against centred
differences of order 2, 4 and 8.](assets/accuracy.png)
