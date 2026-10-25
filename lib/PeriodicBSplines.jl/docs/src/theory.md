# Theory

## The cardinal cubic B-spline

The cardinal cubic B-spline is the convolution of four unit boxes ``\chi_{[-1/2, 1/2]}``:

```math
B(u) = \begin{cases}
\tfrac16 \left(4 - 6u^2 + 3|u|^3\right), & |u| < 1, \\
\tfrac16 (2 - |u|)^3, & 1 \le |u| < 2, \\
0, & |u| \ge 2 .
\end{cases}
```

It is a piecewise cubic with knots at the integers, twice continuously differentiable, positive on
``(-2, 2)`` and zero outside, with ``\int B = 1``, ``B(0) = 2/3`` and ``B(\pm 1) = 1/6``. Being a
fourfold convolution of a box, its Fourier transform is

```math
\hat B(\xi) = \left(\frac{\sin(\xi/2)}{\xi/2}\right)^4 ,
```

which decays like ``\xi^{-4}``: B-splines are smooth, and they filter short wavelengths. `bspline`,
`dbspline` and `d2bspline` evaluate ``B``, ``B'`` and ``B''``.

## Periodic splines on a uniform grid

On the circle ``[0, 2\pi)`` with ``N_c`` equispaced centres ``s_m = 2\pi (m-1)/N_c`` and spacing
``\Delta = 2\pi/N_c``, the periodic splines are

```math
\phi_m(s) = \sum_{p \in \mathbb Z} B\!\left(\frac{s - s_m - 2\pi p}{\Delta}\right), \qquad m = 1, \dots, N_c .
```

They are a basis of the periodic cubic splines with knots at the ``s_m``, a space of dimension
``N_c``. Each ``\phi_m`` is supported on four intervals, so at any point at most four splines are
nonzero; and they are a partition of unity, ``\sum_m \phi_m(s) = 1``. At least four splines are
needed for the four that overlap a point to be distinct.

## Collocation matrices

On ``N_s`` equispaced points ``s_l = 2\pi (l-1)/N_s``, `collocation(Ns, Nc)` returns the sparse
``N_s \times N_c`` matrices

```math
\Phi_{lm} = \phi_m(s_l), \qquad \Phi'_{lm} = \phi_m'(s_l) = \frac{1}{\Delta} B'\!\left(\frac{s_l - s_m}{\Delta}\right),
```

with at most four nonzeros per row. ``\Phi y`` evaluates the spline with coefficients ``y`` on the
fine grid, and ``\Phi' y`` its derivative, exactly.

![The collocation matrix with Ns = 40, Nc = 8, the Gram matrix and the derivative Gram matrix: both
circulant and banded, with the corners that make them periodic.](assets/matrices.png)

## Gram matrices

The Gram (mass) matrix and the derivative Gram matrix are

```math
M_{mm'} = \int_0^{2\pi} \phi_m \phi_{m'} \, ds = \Delta\, B_7(m' - m), \qquad
D_{mm'} = \int_0^{2\pi} \phi_m \phi_{m'}' \, ds ,
```

where ``B_7 = B * B`` is the cardinal B-spline of degree 7. Both are circulant with seven diagonals
(``N_c \ge 7``):

```math
M:\ \frac{\Delta}{5040}\,(1,\ 120,\ 1191,\ 2416,\ 1191,\ 120,\ 1), \qquad
D:\ \frac{1}{720}\,(-1,\ -56,\ -245,\ 0,\ 245,\ 56,\ 1).
```

``D`` is antisymmetric, as integration by parts requires on the circle. `gram` and
`derivative_gram` compute them by Gauss–Legendre quadrature on each knot interval, exact for these
piecewise polynomials. Their discrete versions on a fine grid with ``t_c = N_s/N_c`` points per
spacing, ``\frac{2\pi}{N_s}\Phi^{\mathsf T}\Phi`` and ``\frac{2\pi}{N_s}\Phi^{\mathsf T}\Phi'``,
are the trapezoidal rule applied to the same integrands; they agree with ``M`` and ``D`` to 5% and
3% with ``t_c = 1``, 0.15% with ``t_c = 2``, ``10^{-4}`` with ``t_c = 4``.

## Fourier symbols and the Galerkin derivative

A circulant matrix acts on the harmonic coefficients ``c_m = e^{i\theta m}`` by multiplication by
its symbol. For ``M`` and ``D``

```math
\hat M(\theta) = \frac{\Delta}{5040}\left(2416 + 2382\cos\theta + 240\cos 2\theta + 2\cos 3\theta\right), \qquad
\hat D(\theta) = \frac{i}{360}\left(245\sin\theta + 56\sin 2\theta + \sin 3\theta\right),
```

(`mass_symbol` returns ``\hat M/\Delta``, `derivative_symbol` returns ``\hat D``). The Galerkin
derivative ``M^{-1}D`` therefore multiplies the harmonic by ``i\theta^*/\Delta`` with the
*modified wavenumber*

```math
\theta^*(\theta) = \frac{\operatorname{Im}\hat D(\theta)}{\hat M(\theta)/\Delta},
```

against the exact ``\theta``; `modified_wavenumber` computes it. With ``p = 2\pi/\theta`` splines
per period of the harmonic:

| points per period | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|
| splines, Galerkin ``\theta^*/\theta`` | 0.988 | 0.9994 | 0.9999 | 1.0000 | 1.0000 |
| centred differences, order 8 | 0.824 | 0.970 | 0.994 | 0.998 | 1.000 |
| centred differences, order 4 | 0.620 | 0.849 | 0.931 | 0.965 | 0.988 |
| centred differences, order 2 | 0.414 | 0.637 | 0.757 | 0.827 | 0.900 |

The Galerkin derivative on splines behaves like a compact scheme: it is accurate up to close to
the shortest resolvable wavelength. Like every centred scheme, it gives zero for the sawtooth
``\theta = \pi``.

## Interpolation

The spline ``\sum_m c_m \phi_m`` takes the values ``f_m`` at the centres if

```math
\tfrac16 \left(c_{m-1} + 4c_m + c_{m+1}\right) = f_m ,
```

a circulant system whose symbol ``(4 + 2\cos\theta)/6 \ge 1/3`` never vanishes: interpolation is
always possible and well conditioned. `interpolate` solves it with a discrete Fourier transform;
`evaluate` evaluates the result anywhere. The error is ``O(\Delta^4)`` for smooth functions. For a
harmonic, it is about 10% with 3.3 splines per period, 1% with 5 and 0.1% with 8.

## How many splines per period: the time coarsening of ReSolver

In the two-grid preconditioner the coarse time grid must resolve the slow dynamics that the
smoother cannot damp, which in Kuramoto–Sivashinsky have physical frequencies ``|\omega| \lesssim
2``, periods of about 3 time units and more. With a fine time step ``\Delta t``, the coarse spacing
is ``t_c \Delta t``, and a harmonic of frequency ``\omega`` has ``2\pi/(\omega\, t_c \Delta t)``
splines per period. With ``\Delta t \approx 0.35``, ``t_c = 2`` gives about 4.5 splines per period of
the fastest slow component, ``t_c = 4`` about 2.2, and ``t_c = 8`` about 1.1. The table above
explains why ``t_c = 2`` is safe, ``t_c = 4`` needs help from the smoother (two smoothing steps on
each side of the coarse correction), and ``t_c = 8`` fails.
