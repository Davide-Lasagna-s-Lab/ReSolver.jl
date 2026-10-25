"""Restriction in Fourier, coarse problem in the time domain: a test at the orbit T ≈ 25 (dense).

The coarse unknowns are the values of the first r = 16 sine coefficients at Nc equispaced coarse
times (Nc odd). Transfers in time are spectral: the prolongation P_t is trigonometric interpolation
from the coarse to the fine slices (zero padding in temporal Fourier), the restriction R_t the
truncation of the temporal Fourier series to |n| <= (Nc-1)/2 sampled at the coarse times,
R_t = (Nc/Ns) P_tᵀ, an exact low-pass. Coarse operators compared:

    galerkin       R 𝒥 P, exact (dense in time: the band-limited basis functions are global)
    colloc-spec    rediscretised: -(L0 + A(s_m)) at the coarse times (collocation, the orbit
                   interpolated there) + ω D with the spectral derivative of the coarse grid
    colloc-fd8     the same with centred differences of order 8 (local in time: banded)
    colloc-c4      the same with the compact derivative of cubic-spline interpolation,
                   (y'_{m-1} + 4y'_m + y'_{m+1})/6 = (y_{m+1} - y_{m-1})/2Δ: banded after multiplying
                   the coarse equations by the left stencil
    colloc-cg      the same with the Galerkin derivative of the B-splines, M⁻¹D (PeriodicBSplines):
                   nearly spectral, and banded after multiplying by M
    splines        B-splines on Nc coarse points, exact Galerkin (as in TwoGrid)

The border (period, phase condition) is Galerkin in all cases. For each, the spectral radius of
the cycle V(2,2) and its eigenvalues above 0.1. Prints the table and writes fouriertransfer.npz.

    python3 fouriertransfer.py
"""
import os
import numpy as np
import kslib as K
from kslib import PHI, KAPPA, L, NX

A, om = K.orbit(1); Ns = A.shape[1]; n = 32 * Ns + 1; r = 16
J = K.read_dense("M1_J.bin", n); Bi = K.read_dense("M1_Binv.bin", n)
S = np.eye(n) - Bi @ J; S2 = S @ S
s = 2 * np.pi * np.arange(Ns) / Ns

def trig_interp(Nc, Ns_):
    """Ns_ x Nc matrix of trigonometric interpolation from Nc equispaced points to Ns_ points (Nc odd)."""
    sc = 2 * np.pi * np.arange(Nc) / Nc; sf = 2 * np.pi * np.arange(Ns_) / Ns_
    N = (Nc - 1) // 2
    d = sf[:, None] - sc[None, :]
    return (1 + 2 * sum(np.cos(k * d) for k in range(1, N + 1))) / Nc

def spectral_derivative(Nc):
    sc = 2 * np.pi * np.arange(Nc) / Nc; N = (Nc - 1) // 2
    d = sc[:, None] - sc[None, :]
    return -2 * sum(k * np.sin(k * d) for k in range(1, N + 1)) / Nc       # d/ds of the interpolant

def fd8(Nc):
    h = 4
    Am = np.array([[float(j) ** (2 * i - 1) for j in range(1, h + 1)] for i in range(1, h + 1)])
    w = np.linalg.solve(Am, [0.5 if i == 0 else 0.0 for i in range(h)])
    D = np.zeros((Nc, Nc)); ds = 2 * np.pi / Nc
    for m in range(Nc):
        for j in range(1, h + 1):
            D[m, (m + j) % Nc] += w[j - 1] / ds; D[m, (m - j) % Nc] -= w[j - 1] / ds
    return D

def compact(left, right, Nc):
    """M⁻¹D for circulant stencils left (of M) and right (of D, divided by Δ), centred."""
    h = len(left) // 2; ds = 2 * np.pi / Nc
    M = np.zeros((Nc, Nc)); Dm = np.zeros((Nc, Nc))
    for m in range(Nc):
        for j in range(-h, h + 1):
            M[m, (m + j) % Nc] += left[j + h]; Dm[m, (m + j) % Nc] += right[j + h] / ds
    return np.linalg.solve(M, Dm)
C4 = (np.array([1, 4, 1]) / 6, np.array([-0.5, 0, 0.5]))
CG = (np.array([1, 120, 1191, 2416, 1191, 120, 1]) / 5040, np.array([-1, -56, -245, 0, 245, 56, 1]) / 720)

def bsp(u):
    v = np.abs(u)
    return np.where(v < 1, (4 - 6 * v ** 2 + 3 * v ** 3) / 6, np.where(v < 2, (2 - v) ** 3 / 6, 0.0))

# spatial operator at any time, in sine coordinates: L0 + A(s) restricted to r sines
SG = np.sin(2 * np.pi * np.outer(K.X_FULL, np.arange(1, 33)) / L); proj = SG.T * 2 / NX
kx = 2 * np.pi / L * np.fft.fftfreq(NX, 1 / NX)
ddx = lambda f: np.real(np.fft.ifft(1j * kx * np.fft.fft(f)))
def spatial(u):
    Adv = np.column_stack([proj @ -ddx(u * SG[:, k]) for k in range(32)])
    return (np.diag(KAPPA ** 2 - KAPPA ** 4) + Adv)[:r, :r]

def prolongation(Tt):
    P = np.zeros((n, r * Tt.shape[1] + 1))
    P[:-1, :-1] = np.kron(Tt, PHI[:, :r]); P[-1, -1] = 1.0
    return P

def cycle(P, R, Kc):
    C = np.eye(n) - P @ np.linalg.solve(Kc, R @ J)
    l = np.linalg.eigvals(S2 @ C @ S2)
    return np.abs(l).max(), int(np.sum(np.abs(l) > 0.1))

rows = []
for Nc in (37, 25, 19, 13, 9):
    Pt = trig_interp(Nc, Ns)
    P = prolongation(Pt)
    R = P.T.copy(); R[:-1, :] *= Nc / Ns                                   # R_t = (Nc/Ns) P_tᵀ in time
    res = {}
    res["galerkin"] = cycle(P, R, R @ J @ P)
    # rediscretised coarse operators: orbit interpolated to the coarse times
    Ac = (trig_interp(Ns, Nc) @ A.T).T                                    # 65 x Nc
    for name, D in (("colloc-spec", spectral_derivative(Nc)), ("colloc-fd8", fd8(Nc)),
                    ("colloc-c4", compact(*C4, Nc)), ("colloc-cg", compact(*CG, Nc))):
        Kc = R @ J @ P                                                      # border from Galerkin
        Kf = np.kron(om * D, np.eye(r))
        for m in range(Nc):
            Kf[m * r:(m + 1) * r, m * r:(m + 1) * r] -= spatial(Ac[:, m])
        Kc[:-1, :-1] = Kf
        res[name] = cycle(P, R, Kc)
    D_ = 2 * np.pi / Nc
    Tt = np.array([[bsp(((sl - D_ * m + np.pi) % (2 * np.pi) - np.pi) / D_) for m in range(Nc)] for sl in s])
    Ps = prolongation(Tt)
    res["splines"] = cycle(Ps, Ps.T, Ps.T @ J @ Ps)
    rows.append((Nc, res))
    print(f"Nc = {Nc:2d} (t_c ≈ {Ns / Nc:.1f}, coarse Nyquist |ω| ≈ {om * (Nc - 1) / 2:.1f}): " +
          "  ".join(f"{k}: ρ {v[0]:.3g}, {v[1]} > 0.1" for k, v in res.items()))
np.savez(os.path.join(K.HERE, "fouriertransfer.npz"), Nc=[r_[0] for r_ in rows],
         **{k: np.array([r_[1][k] for r_ in rows]) for k in ("galerkin", "colloc-spec", "colloc-fd8", "colloc-c4", "colloc-cg", "splines")})
