"""Coarsening in time: B-splines against temporal Fourier modes, at the orbit T ≈ 25 (dense).

The coarse space holds the first r = 16 sine modes in space and, in time, either periodic cubic
B-splines on a grid t_c times coarser (as in TwoGrid), every slice, or the temporal Fourier modes
|n| ≤ N (a spectral coarsening in time, the analogue of the sines in space). For each, the exact
Galerkin coarse operator K = Pᵀ𝒥P, the cycle E = S² (I - P K⁻¹ Pᵀ𝒥) S², its spectral radius and
its eigenvalues above 0.1, against the number of coarse unknowns; and the structure of K: banded in
time for the splines, coupled across all harmonics within the temporal bandwidth of the orbit for
Fourier modes. The neutral profile is left out here, for both. Writes timecoarse.npz.

    python3 timecoarse.py
"""
import os
import numpy as np
import kslib as K
from kslib import PHI

A, om = K.orbit(1); Ns = A.shape[1]; n = 32 * Ns + 1; r = 16
J = K.read_dense("M1_J.bin", n); Bi = K.read_dense("M1_Binv.bin", n)
S = np.eye(n) - Bi @ J; S2 = S @ S
s = 2 * np.pi * np.arange(Ns) / Ns

def bsp(u):
    v = np.abs(u)
    return np.where(v < 1, (4 - 6 * v ** 2 + 3 * v ** 3) / 6, np.where(v < 2, (2 - v) ** 3 / 6, 0.0))

def splines(Nb):
    D = 2 * np.pi / Nb
    return np.array([[bsp(((sl - D * m + np.pi) % (2 * np.pi) - np.pi) / D) for m in range(Nb)] for sl in s])

def fourier(N):
    cols = [np.ones(Ns)] + [f(nn * s) for nn in range(1, N + 1) for f in (np.cos, np.sin)]
    return np.array(cols).T / np.sqrt(Ns)

def cycle(Tb):
    P = np.zeros((n, r * Tb.shape[1] + 1))
    P[:-1, :-1] = np.kron(Tb, PHI[:, :r]); P[-1, -1] = 1.0
    Kc = P.T @ J @ P
    C = np.eye(n) - P @ np.linalg.solve(Kc, P.T @ J)
    l = np.linalg.eigvals(S2 @ C @ S2)
    return Kc, np.abs(l).max(), int(np.sum(np.abs(l) > 0.1))

out = {"spline": [], "fourier": []}
for tc in (1, 2, 3, 4, 6, 8):
    Tb = np.eye(Ns) if tc == 1 else splines(round(Ns / tc))
    Kc, rho, cnt = cycle(Tb)
    out["spline"].append((Tb.shape[1], rho, cnt))
    print(f"splines t_c = {tc}: {Tb.shape[1]} time functions, ρ(E) = {rho:.3g}, |λ| > 0.1: {cnt}")
    if tc == 2: Ks = Kc
for N in (3, 5, 7, 9, 12, 15, 18, 24, 36):
    Tb = fourier(N)
    Kc, rho, cnt = cycle(Tb)
    out["fourier"].append((Tb.shape[1], rho, cnt, N * om))
    print(f"Fourier |n| ≤ {N} (|ω n| ≤ {N * om:.2f}): {Tb.shape[1]} time functions, ρ(E) = {rho:.3g}, |λ| > 0.1: {cnt}")
    if N == 18: Kf = Kc

# coupling between temporal harmonics: block norms of 𝒥 in space-time Fourier, k, k' ≤ 16, against
# the physical frequency distance ω |n - n'|
Jh = K.op_fourier(J, Ns)
nord, _ = K.n_order(Ns)
Jsub = Jh.reshape((32, Ns, 32, Ns))[:r, :, :r, :]
dist, val = [], []
for i in range(Ns):
    for j in range(Ns):
        if i != j:
            dist.append(om * abs(nord[i] - nord[j])); val.append(np.linalg.norm(Jsub[:, i, :, j], 2))
dist, val = np.array(dist), np.array(val)
bins = np.arange(0, 9, 0.25); prof = np.array([val[(dist >= a) & (dist < a + 0.25)].max() if np.any((dist >= a) & (dist < a + 0.25)) else np.nan for a in bins])
diag = np.median([np.linalg.norm(Jsub[:, i, :, i], 2) for i in range(Ns)])
print("largest coupling between harmonics, relative to the diagonal blocks, against ω|n - n'|:")
for a, p in zip(bins[::4], prof[::4]): print(f"   {a:4.1f}: {p / diag:.1e}")
np.savez(os.path.join(K.HERE, "timecoarse.npz"), spline=np.array(out["spline"]), fourier=np.array(out["fourier"]),
         Ks=Ks, Kf=Kf, bins=bins, prof=prof / diag, om=om)
