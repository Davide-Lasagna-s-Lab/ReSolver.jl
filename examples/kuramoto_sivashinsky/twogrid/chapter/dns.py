"""A long chaotic trajectory of Kuramoto-Sivashinsky, u_t = -u u_x - u_xx - u_xxxx on [0, 39) in the
odd subspace, by ETDRK4 (Kassam & Trefethen 2005) on 128 Fourier modes, and its POD: the left singular
vectors of the snapshots of the first 32 sine coefficients (no mean, the odd subspace has none),
ordered by energy. Writes pod/pod_attractor.txt (32 x 32, the modes in sine coordinates, by column),
pod/energy_attractor.txt and pod/dns_snapshots.npy (sine coefficients, 32 x snapshots). Convergence:
the POD of records of increasing length, and of four disjoint quarters, against the full record
(pod/convergence.txt: length, energies of modes 1..16, largest principal angle of the 4/8/12/16-mode
subspaces to those of the full record, in degrees).

    python3 dns.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L, N, h = 39.0, 128, 0.02
t_skip, t_end, every = 200.0, 40200.0, 0.5
k = 2 * np.pi / L * np.fft.rfftfreq(N, 1 / N)
Lin = k ** 2 - k ** 4
E, E2 = np.exp(h * Lin), np.exp(h * Lin / 2)
r = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)          # contour points
LR = h * Lin[:, None] + r[None, :]
Q  = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))
g = -0.5j * k
def nl(v):
    return g * np.fft.rfft(np.fft.irfft(v, N) ** 2)
def odd(v):
    return 1j * v.imag                                            # odd fields: imaginary coefficients

rng = np.random.default_rng(0)
x = np.arange(N) * L / N
u0 = sum(rng.standard_normal() * 0.1 * np.sin(2 * np.pi * m * x / L) for m in range(1, 9))
v = odd(np.fft.rfft(u0))
snaps = []
nsteps, nevery = int(round(t_end / h)), int(round(every / h))
for n in range(1, nsteps + 1):
    Nv = nl(v)
    a = E2 * v + Q * Nv; Na = nl(a)
    b = E2 * v + Q * Na; Nb = nl(b)
    c = E2 * a + Q * (2 * Nb - Nv); Nc = nl(c)
    v = odd(E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3)
    if n * h > t_skip and n % nevery == 0:
        snaps.append(-2 * v[1:33].imag / N)                       # u = Σ b_k sin(2π k x / L)
S = np.array(snaps).T
U, s, _ = np.linalg.svd(S, full_matrices=False)
for i in range(32):
    if U[np.argmax(np.abs(U[:, i])), i] < 0: U[:, i] *= -1
os.makedirs(os.path.join(HERE, "pod"), exist_ok=True)
np.savetxt(os.path.join(HERE, "pod", "pod_attractor.txt"), U)
np.savetxt(os.path.join(HERE, "pod", "energy_attractor.txt"), s ** 2 / np.sum(s ** 2))
np.save(os.path.join(HERE, "pod", "dns_snapshots.npy"), S)
def pod(S):
    U, s, _ = np.linalg.svd(S, full_matrices=False)
    return U, s ** 2 / np.sum(s ** 2)
def angle(U1, U2, r):
    c = np.linalg.svd(U1[:, :r].T @ U2[:, :r], compute_uv=False).min()
    return np.degrees(np.arccos(min(1.0, c)))
rows = []
ns = S.shape[1]
for frac in (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2):
    Ui, ei = pod(S[:, :int(ns * frac)])
    rows.append([ns * frac * every] + list(ei[:16]) + [angle(Ui, U, r) for r in (4, 8, 12, 16)])
for q in range(4):
    Ui, ei = pod(S[:, q * ns // 4:(q + 1) * ns // 4])
    rows.append([-(q + 1)] + list(ei[:16]) + [angle(Ui, U, r) for r in (4, 8, 12, 16)])
np.savetxt(os.path.join(HERE, "pod", "convergence.txt"), np.array(rows))
print("record length, largest principal angle (deg) of the 4/8/12/16-mode POD subspaces to the full record:")
for row in rows:
    print(f"   {('quarter ' + str(int(-row[0]))) if row[0] < 0 else f'{row[0]:7.0f}'}: " + " ".join(f"{v:6.2f}" for v in row[-4:]))
e = np.cumsum(s ** 2) / np.sum(s ** 2)
print(f"{S.shape[1]} snapshots, t = {t_skip}..{t_end}; rms u = {np.sqrt(np.mean(np.sum(S ** 2, 0)) / 2):.3f}")
print("POD modes for 90/99/99.9/99.99% of the energy:", [int(np.searchsorted(e, p) + 1) for p in (0.9, 0.99, 0.999, 0.9999)])
print("energy fraction of the sine modes k = 1..12:", np.round(np.mean(S ** 2, 1)[:12] / np.mean(np.sum(S ** 2, 0)), 4))
print("mean sine wavenumber of POD modes 1..16:", np.round([np.arange(1, 33) @ U[:, i] ** 2 for i in range(16)], 1))
