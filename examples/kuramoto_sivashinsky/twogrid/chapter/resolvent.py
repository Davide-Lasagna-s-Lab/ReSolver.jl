"""The smoother as a resolvent, and coarse bases from resolvent analysis.

One smoothing step multiplies the error by S = B^-1 A = (ω ∂s - L0)^-1 A(s): the resolvent of the
linear operator L0 at every temporal frequency, applied to the advection by the orbit, A(s) v =
-∂x(u(s) v), acting as a forcing. Its gains and modes are those of an input-output (resolvent)
analysis about the time-periodic orbit. Three spatial bases are built from them, each from the
response (output) modes of the operator named, collected and weighted by their gains, then
compressed by an SVD into one orthonormal basis of the 32 sine coordinates ordered by importance:

    resS      the space-time smoother S itself, at the orbit T ≈ 25 (dense, from operators.jl):
              its left singular vectors with gain > 0.5
    resF      the frozen-time smoother (iω - L0)^-1 A(s_l), slice by slice, |ω| ≤ 3: no
              space-time operator needed, 32 x 32 matrices only
    resM      the classical resolvent about the mean of the orbit, (iω - L0 - A(ū))^-1, |ω| ≤ 3
              (McKeon & Sharma 2010)

Writes bases/<name>.txt (32 x 32, sine coordinates, by column), resolvent.npz (gains and modes for
the figures) and prints a summary.

    python3 resolvent.py
"""
import os
import numpy as np
import kslib as K
from kslib import PHI, KAPPA, L, NX

HERE = K.HERE
os.makedirs(os.path.join(HERE, "bases"), exist_ok=True)
A, om = K.orbit(1); Ns = A.shape[1]; n = 32 * Ns + 1

# ---- the space-time smoother, dense ----
J = K.read_dense("M1_J.bin", n)
Bi = K.read_dense("M1_Binv.bin", n)
S = np.eye(n) - Bi @ J
U, sig, Vt = np.linalg.svd(S)
print(f"S at T = {2 * np.pi / om:.1f}: largest gain {sig[0]:.2f}; gains > 1: {np.sum(sig > 1)}, > 0.5: {np.sum(sig > 0.5)}, > 0.2: {np.sum(sig > 0.2)} of {n}")

def compress(modes, weights):
    """modes: 32 x m (complex) sine coordinates; one orthonormal basis by weighted SVD."""
    M = modes * weights[None, :]
    M = np.hstack([M.real, M.imag])
    Ub, sb, _ = np.linalg.svd(M, full_matrices=False)
    for i in range(32):
        if Ub[np.argmax(np.abs(Ub[:, i])), i] < 0: Ub[:, i] *= -1
    return Ub, sb

# response modes of S: slices of the left singular vectors, in sine coordinates
sel = np.where(sig > 0.5)[0]
slices = np.hstack([K.odd_to_sine(U[:, i], Ns) for i in sel])            # 32 x (len(sel) Ns)
wts = np.repeat(sig[sel], Ns)
resS, sS = compress(slices, wts)
slices_in = np.hstack([K.odd_to_sine(Vt[i], Ns) for i in sel])
resS_in, _ = compress(slices_in, wts)

# ---- operators in sine coordinates on the grid of ReSolver (pseudo-spectral, 65 points) ----
SG = np.sin(2 * np.pi * np.outer(K.X_FULL, np.arange(1, 33)) / L)         # grid values of the sines, 65 x 32
proj = SG.T * 2 / NX                                                       # grid values to sine coefficients
kx = 2 * np.pi / L * np.fft.fftfreq(NX, 1 / NX)
def ddx(f):
    return np.real(np.fft.ifft(1j * kx * np.fft.fft(f)))
def advection(u):
    """32 x 32 matrix of v -> -∂x(u v) on sine coefficients."""
    return np.column_stack([proj @ -ddx(u * SG[:, k]) for k in range(32)])
L0 = np.diag(KAPPA ** 2 - KAPPA ** 4)
ws = np.linspace(-3, 3, 61)

# ---- frozen-time smoother, slice by slice ----
modesF, gainsF = [], []
for l in range(Ns):
    Al = advection(A[:, l])
    for w in ws:
        Ul, sl, _ = np.linalg.svd(np.linalg.solve(1j * w * np.eye(32) - L0, Al))
        modesF.append(Ul[:, :8]); gainsF.append(sl[:8])
resF, sF = compress(np.hstack(modesF), np.hstack(gainsF))

# ---- classical resolvent about the mean of the orbit ----
ubar = A.mean(axis=1)
Lbar = L0 + advection(ubar)
modesM, gainsM = [], []
for w in ws:
    Um, sm, _ = np.linalg.svd(np.linalg.inv(1j * w * np.eye(32) - Lbar))
    modesM.append(Um[:, :8]); gainsM.append(sm[:8])
resM, sM = compress(np.hstack(modesM), np.hstack(gainsM))
print("mean-flow resolvent: largest gain over |ω| ≤ 3:", f"{max(g[0] for g in gainsM):.1f}",
      f"at ω = {ws[np.argmax([g[0] for g in gainsM])]:.2f}; eigenvalues of L̄ with Re > 0: {np.sum(np.linalg.eigvals(Lbar).real > 0)}")

for name, Bm in (("resS", resS), ("resS_in", resS_in), ("resF", resF), ("resM", resM)):
    np.savetxt(os.path.join(HERE, "bases", f"{name}.txt"), Bm)
    kbar = [np.arange(1, 33) @ Bm[:, i] ** 2 for i in range(12)]
    print(f"{name:8s} modes 1..12, mean sine wavenumber: " + " ".join(f"{v:.1f}" for v in kbar))

# how much of the amplified subspace of S each basis holds: fraction of the gain-weighted response
# modes outside the first r modes of each basis
podA = np.loadtxt(os.path.join(HERE, "pod", "pod_attractor.txt"))
svd = np.loadtxt(os.path.join(K.TG, "explore", "global_basis_1.txt"))
bases = {"sine": np.eye(32), "podA": podA, "svd": svd, "resS": resS, "resF": resF, "resM": resM}
Sl = slices * wts[None, :]
tot = np.sum(np.abs(Sl) ** 2)
print("fraction of the amplified response of S outside the first r modes (r = 8 / 12 / 16):")
held = {}
for b, Bm in bases.items():
    held[b] = [np.sqrt(np.sum(np.abs(Sl - Bm[:, :r] @ (Bm[:, :r].T @ Sl)) ** 2) / tot) for r in range(1, 33)]
    print(f"   {b:6s} " + " / ".join(f"{held[b][r - 1]:.3f}" for r in (8, 12, 16)))

np.savez(os.path.join(HERE, "resolvent.npz"), sig=sig, Ulead=U[:, :12], Vlead=Vt[:12].T,
         ws=ws, gainsF=np.array(gainsF).reshape((Ns, len(ws), 8)), gainsM=np.array(gainsM),
         modesM=np.array(modesM), held=np.array([held[b] for b in bases]), held_names=np.array(list(bases)),
         resS=resS, resF=resF, resM=resM, sS=sS, sF=sF, sM=sM)
