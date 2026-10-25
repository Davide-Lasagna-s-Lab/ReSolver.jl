"""Shared helpers of the chapter figures: data, bases and transforms of Kuramoto-Sivashinsky, L = 39,
odd subspace, on the grid of ReSolver (Nx = 65 points, 32 sine modes)."""
import os
import numpy as np
from scipy.linalg import eig

HERE = os.path.dirname(os.path.abspath(__file__))
TG = os.path.join(HERE, "..")
DAT = os.path.join(TG, "data")
RES = os.path.join(TG, "results")
DENSE = os.path.join(HERE, "dense")
SLOW = os.path.join(TG, "explore", "slowdirs")
L, NX = 39.0, 65
X_HALF = np.arange(1, 33) * L / NX                       # points j = 1..32 of an odd field (the odd basis)
X_FULL = np.arange(NX) * L / NX
KAPPA = 2 * np.pi * np.arange(1, 33) / L
# orthonormal sine basis on the half grid: column k-1 is sin(2πk x/L)
PHI = np.sin(2 * np.pi * np.outer(X_HALF, np.arange(1, 33)) / L) / np.sqrt(NX / 4)

BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#285aaa", "#be3232", "#288c50", "#6e6e6e", "#e08a00", "#7b3fa0"

def style(plt):
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

def orbit(M):
    """Field on the grid (65 x Ns) and frequency ω of the prime orbit M = 1, 2, 4, 8."""
    if M == 1:
        return np.loadtxt(os.path.join(DAT, "orbit.txt")), np.loadtxt(os.path.join(DAT, "params.txt"))[1]
    A = np.loadtxt(os.path.join(DAT, f"long{M}_orbit.txt"))
    return A, np.exp(np.loadtxt(os.path.join(DAT, f"long{M}_orbit_p.txt")).ravel()[0])

def sine_coeffs(A):
    """Sine coefficients b_k (32 x Ns) of fields on the grid, u = Σ b_k sin(2πk x/L)."""
    S = np.sin(2 * np.pi * np.outer(np.arange(1, 33), X_FULL) / L) * 2 / NX
    return S @ A

def full_field(Fh):
    """Half-grid values (32 x Ns) of an odd field to the full grid (65 x Ns)."""
    return np.vstack([np.zeros(Fh.shape[1]), Fh, -Fh[::-1]])

def odd_to_sine(v, Ns):
    """A vector of the odd basis (32 Ns, slice-major, + parameter) to sine coordinates, 32 x Ns."""
    return PHI.T @ v[:32 * Ns].reshape((Ns, 32)).T

def n_order(Ns):
    """Temporal indices n = 0, 1, -1, 2, -2, … and their positions in an FFT of length Ns."""
    n = np.fft.fftfreq(Ns, 1 / Ns).astype(int)
    order = np.argsort(2 * np.abs(n) - (n > 0))
    return n[order], order

def to_fourier(V, Ns):
    """Columns of V (odd basis, field part 32 Ns, slice-major) to space-time Fourier coefficients,
    k-major (k = 1..32), n ordered 0, 1, -1, 2, -2, … inside; unitary."""
    m = V.shape[1]
    C = np.einsum("jk,ljm->klm", PHI, V[:32 * Ns].reshape((Ns, 32, m)))
    _, order = n_order(Ns)
    Ch = np.fft.fft(C, axis=1, norm="ortho")[:, order, :]
    return Ch.reshape((32 * Ns, m))

def op_fourier(A, Ns):
    """The field block of an operator on the odd basis in the space-time Fourier basis, W A W^H."""
    B = to_fourier(A[:32 * Ns, :32 * Ns], Ns)
    return to_fourier(B.conj().T, Ns).conj().T

def read_dense(name, n):
    return np.fromfile(os.path.join(DENSE, name)).reshape((n, n)).T

def read_eig(name):
    a = np.loadtxt(os.path.join(DENSE, name))
    return a[:, 0] + 1j * a[:, 1]

def slow_directions(M):
    """Unit-norm real slow directions of J B^-1 (harmonic Ritz nearest 0, 20 M), (dirs, 32, Ns) half-grid."""
    A, om = orbit(M); Ns = A.shape[1]; m = 120 * M; N = NX * Ns
    H = np.fromfile(os.path.join(SLOW, f"H_{M}.bin")).reshape((m, m + 1)).T
    Z = np.fromfile(os.path.join(SLOW, f"Z_{M}.bin")).reshape((m, N + 1))
    th, Y = eig(H.T @ H, H[:m, :m].T)
    ok = np.isfinite(th); th, Y = th[ok], Y[:, ok]
    res = np.array([np.linalg.norm(H @ Y[:, i] - th[i] * np.r_[Y[:, i], 0]) / np.linalg.norm(Y[:, i]) for i in range(len(th))])
    good = np.where(res < 1e-4)[0]
    good = good[np.argsort(np.abs(th[good]))][:20 * M]
    vecs = []
    for i in good:
        z = Y[:, i] @ Z[:, :N]
        for part in ((z.real,) if abs(th[i].imag) < 1e-12 else (z.real, z.imag)):
            f = part.reshape((Ns, NX)).T[1:33, :]
            vecs.append(f / np.linalg.norm(f))
    return np.array(vecs), A, om
