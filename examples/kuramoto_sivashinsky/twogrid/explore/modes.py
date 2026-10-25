"""The spatial modes of the slow directions, slice by slice, against the sine modes.

From the output of slowdirs.jl: the slow directions z of the preconditioned Jacobian J B^-1 (harmonic
Ritz vectors nearest 0, 20 M of them, mapped by B^-1) at the prime orbits M = 1 and 8; and, from the
dense operators at M = 1, the Floquet profiles, the spectrum of one smoothing step S = I - B^-1 J and
the exact two-grid operator for three spatial bases. Prints the measurements and writes the figures
modes_*.pdf, and the global SVD bases (global_basis_M.txt, used by svdbasis.jl).

    python3 modes.py [outdir]
"""
import sys, os
import numpy as np
from scipy.linalg import eig
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#285aaa", "#be3232", "#288c50", "#6e6e6e", "#e08a00", "#7b3fa0"
HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, "slowdirs")
DAT = os.path.join(HERE, "..", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "figs")
os.makedirs(OUT, exist_ok=True)
L, NX = 39.0, 65
x_half = np.arange(1, 33) * L / NX                       # grid points j = 1..32 (0-based) of an odd field
# orthonormal sine basis on these points: column k-1 is sin(2πk x/L), k = 1..32
PHI = np.sin(2 * np.pi * np.outer(x_half, np.arange(1, 33)) / L) / np.sqrt(NX / 4)
assert np.allclose(PHI.T @ PHI, np.eye(32))

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=130, bbox_inches="tight")

def full(u):                                             # odd extension of half-grid values to x_j, j = 0..64
    return np.r_[0.0, u, -u[::-1]]

x_full = np.arange(NX) * L / NX

def orbit(M):
    if M == 1:
        A = np.loadtxt(os.path.join(DAT, "orbit.txt")); om = np.loadtxt(os.path.join(DAT, "params.txt"))[1]
    else:
        A = np.loadtxt(os.path.join(DAT, f"long{M}_orbit.txt")); om = np.exp(np.loadtxt(os.path.join(DAT, f"long{M}_orbit_p.txt")).ravel()[0])
    return A, om

def slow_directions(M):
    """Unit-norm real slow directions, as an array (directions, 32, Ns) of half-grid values."""
    A, om = orbit(M); Ns = A.shape[1]; m = 120 * M; N = NX * Ns
    H = np.fromfile(os.path.join(SD, f"H_{M}.bin")).reshape((m, m + 1)).T
    Z = np.fromfile(os.path.join(SD, f"Z_{M}.bin")).reshape((m, N + 1))
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
    print(f"M = {M}: T = {2 * np.pi / om:.2f}, Ns = {Ns}: {len(good)} harmonic Ritz pairs nearest 0 "
          f"(|θ| ≤ {np.abs(th[good]).max():.2f}), {len(vecs)} real directions")
    return np.array(vecs), A, om

D = {}
for M in (1, 8):
    V, A, om = slow_directions(M)
    nv, _, Ns = V.shape
    U = np.zeros((Ns, 32, 32)); S = np.zeros((Ns, 32))
    for l in range(Ns):
        u, s, _ = np.linalg.svd(V[:, :, l].T, full_matrices=False)
        U[l][:, :u.shape[1]], S[l][:len(s)] = u, s
    G, sg, _ = np.linalg.svd(V.transpose(1, 0, 2).reshape(32, -1), full_matrices=False)  # one basis for all
    for i in range(32):                                  # sign: largest value positive
        if G[np.argmax(np.abs(G[:, i])), i] < 0: G[:, i] *= -1
    D[M] = dict(V=V, A=A, om=om, U=U, S=S, G=G, sg=sg, Ns=Ns, nv=nv)
    np.savetxt(os.path.join(HERE, f"global_basis_{M}.txt"), PHI.T @ G)   # in sine coordinates, by column

def outside(V, bases):
    """Fraction of the norm of each direction outside a basis per slice, bases[l] (32 × r)."""
    miss = np.zeros(V.shape[0])
    for l in range(V.shape[2]):
        Q = bases[l]; Sl = V[:, :, l].T
        miss += np.sum((Sl - Q @ (Q.T @ Sl)) ** 2, axis=0)
    return np.sqrt(miss)

R = list(range(2, 25, 2))
OUTS = {}
for M in (1, 8):
    d = D[M]; Ns = d["Ns"]
    o = {}
    o["per slice"] = [outside(d["V"], [d["U"][l][:, :r] for l in range(Ns)]) for r in R]
    o["one basis"] = [outside(d["V"], [d["G"][:, :r]] * Ns) for r in R]
    o["sines"] = [outside(d["V"], [PHI[:, :r]] * Ns) for r in R]
    other = D[9 - M]["G"]
    o["one basis, other orbit"] = [outside(d["V"], [other[:, :r]] * Ns) for r in R]
    OUTS[M] = o
    print(f"M = {M}, fraction of the norm outside, median / max over the directions:")
    for k, v in o.items():
        print(f"   {k:24s} " + "  ".join(f"r={r}: {np.median(a):.3f}/{a.max():.3f}" for r, a in zip(R, v) if r in (8, 12, 16, 20)))

# sine content of the modes
for M in (1, 8):
    d = D[M]
    C = np.median(np.array([(PHI.T @ d["U"][l]) ** 2 for l in range(d["Ns"])]), axis=0)   # k × mode
    d["C"] = C
    kbar = [(np.arange(1, 33) @ C[:, i]) / C[:, i].sum() for i in range(32)]
    k99 = [int(np.searchsorted(np.cumsum(C[:, i]) / C[:, i].sum(), 0.99) + 1) for i in range(32)]
    print(f"M = {M}: per-slice modes 1..12, mean sine wavenumber " + " ".join(f"{v:.1f}" for v in kbar[:12]))
    print(f"        sine modes holding 99% of each:       " + " ".join(f"{v}" for v in k99[:12]))
    Cg = (PHI.T @ d["G"]) ** 2
    kg = [(np.arange(1, 33) @ Cg[:, i]) for i in range(32)]
    print(f"        global modes 1..12, mean sine wavenumber " + " ".join(f"{v:.1f}" for v in kg[:12]))
    s = d["sg"] / d["sg"][0]
    print(f"        global singular values / first: " + " ".join(f"{v:.1e}" for v in s[[3, 7, 11, 15, 19, 23]]) + " (modes 4, 8, …, 24)")
    e = np.cumsum(d["sg"] ** 2) / np.sum(d["sg"] ** 2)
    print(f"        global modes for 99%, 99.9%, 99.99% of the energy: {np.searchsorted(e, 0.99) + 1}, {np.searchsorted(e, 0.999) + 1}, {np.searchsorted(e, 0.9999) + 1}")
for r in (8, 12, 16):
    sv = np.linalg.svd(D[1]["G"][:, :r].T @ D[8]["G"][:, :r], compute_uv=False)
    print(f"largest principal angle between the global {r}-mode bases of M = 1 and M = 8: {np.degrees(np.arccos(min(1, sv.min()))):.1f}°")

# Floquet profiles of the slowest ladders, M = 1
d = D[1]; Ns = d["Ns"]; om = d["om"]; nb = 32 * Ns
J = np.fromfile(os.path.join(SD, "J_1.bin")).reshape((nb + 1, nb + 1)).T[:nb, :nb]
lam, W = np.linalg.eig(J)
groups = []
for i in np.argsort(np.abs(lam.real)):
    for gr in groups:
        if abs(lam[i].real - lam[gr[0]].real) < 2e-3:
            gr.append(i); break
    else:
        groups.append([i])
groups = [gr for gr in groups if len(gr) >= 5]
groups.sort(key=lambda gr: abs(np.mean(lam[gr].real)))
FLQ = []
s_grid = 2 * np.pi * np.arange(Ns) / Ns
print("slowest Floquet ladders at M = 1 (exponent μ = -Re λ; profile from the rung nearest the real axis):")
for gr in groups:
    i = min(gr, key=lambda j: abs(lam[j].imag))
    if abs(lam[i].imag) > 0.6 * om or any(abs(-lam[i].real - f[0]) < 1e-2 for f in FLQ):
        continue                                         # not a ladder through the real axis, or a duplicate
    if len(FLQ) == 7: break
    n = lam[i].imag / om
    q = W[:, i].reshape((Ns, 32)).T * np.exp(-1j * n * s_grid)[None, :]     # periodic or antiperiodic part
    q = q * np.exp(-1j * np.angle(np.sum(q ** 2)) / 2); q = q.real / np.linalg.norm(q.real)
    mu = -lam[i].real
    # how much of it the first r sine modes and the first r per-slice SVD modes hold
    fs = [np.sqrt(np.sum((q - PHI[:, :r] @ (PHI[:, :r].T @ q)) ** 2)) for r in (8, 12, 16)]
    fu = [np.sqrt(sum(np.sum((q[:, l] - d["U"][l][:, :r] @ (d["U"][l][:, :r].T @ q[:, l])) ** 2) for l in range(Ns))) for r in (8, 12, 16)]
    kind = "" if abs(n) < 0.02 else (" (antiperiodic)" if abs(abs(n) - 0.5) < 0.02 else " (complex pair)")
    print(f"   μ = {mu:+.3f}{kind}, {len(gr)} rungs; outside 8/12/16 sines "
          + "/".join(f"{v:.3f}" for v in fs) + ", outside 8/12/16 per-slice modes " + "/".join(f"{v:.3f}" for v in fu))
    FLQ.append((mu, kind, q))

# the spectrum of one smoothing step, S = I - B^-1 J, at M = 1: the directions GMRES finds slow
# (|1 - λ| < 0.6, the harmonic Ritz values used above) and those the smoother amplifies (|λ| > 1)
Jb = np.fromfile(os.path.join(SD, "J_1.bin")).reshape((nb + 1, nb + 1)).T
Bi = np.fromfile(os.path.join(SD, "Binv_1.bin")).reshape((nb + 1, nb + 1)).T
S1 = np.eye(nb + 1) - Bi @ Jb
lamS, WS = np.linalg.eig(S1)
def real_slices(idx):
    out = []
    for i in idx:
        v = WS[:nb, i].reshape((Ns, 32)).T
        for part in ((v.real,) if abs(lamS[i].imag) < 1e-10 else (v.real, v.imag)):
            out.append(part / np.linalg.norm(part))
    return np.array(out)
SETS = {"slow for GMRES, $|1 - \\lambda| < 0.6$": np.where(np.abs(1 - lamS) < 0.6)[0],
        "amplified, $|\\lambda| > 1$": np.where(np.abs(lamS) > 1)[0],
        "not damped, $|\\lambda| > 0.5$": np.where(np.abs(lamS) > 0.5)[0]}
print(f"spectrum of S at M = 1 ({nb + 1} eigenvalues): largest |λ| = {np.abs(lamS).max():.2f}; "
      + ", ".join(f"{k.split(',')[0]}: {len(v)}" for k, v in SETS.items()))
# basis from smoothed test vectors, as in svdbasis.jl: 8 random fields smoothed 4 times, SVD of the slices
rng = np.random.default_rng(1)
tv = []
for _ in range(8):
    z = rng.standard_normal(nb + 1); z[nb] = 0
    for _ in range(4):
        z = S1 @ z; z[nb] = 0; z /= np.linalg.norm(z)
    tv.append(z[:nb].reshape((Ns, 32)).T)
GS = PHI.T @ np.linalg.svd(np.hstack(tv), full_matrices=False)[0]
BASES = {"sines": np.eye(32), "SVD of the slow directions": PHI.T @ D[1]["G"], "smoothed test vectors": GS}
OUTS_S = {}
for name, idx in SETS.items():
    Vs = real_slices(idx)                                 # in odd-basis (half-grid) values
    for bname, Gb in BASES.items():
        Q = PHI @ Gb                                      # basis in half-grid values
        OUTS_S[(name, bname)] = [np.median(outside(Vs, [Q[:, :r]] * Ns)) for r in R]
    print(f"   {name.split(',')[0]} ({len(Vs)} real vectors), median fraction outside r = 8/12/16 modes: "
          + "; ".join(f"{b}: " + "/".join(f"{OUTS_S[(name, b)][R.index(r)]:.3f}" for r in (8, 12, 16)) for b in BASES))

# exact two-grid analysis at M = 1: E = S^2 (I - P K^-1 P^T J) S^2 with every slice (t_c = 1), the
# Galerkin K = P^T J P, P = r modes of a basis at every slice plus the period
S2 = S1 @ S1
TG = {}
PERSLICE = "SVD of the slow directions, per slice"
for bname, Gb in list(BASES.items()) + [(PERSLICE, None)]:
    for r in (6, 8, 10, 12, 14, 16):
        P = np.zeros((nb + 1, r * Ns + 1))
        for l in range(Ns):
            Q = d["U"][l] if Gb is None else PHI @ Gb
            P[32 * l:32 * (l + 1), r * l:r * (l + 1)] = Q[:, :r]
        P[nb, -1] = 1.0
        C = np.eye(nb + 1) - P @ np.linalg.solve(P.T @ Jb @ P, P.T @ Jb)
        ev = np.linalg.eigvals(S2 @ C @ S2)
        TG[(bname, r)] = ev
        print(f"   two-grid V(2,2) at M = 1, {bname}, r = {r}: spectral radius {np.abs(ev).max():.2f}, "
              f"eigenvalues with |λ| > 0.1: {np.sum(np.abs(ev) > 0.1)}, > 0.5: {np.sum(np.abs(ev) > 0.5)}")

# space-time density of the slow directions, M = 1
rho = np.sum(d["V"] ** 2, axis=0)                       # 32 × Ns

# ---------------------------------------------------------------- figures
T1 = 2 * np.pi / D[1]["om"]
t1 = np.arange(D[1]["Ns"] + 1) * T1 / D[1]["Ns"]
def spacetime(ax, F, T, title, sym=True, cmap="RdBu_r"):
    Ns = F.shape[1]; t = np.arange(Ns + 1) * T / Ns
    Fp = np.c_[F, F[:, :1]]
    v = np.abs(F).max()
    im = ax.pcolormesh(t, np.r_[x_full, L], np.r_[Fp, Fp[:1]], cmap=cmap, shading="gouraud",
                       vmin=-v if sym else 0, vmax=v)
    ax.set_title(title, fontsize=8.5); ax.set_xlabel("$t$")
    return im

def full_field(Fh):                                      # half-grid values (32 × Ns) to the full grid
    return np.vstack([np.zeros(Fh.shape[1]), Fh, -Fh[::-1]])

# 1. the orbit, the density of the slow directions, and the Floquet profiles
fig, ax = plt.subplots(2, 4, figsize=(8.2, 4.3), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.12, hspace=0.38))
ax = ax.ravel()
spacetime(ax[0], D[1]["A"], T1, "orbit $u(x, t)$")
spacetime(ax[1], np.vstack([np.zeros(rho.shape[1]), rho, rho[::-1]]), T1, "slow directions: $\\sum_d z_d^2$", sym=False, cmap="magma_r")
labels = []
for a, (mu, kind, q) in zip(ax[2:], FLQ[:6]):
    spacetime(a, full_field(q), T1, "neutral profile, $\\mu = 0$" if abs(mu) < 5e-4 else f"Floquet profile, $\\mu = {mu:+.3f}$")
for a in ax[::4]: a.set_ylabel("$x$")
for a in ax[:4]: a.set_xlabel("")
save(fig, "modes_floquet")

# 2. per-slice SVD modes at three times, against the orbit
times = [0, D[1]["Ns"] // 3, 2 * D[1]["Ns"] // 3]
fig, ax = plt.subplots(1, 3, figsize=(8.2, 3.9), sharey=True, gridspec_kw=dict(wspace=0.32))
for a, l in zip(ax, times):
    u = D[1]["U"][l]; s = D[1]["S"][l]
    a.plot(x_full, full(D[1]["A"][1:33, l]) / np.abs(D[1]["A"][:, l]).max() * 0.9 + 1.2 * 0, color="k", lw=1.2)
    a.text(L + 0.5, 0, "$u$", va="center", fontsize=8)
    for i in range(8):
        y0 = -1.6 * (i + 1)
        ui = u[:, i] * (1 if u[np.argmax(np.abs(u[:, i])), i] > 0 else -1)
        a.plot(x_full, y0 + full(ui) / np.abs(ui).max() * 0.7, color=BLUE if i < 4 else PURPLE, lw=1.0)
        a.text(L + 0.5, y0, f"{i + 1}: {s[i] / s[0]:.0e}".replace("e-0", "e-").replace("1e+00", "1"), va="center", fontsize=7)
    a.set_title(f"$t = {l * T1 / D[1]['Ns']:.1f}$", fontsize=8.5); a.set_xlabel("$x$"); a.set_yticks([])
    a.spines["left"].set_visible(False); a.set_xlim(0, L)
save(fig, "modes_perslice")

# 3. singular values, sine content of the modes, and the fraction outside each basis
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6), gridspec_kw=dict(wspace=0.62, width_ratios=[1, 1.15, 1.25]))
a = ax[0]
for M, c in ((1, BLUE), (8, RED)):
    s = D[M]["S"] / D[M]["S"][:, :1]
    a.semilogy(np.arange(1, 33), np.median(s, 0), "o-" if M == 1 else "-", color=c, ms=2.5, lw=1.1, label=f"$T = {2 * np.pi / D[M]['om']:.0f}$, per slice")
    a.fill_between(np.arange(1, 33), np.percentile(s, 10, 0), np.percentile(s, 90, 0), color=c, alpha=0.15, lw=0)
    a.semilogy(np.arange(1, 33), D[M]["sg"] / D[M]["sg"][0], "--", color=c, lw=0.9, label=f"$T = {2 * np.pi / D[M]['om']:.0f}$, one basis")
a.set_ylim(1e-5, 1.5); a.set_xlabel("mode $i$"); a.set_ylabel("$\\sigma_i / \\sigma_1$"); a.legend(frameon=False, fontsize=6.5)
a.set_title("singular values", fontsize=8.5)
a = ax[1]
C = D[8]["C"][:, :16]
im = a.imshow(C / C.max(0), origin="lower", aspect="auto", cmap="Blues", norm=LogNorm(1e-3, 1),
              extent=(0.5, 16.5, 0.5, 32.5))
a.set_xlabel("per-slice mode $i$"); a.set_ylabel("sine wavenumber $k$"); a.set_title("sine content of the modes", fontsize=8.5)
a.axhline(16.5, color=RED, lw=0.7, ls="--")
cb = fig.colorbar(im, ax=a, fraction=0.06, pad=0.03); cb.set_label("energy / largest", fontsize=7)
a = ax[2]
for (k, c) in (("per slice", BLUE), ("one basis", GREEN), ("one basis, other orbit", ORANGE), ("sines", GREY)):
    for M, ls in ((8, "-"), (1, ":")):
        y = [np.median(v) for v in OUTS[M][k]]
        a.semilogy(R, y, ls, color=c, lw=1.2, marker="o" if M == 8 else None, ms=2.5,
                   label=(k if k != "one basis, other orbit" else "one basis, from the other orbit") if M == 8 else None)
a.axhline(0.01, color=GREY, lw=0.5, ls=":")
a.set_xlabel("modes kept $r$"); a.set_ylabel("fraction outside (median)"); a.set_ylim(1e-4, 1)
a.set_title("slow directions not held", fontsize=8.5); a.legend(frameon=False, fontsize=6.5)
save(fig, "modes_content")

# 4. the global modes, against sines
fig, ax = plt.subplots(2, 4, figsize=(8.2, 3.3), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.35))
for i, a in enumerate(ax.ravel()):
    gi = D[8]["G"][:, i]; c = PHI.T @ gi; k = np.argmax(np.abs(c))
    a.plot(x_full, full(gi) / np.abs(gi).max(), color=BLUE, lw=1.2)
    a.plot(x_full, np.sign(c[k]) * np.sin(2 * np.pi * (k + 1) * x_full / L), color=GREY, lw=0.7, ls="--")
    a.set_title(f"mode {i + 1}: {100 * c[k] ** 2:.0f}% in $k = {k + 1}$", fontsize=8)
    a.set_yticks([])
for a in ax[1]: a.set_xlabel("$x$")
save(fig, "modes_global")
# 5. the spectrum of the smoothing step, and which basis holds what
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.7), gridspec_kw=dict(wspace=0.45, width_ratios=[1.05, 1, 1]))
a = ax[0]
th = np.linspace(0, 2 * np.pi, 200)
a.fill(1 + 0.6 * np.cos(th), 0.6 * np.sin(th), color=ORANGE, alpha=0.18, lw=0)
a.plot(np.cos(th), np.sin(th), color=GREY, lw=0.7, ls="--")
amp = np.abs(lamS) > 1
a.plot(lamS.real[~amp], lamS.imag[~amp], ".", color=BLUE, ms=1.5)
a.plot(lamS.real[amp], lamS.imag[amp], ".", color=RED, ms=2.5)
a.set_xlim(-3, 4); a.set_ylim(-3.5, 3.5); a.set_aspect("equal")
a.set_xlabel("Re $\\lambda$"); a.set_ylabel("Im $\\lambda$"); a.set_title("eigenvalues of $S = I - B^{-1}\\mathcal{J}$", fontsize=8.5)
a.annotate("slow for GMRES:\n$|1 - \\lambda| < 0.6$", xy=(1.3, 0.3), xytext=(1.2, 2.2), color=ORANGE, fontsize=7,
           arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
a.text(1.9, -2.9, "amplified: $|\\lambda| > 1$", color=RED, fontsize=7)
cols = {"sines": GREY, "SVD of the slow directions": GREEN, "smoothed test vectors": PURPLE, PERSLICE: BLUE}
rr = (6, 8, 10, 12, 14, 16)
for bname in list(BASES) + [PERSLICE]:
    ax[1].plot(rr, [np.sum(np.abs(TG[(bname, r)]) > 0.1) for r in rr], "o-", color=cols[bname], ms=2.5, lw=1.1, label=bname)
    ax[2].semilogy(rr, [np.abs(TG[(bname, r)]).max() for r in rr], "o-", color=cols[bname], ms=2.5, lw=1.1, label=bname)
ax[1].set_xlabel("modes kept $r$"); ax[1].set_ylabel("eigenvalues of $E$ with $|\\lambda| > 0.1$")
ax[1].set_title("two-grid V(2,2), $T = 25$", fontsize=8.5); ax[1].legend(frameon=False, fontsize=6.5, loc="upper right")
ax[2].axhline(1, color=GREY, lw=0.5, ls=":")
ax[2].set_xlabel("modes kept $r$"); ax[2].set_ylabel("largest $|\\lambda|$ of $E$"); ax[2].set_title("spectral radius of the cycle", fontsize=8.5)
save(fig, "modes_smoother")
print("figures written to", OUT)
