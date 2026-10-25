"""Figures of the chapter on the two-grid preconditioner.      python3 figs.py outdir [part …]

Parts: orbits, symbols, coarsebasis, timecontent, pod, gallery, spectra, smoother, coarse, kspec,
kmodes, exact, preconditioned, cycles, escape, bases, histories, podscan, timefourier, fouriertransfer,
fourierimpl."""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import kslib as K
from kslib import BLUE, RED, GREEN, GREY, ORANGE, PURPLE, L, NX, PHI, KAPPA

K.style(plt)
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
which = sys.argv[2:] or ["orbits", "symbols", "coarsebasis", "timecontent", "pod", "gallery", "spectra",
                         "smoother", "coarse", "kspec", "kmodes", "exact", "preconditioned", "cycles",
                         "escape", "bases", "histories", "podscan"]

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

def spacetime(ax, F, T, title=None, sym=True, cmap="RdBu_r", vmax=None):
    Ns = F.shape[1]; t = np.arange(Ns + 1) * T / Ns
    Fp = np.c_[F, F[:, :1]]
    v = vmax or np.abs(F).max()
    im = ax.pcolormesh(t, np.r_[K.X_FULL, L], np.r_[Fp, Fp[:1]], cmap=cmap, shading="gouraud",
                       vmin=-v if sym else 0, vmax=v, rasterized=True)
    if title: ax.set_title(title, fontsize=8.5)
    return im

def st_spectrum(A, om):
    """Energy |b_k(n)|^2 of a field on the grid, k = 1..32, n ordered by FFT; and physical frequencies."""
    B = K.sine_coeffs(A)
    Bh = np.fft.fft(B, axis=1) / B.shape[1]
    n = np.fft.fftfreq(B.shape[1], 1 / B.shape[1])
    return np.abs(Bh) ** 2, om * n

# ---------------------------------------------------------------- test orbits
if "orbits" in which:
    fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08, width_ratios=[1, 1.4, 2, 2.8]))
    for a, M in zip(ax, (1, 2, 4, 8)):
        A, om = K.orbit(M)
        im = spacetime(a, A, 2 * np.pi / om, f"$T = {2 * np.pi / om:.1f}$", vmax=3.2)
        a.set_xlabel("$t$")
    ax[0].set_ylabel("$x$")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="$u$")
    save(fig, "ch_orbits")

    # space-time spectrum of the orbits
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.6), gridspec_kw=dict(wspace=0.6, width_ratios=[1.3, 1, 1]))
    E1, w1 = st_spectrum(*K.orbit(1))
    order = np.argsort(w1)
    im = ax[0].pcolormesh(w1[order], np.arange(1, 33), E1[:, order], norm=LogNorm(1e-8, E1.max()), cmap="Blues", shading="nearest", rasterized=True)
    ax[0].set_xlim(-6, 6); ax[0].set_xlabel("physical frequency $\\omega n$"); ax[0].set_ylabel("sine mode $k$")
    ax[0].set_title("$|\\hat u(k, n)|^2$, $T = 25$", fontsize=8.5)
    fig.colorbar(im, ax=ax[0], fraction=0.05, pad=0.02)
    for M, c in ((1, BLUE), (8, RED)):
        E, w = st_spectrum(*K.orbit(M))
        ax[1].semilogy(np.arange(1, 33), E.sum(1) / E.sum(), "o-" if M == 1 else "-", color=c, ms=2.5, lw=1.1, label=f"$T = {2 * np.pi / K.orbit(M)[1]:.0f}$")
        ew = E.sum(0); bins = np.linspace(0, 8, 33); h = np.histogram(np.abs(w), bins, weights=ew)[0]
        ax[2].semilogy(0.5 * (bins[1:] + bins[:-1]), h / ew.sum(), "o-" if M == 1 else "-", color=c, ms=2.5, lw=1.1)
    ax[1].set_xlabel("sine mode $k$"); ax[1].set_ylabel("energy fraction"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("in space", fontsize=8.5); ax[1].set_ylim(1e-12, 1)
    ax[2].set_xlabel("$|\\omega n|$"); ax[2].set_title("in time", fontsize=8.5); ax[2].set_ylim(1e-10, 1)
    save(fig, "ch_orbit_spectra")

# ---------------------------------------------------------------- symbols of the operators
if "symbols" in which:
    A, om = K.orbit(1)
    n = np.arange(0, 61); W = om * n
    Bhat = np.abs(1j * W[None, :] + (KAPPA ** 4 - KAPPA ** 2)[:, None])
    amax = np.abs(A).max(); arms = np.sqrt(np.mean(A ** 2))
    est = KAPPA[:, None] * arms / Bhat
    sm = np.loadtxt(os.path.join(K.RES, "smoothing_map_M1.txt"))
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    im = ax[0].pcolormesh(W, np.arange(1, 33), Bhat, norm=LogNorm(1e-2, 1e6), cmap="viridis", shading="nearest", rasterized=True)
    ax[0].set_title("$|\\hat B(k, n)| = |i\\omega n + \\kappa^4 - \\kappa^2|$", fontsize=8.5)
    fig.colorbar(im, ax=ax[0], fraction=0.05, pad=0.02)
    im = ax[1].pcolormesh(W, np.arange(1, 33), est, norm=LogNorm(1e-3, 1e2), cmap="magma_r", shading="nearest", rasterized=True)
    ax[1].contour(W, np.arange(1, 33), est, [0.5, 1], colors=[ORANGE, RED], linewidths=0.9)
    ax[1].set_title("estimate $\\kappa\\, u_{\\mathrm{rms}} / |\\hat B|$", fontsize=8.5)
    fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.02)
    wn, amp = sm[0], sm[1:]
    order = np.argsort(wn)
    im = ax[2].pcolormesh(wn[order], np.arange(1, amp.shape[0] + 1), amp[:, order], norm=LogNorm(1e-3, 1e2), cmap="magma_r", shading="nearest", rasterized=True)
    ax[2].contour(wn[order], np.arange(1, amp.shape[0] + 1), amp[:, order], [0.5, 1], colors=[ORANGE, RED], linewidths=0.9)
    ax[2].set_title("measured $\\|S v\\|/\\|v\\|$", fontsize=8.5)
    fig.colorbar(im, ax=ax[2], fraction=0.05, pad=0.02)
    for a in ax: a.set_xlim(0, 6); a.set_xlabel("physical frequency $\\omega n$")
    ax[0].set_ylabel("sine mode $k$")
    save(fig, "ch_symbols")

# ---------------------------------------------------------------- coarse basis functions
if "coarsebasis" in which:
    A, om = K.orbit(1); Ns = A.shape[1]; T = 2 * np.pi / om
    def bsp(u):
        v = np.abs(u); return np.where(v < 1, (4 - 6 * v ** 2 + 3 * v ** 3) / 6, np.where(v < 2, (2 - v) ** 3 / 6, 0))
    def dbsp(u):
        v = np.abs(u); return np.where(v < 1, -2 * u + 1.5 * u * v, np.where(v < 2, -np.sign(u) * (2 - v) ** 2 / 2, 0))
    fig = plt.figure(figsize=(8.4, 4.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)
    s_fine = 2 * np.pi * np.arange(Ns) / Ns; ss = np.linspace(0, 2 * np.pi, 2000)
    for j, tc in enumerate((2, 4)):
        a = fig.add_subplot(gs[0, j])
        Nc = round(Ns / tc); D = 2 * np.pi / Nc
        for m in range(4, 9):
            c = plt.cm.viridis((m - 4) / 5)
            u = (ss - D * m) / D; a.plot(ss * T / (2 * np.pi), bsp(u), color=c, lw=1.0)
            uf = (s_fine - D * m) / D; a.plot(s_fine * T / (2 * np.pi), bsp(uf), "o", color=c, ms=2.2)
        a.set_xlim(2.5 * tc * T / Ns, 10.5 * tc * T / Ns); a.set_xlabel("$t$")
        a.set_title(f"B-splines $\\phi_m$, $t_c = {tc}$ (dots: slices)", fontsize=8.5)
    a = fig.add_subplot(gs[0, 2])
    Nc = round(Ns / 2); D = 2 * np.pi / Nc
    Pt = np.array([[bsp(((s - D * m + np.pi) % (2 * np.pi) - np.pi) / D) for m in range(Nc)] for s in s_fine])
    a.imshow(Pt, aspect="auto", cmap="Greys", interpolation="nearest")
    a.set_xlabel("coarse time point $m$"); a.set_ylabel("slice $l$"); a.set_title("$P_t$: $\\phi_m(s_l)$, $t_c = 2$", fontsize=8.5)
    # a prolonged coarse unknown: sine k = 3 times a B-spline
    a = fig.add_subplot(gs[1, 0])
    m = 12; phi = np.array([bsp(((s - D * m + np.pi) % (2 * np.pi) - np.pi) / D) for s in s_fine])
    F = np.outer(np.sin(2 * np.pi * 3 * K.X_FULL / L), phi)
    spacetime(a, F, T, "$P e$: $\\sin(\\kappa_3 x)\\,\\phi_{12}(t)$"); a.set_xlabel("$t$"); a.set_ylabel("$x$")
    # the neutral profile and its part outside the first 16 sines
    B = K.sine_coeffs(A)
    dB = np.real(np.fft.ifft(1j * np.fft.fftfreq(Ns, 1 / Ns) * np.fft.fft(B, axis=1), axis=1)) * om   # ∂t
    S = np.sin(2 * np.pi * np.outer(K.X_FULL, np.arange(1, 33)) / L)
    a = fig.add_subplot(gs[1, 1]); spacetime(a, S @ dB, T, "neutral profile $\\partial_t u$"); a.set_xlabel("$t$")
    dB16 = dB.copy(); dB16[:16] = 0
    a = fig.add_subplot(gs[1, 2]); spacetime(a, S @ dB16, T, f"its part outside $k \\leq 16$ (×{np.abs(S @ dB).max() / np.abs(S @ dB16).max():.0f})")
    a.set_xlabel("$t$")
    save(fig, "ch_coarse_basis")

# ---------------------------------------------------------------- temporal content of the slow directions
if "timecontent" in which:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.5), gridspec_kw=dict(wspace=0.35))
    for M, c in ((1, BLUE), (8, RED)):
        V, A, om = K.slow_directions(M); Ns = A.shape[1]
        Vh = np.fft.fft(np.einsum("jk,djl->dkl", PHI, V), axis=2)
        w = np.abs(om * np.fft.fftfreq(Ns, 1 / Ns))
        e = np.sum(np.abs(Vh) ** 2, axis=(0, 1)); o = np.argsort(w)
        ax[0].plot(w[o], np.cumsum(e[o]) / e.sum(), "-", color=c, lw=1.3, label=f"$T = {2 * np.pi / om:.0f}$")
        ek = np.sum(np.abs(Vh) ** 2, axis=(0, 2))
        ax[1].semilogy(np.arange(1, 33), ek / ek.sum(), "o-" if M == 1 else "-", color=c, ms=2.5, lw=1.1)
    dt = 2 * np.pi / K.orbit(1)[1] / 73
    for tc, ls in ((2, "--"), (4, "-."), (8, ":")):
        wc = 2 * np.pi / (4 * tc * dt)
        ax[0].axvline(wc, color=GREY, ls=ls, lw=0.8); ax[0].text(wc + 0.05, 0.05, f"$t_c = {tc}$", fontsize=7, color=GREY, rotation=90)
    ax[0].set_xlim(0, 4); ax[0].set_xlabel("$|\\omega n|$"); ax[0].set_ylabel("cumulative energy"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("slow directions: in time", fontsize=8.5)
    ax[1].set_xlabel("sine mode $k$"); ax[1].set_ylabel("energy fraction"); ax[1].set_title("in space", fontsize=8.5)
    ax[1].axvline(16.5, color=GREY, ls=":", lw=0.8)
    save(fig, "ch_slow_content")

# ---------------------------------------------------------------- POD modes
if "pod" in which:
    U = np.loadtxt(os.path.join(K.HERE, "pod", "pod_attractor.txt"))
    en = np.loadtxt(os.path.join(K.HERE, "pod", "energy_attractor.txt"))
    Sn = np.load(os.path.join(K.HERE, "pod", "dns_snapshots.npy"))
    A, om = K.orbit(1); Bo = K.sine_coeffs(A)
    Uo, so, _ = np.linalg.svd(Bo, full_matrices=False)
    fig = plt.figure(figsize=(8.4, 5.6))
    gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.3)
    a = fig.add_subplot(gs[0, :2])
    tt = np.arange(Sn.shape[1])[:400] * 0.5
    Sf = np.sin(2 * np.pi * np.outer(K.X_FULL, np.arange(1, 33)) / L) @ Sn[:, :400]
    spacetime(a, Sf, tt[-1] + 0.5, "chaotic trajectory (first 200 of 40 000 time units)"); a.set_xlabel("$t$"); a.set_ylabel("$x$")
    a = fig.add_subplot(gs[0, 2:])
    es = np.mean(Sn ** 2, 1); es = np.sort(es)[::-1] / es.sum()
    a.semilogy(np.arange(1, 33), en, "o-", color=BLUE, ms=2.5, lw=1.1, label="POD modes, attractor")
    a.semilogy(np.arange(1, 33), so ** 2 / np.sum(so ** 2), "s-", color=GREEN, ms=2.5, lw=1.0, mfc="none", label="POD modes, orbit $T = 25$")
    a.semilogy(np.arange(1, 33), es, "-", color=GREY, lw=1.0, label="sine modes, sorted by energy")
    a.set_ylim(1e-14, 1); a.set_xlabel("mode"); a.set_ylabel("energy fraction"); a.legend(frameon=False, fontsize=7, loc="upper right")
    a.set_title("energy of the modes", fontsize=8.5)
    for i in range(8):
        a = fig.add_subplot(gs[1 + i // 4, i % 4])
        u = U[:, i]; kk = np.argmax(np.abs(u))
        prof = np.sin(2 * np.pi * np.outer(K.X_FULL, np.arange(1, 33)) / L) @ u
        a.plot(K.X_FULL, prof / np.abs(prof).max(), color=BLUE, lw=1.2)
        a.plot(K.X_FULL, np.sign(u[kk]) * np.sin(2 * np.pi * (kk + 1) * K.X_FULL / L), "--", color=GREY, lw=0.7)
        a.set_title(f"POD {i + 1} ({100 * en[i]:.2g}%): {100 * u[kk] ** 2:.0f}% $k = {kk + 1}$", fontsize=7.5)
        a.set_yticks([]); a.set_xlabel("$x$") if i >= 4 else None
    save(fig, "ch_pod_modes")

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.3))
    for a, Ub, t in ((ax[0], U, "POD of the attractor"), (ax[1], Uo, "POD of the orbit $T = 25$")):
        im = a.imshow(Ub[:, :20] ** 2, origin="lower", aspect="auto", cmap="Blues", norm=LogNorm(1e-4, 1), extent=(0.5, 20.5, 0.5, 32.5))
        a.set_xlabel("POD mode"); a.set_ylabel("sine wavenumber $k$"); a.set_title(t, fontsize=8.5)
        a.axhline(16.5, color=RED, lw=0.6, ls="--")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="energy in $k$")
    save(fig, "ch_pod_content")

# ================================================================ dense analysis at T ≈ 25
NS1 = 73; N1 = 32 * NS1 + 1
def dims(cfg):
    return [int(v) for v in np.loadtxt(os.path.join(K.DENSE, f"M1_dims_{cfg}.txt"))]
def eigE(cfg, case="M1"):
    return K.read_eig(f"{case}_eig_{cfg}.txt")

# ---------------------------------------------------------------- structure of the operators
if "gallery" in which:
    J = K.read_dense("M1_J.bin", N1); Bi = K.read_dense("M1_Binv.bin", N1)
    Minv = K.read_dense("M1_Minv_sine_r16_tc2_V22.bin", N1)
    Jh = K.op_fourier(J, NS1)
    Bh = np.diag(np.diag(Jh))                                   # B is diagonal in this basis
    Bih = K.op_fourier(Bi, NS1)
    Ah = np.diag(1 / np.diag(Bih)) - Jh                          # A = B - J
    Sh = Bih @ Ah
    Eh = K.op_fourier(np.eye(N1) - Minv @ J, NS1)
    print("check: B^-1 diagonal in Fourier:", np.abs(Bih - np.diag(np.diag(Bih))).max() / np.abs(Bih).max())
    def blocknorms(Mh):
        return np.array([[np.linalg.norm(Mh[k * NS1:(k + 1) * NS1, j * NS1:(j + 1) * NS1], 2) for j in range(32)] for k in range(32)])
    nord, _ = K.n_order(NS1); nat = np.argsort(nord)                 # n = -36 … 36
    def block(Mh, k, j):
        return Mh[k * NS1:(k + 1) * NS1, j * NS1:(j + 1) * NS1][np.ix_(nat, nat)]
    fig, ax = plt.subplots(2, 3, figsize=(8.4, 5.6), gridspec_kw=dict(wspace=0.35, hspace=0.38))
    ext = (0.5, 32.5, 32.5, 0.5); extn = (-36.5, 36.5, 36.5, -36.5)
    panels = [(blocknorms(Jh), "$\\|\\hat{\\mathcal{J}}_{kk'}\\|$", ext, LogNorm(1e-3, 1e3)),
              (blocknorms(Ah), "$\\|\\hat A_{kk'}\\|$: advection", ext, LogNorm(1e-3, 1e3)),
              (blocknorms(Sh), "$\\|\\hat S_{kk'}\\|$: one smoothing step", ext, LogNorm(1e-4, 1e2)),
              (np.abs(block(Jh, 3, 3)), "$|\\hat{\\mathcal{J}}_{(4,n),(4,n')}|$", extn, LogNorm(1e-3, 1e1)),
              (np.abs(block(Sh, 3, 3)), "$|\\hat S_{(4,n),(4,n')}|$", extn, LogNorm(1e-4, 1e2)),
              (blocknorms(Eh), "$\\|\\hat E_{kk'}\\|$: cycle V(2,2), $r = 16$, $t_c = 2$", ext, LogNorm(1e-4, 1e2))]
    for a, (Mx, t, e, nm) in zip(ax.ravel(), panels):
        im = a.imshow(Mx, norm=nm, cmap="magma_r", interpolation="nearest", extent=e)
        a.set_title(t, fontsize=8.5)
        if e == ext:
            a.set_xlabel("$k'$"); a.set_ylabel("$k$")
            a.axhline(16.5, color=GREEN, lw=0.6, ls="--"); a.axvline(16.5, color=GREEN, lw=0.6, ls="--")
        else:
            a.set_xlabel("$n'$"); a.set_ylabel("$n$")
        fig.colorbar(im, ax=a, fraction=0.045, pad=0.03)
    save(fig, "ch_gallery")

# ---------------------------------------------------------------- spectra
if "spectra" in which:
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw=dict(wspace=0.35))
    A1, om1 = K.orbit(1)
    for case, c, ms in (("M1", BLUE, 2.5), ("Q2", RED, 1.2)):
        lj = K.read_eig(f"{case}_eigJ.txt"); lb = K.read_eig(f"{case}_eigJBinv.txt")
        lab = "$T = 25$" if case == "M1" else "$T = 51$"
        ax[0].plot(lj.real, lj.imag, ".", color=c, ms=ms, label=lab)
        ax[1].plot(lj.real, lj.imag, ".", color=c, ms=ms)
        ax[2].plot(lb.real, lb.imag, ".", color=c, ms=ms, label=lab)
    kk = KAPPA[:6]; nn = np.arange(-40, 41)
    for kv in kk:
        ax[1].plot(np.full(nn.size, kv ** 4 - kv ** 2), om1 * nn, "x", color=GREY, ms=2.2, mew=0.5)
    ax[0].set_xlim(-2, 60); ax[0].set_ylim(-12, 12); ax[0].set_title("eigenvalues of $\\mathcal{J}$", fontsize=8.5)
    ax[1].set_xlim(-0.9, 1.2); ax[1].set_ylim(-3, 3); ax[1].set_title("near the origin (grey ×: $\\hat B$, $k \\leq 6$)", fontsize=8.5)
    ax[2].set_xlim(-0.5, 2.5); ax[2].set_ylim(-1.6, 1.6); ax[2].set_title("eigenvalues of $\\mathcal{J}B^{-1}$", fontsize=8.5)
    ax[2].add_patch(plt.Circle((0, 0), 0.6, color=ORANGE, alpha=0.15, lw=0)); ax[2].set_aspect("equal")
    ax[0].legend(frameon=False, fontsize=7, markerscale=3)
    for a in ax: a.set_xlabel("Re $\\lambda$"); a.axvline(0, color=GREY, lw=0.4); a.axhline(0, color=GREY, lw=0.4)
    ax[0].set_ylabel("Im $\\lambda$")
    save(fig, "ch_spectra")
    for case in ("M1", "Q2"):
        lb = K.read_eig(f"{case}_eigJBinv.txt")
        print(case, "eigenvalues of JB^-1 with |λ| < 0.6:", np.sum(np.abs(lb) < 0.6), " |λ - 1| > 0.5:", np.sum(np.abs(lb - 1) > 0.5), "of", lb.size)

# ---------------------------------------------------------------- the smoother: gains and modes
if "smoother" in which:
    R = np.load(os.path.join(K.HERE, "resolvent.npz"))
    sig = R["sig"]
    A1, om1 = K.orbit(1); T1 = 2 * np.pi / om1
    fig = plt.figure(figsize=(8.4, 5.2))
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.35)
    a = fig.add_subplot(gs[0, :2])
    a.semilogy(np.arange(1, sig.size + 1), sig, color=BLUE, lw=1.3)
    a.axhline(1, color=GREY, lw=0.5, ls=":"); a.axhline(0.5, color=GREY, lw=0.5, ls=":")
    a.set_xscale("log"); a.set_xlabel("index $i$"); a.set_ylabel("gain $\\sigma_i(S)$"); a.set_title("singular values of one smoothing step", fontsize=8.5)
    a.text(60, 1.3, f"{np.sum(sig > 1)} gains > 1", fontsize=7.5, color=GREY); a.text(150, 0.6, f"{np.sum(sig > 0.5)} > 0.5", fontsize=7.5, color=GREY)
    # (k, ωn) content of the leading response modes
    a = fig.add_subplot(gs[0, 2:])
    U = R["Ulead"]
    Uh = K.to_fourier(U, NS1)
    E = np.sum(np.abs(Uh) ** 2 * sig[:12][None, :] ** 2, axis=1).reshape((32, NS1))
    nord, _ = K.n_order(NS1); o = np.argsort(nord)
    im = a.pcolormesh(om1 * nord[o], np.arange(1, 33), E[:, o], norm=LogNorm(1e-4 * E.max(), E.max()), cmap="Blues", shading="nearest", rasterized=True)
    a.set_xlim(-6, 6); a.set_xlabel("$\\omega n$"); a.set_ylabel("sine mode $k$"); a.set_title("12 leading response modes: $(k, \\omega n)$ content", fontsize=8.5)
    for i in range(4):
        a = fig.add_subplot(gs[1, i])
        u = K.full_field(U[:32 * NS1, i].reshape((NS1, 32)).T)
        spacetime(a, u, T1, f"response mode {i + 1}, $\\sigma = {sig[i]:.1f}$"); a.set_xlabel("$t$")
        if i == 0: a.set_ylabel("$x$")
    save(fig, "ch_smoother_svd")

def coarse_data(cfg):
    n, nc, rc, Nb = dims(cfg)
    Kt = K.read_dense(f"M1_K_{cfg}.bin", nc); G = K.read_dense(f"M1_G_{cfg}.bin", nc)
    P = np.fromfile(os.path.join(K.DENSE, f"M1_P_{cfg}.bin")).reshape((nc, n)).T
    c = np.sum(Kt * G) / np.sum(G * G)                       # K = c PᵀJP up to the time derivative
    return Kt, G, P, c, rc, Nb

# ---------------------------------------------------------------- structure of the coarse matrix
if "coarse" in which:
    fig = plt.figure(figsize=(8.4, 5.4))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.45)
    for j, (cfg, t) in enumerate((("sine_r16_tc1_V11", "every slice, FD8"), ("sine_r16_tc2_V22", "B-splines, $t_c = 2$"))):
        Kt, G, P, c, rc, Nb = coarse_data(cfg)
        a = fig.add_subplot(gs[0, 2 * j])
        a.spy(np.abs(Kt) > 1e-9 * np.abs(Kt).max(), markersize=0.15, color="k", rasterized=True)
        a.set_title(f"$K$, {t}: {Kt.shape[0]} unknowns", fontsize=8); a.tick_params(labelsize=7)
        a = fig.add_subplot(gs[0, 2 * j + 1])
        z = np.abs(Kt[:6 * rc, :6 * rc]); z = np.r_[z]
        im = a.imshow(z, norm=LogNorm(1e-4 * z.max(), z.max()), cmap="magma_r", interpolation="nearest")
        for b in range(1, 6): a.axhline(b * rc - 0.5, color=GREY, lw=0.3); a.axvline(b * rc - 0.5, color=GREY, lw=0.3)
        a.set_title(f"first 6 block rows ({rc} × {rc} blocks)", fontsize=8); a.tick_params(labelsize=7)
        print(cfg, f"K = c PᵀJP with c = {c:.4g}; relative difference {np.linalg.norm(Kt - c * G) / np.linalg.norm(Kt):.2e}")
    # the spatial blocks A_l at three slices, and their norm along the orbit
    Kt, G, P, c, rc, Nb = coarse_data("sine_r16_tc1_V11")
    A1, om1 = K.orbit(1); T1 = 2 * np.pi / om1
    for i, l in enumerate((0, 24, 48)):
        a = fig.add_subplot(gs[1, i])
        Bl = Kt[l * rc:(l + 1) * rc, l * rc:(l + 1) * rc][:16, :16] / c
        Bl = Bl - np.diag(np.diag(Bl))                      # the advection: the diagonal is κ⁴ - κ² plus a little
        v = np.abs(Bl).max()
        im = a.imshow(Bl, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest", extent=(0.5, 16.5, 16.5, 0.5))
        a.set_title(f"advection in the block, $t = {l * T1 / 73:.1f}$", fontsize=8); a.set_xlabel("$k'$"); a.set_ylabel("$k$") if i == 0 else None
        a.tick_params(labelsize=7)
    a = fig.add_subplot(gs[1, 3])
    Ln = np.diag(KAPPA[:16] ** 4 - KAPPA[:16] ** 2) * 1
    off = [np.linalg.norm(Kt[l * rc:(l + 1) * rc, l * rc:(l + 1) * rc][:16, :16] - np.diag(np.diag(Kt[l * rc:(l + 1) * rc, l * rc:(l + 1) * rc][:16, :16])), 2) for l in range(73)]
    a.plot(np.arange(73) * T1 / 73, off / np.max(off), color=BLUE, lw=1.2, label="advection in the block")
    a.plot(np.arange(73) * T1 / 73, np.sqrt(np.mean(A1 ** 2, 0)) / np.sqrt(np.mean(A1 ** 2, 0)).max(), color=GREY, lw=1.0, ls="--", label="$u_{\\mathrm{rms}}(t)$")
    a.set_xlabel("$t$"); a.set_title("along the orbit (normalised)", fontsize=8); a.legend(frameon=False, fontsize=6.5); a.set_ylim(0, 1.1)
    save(fig, "ch_coarse_structure")

# ---------------------------------------------------------------- spectrum of the coarse operator
if "kspec" in which:
    from scipy.linalg import eig as geig
    lj = K.read_eig("M1_eigJ.txt")
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True, gridspec_kw=dict(wspace=0.1))
    for a, (cfg, t) in zip(ax, (("sine_r16_tc1_V11", "every slice, FD8"), ("sine_r16_tc2_V22", "B-splines, $t_c = 2$"), ("sine_r16_tc4_V22", "B-splines, $t_c = 4$"))):
        Kt, G, P, c, rc, Nb = coarse_data(cfg)
        Mc = c * (P.T @ P)
        lk = geig(Kt, Mc, right=False); lg = geig(G, P.T @ P, right=False)
        a.plot(lj.real, lj.imag, "o", color=GREY, ms=3.5, mfc="none", mew=0.6, label="$\\mathcal{J}$")
        a.plot(lg.real, lg.imag, "x", color=ORANGE, ms=3, mew=0.7, label="exact Galerkin $P^{T}\\mathcal{J}P$")
        a.plot(lk.real, lk.imag, ".", color=BLUE, ms=3, label="$K$ of TwoGrid")
        a.set_xlim(-0.8, 1.2); a.set_ylim(-3.2, 3.2); a.set_title(t, fontsize=8.5); a.set_xlabel("Re $\\lambda$")
        a.axvline(0, color=GREY, lw=0.4)
    ax[0].set_ylabel("Im $\\lambda$"); ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    save(fig, "ch_coarse_spectrum")

# ---------------------------------------------------------------- slow modes of the coarse operator
if "kmodes" in which:
    from scipy.linalg import eig as geig
    J = K.read_dense("M1_J.bin", N1)
    lj, Vj = np.linalg.eig(J)
    Kt, G, P, c, rc, Nb = coarse_data("sine_r16_tc2_V22")
    lk, Vk = geig(Kt, c * (P.T @ P))
    A1, om1 = K.orbit(1); T1 = 2 * np.pi / om1
    fig, ax = plt.subplots(2, 4, figsize=(8.4, 4.4), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.35))
    targets = sorted(lk[np.abs(lk.imag) < 1e-8], key=abs)[:4]
    for i, lt in enumerate(targets):
        ik = np.argmin(np.abs(lk - lt)); ij = np.argmin(np.abs(lj - lt))
        for row, (v, lab, lam) in enumerate(((P @ Vk[:, ik], "coarse", lk[ik]), (Vj[:, ij], "fine", lj[ij]))):
            f = np.real(v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v[:-1]))])))
            spacetime(ax[row, i], K.full_field(f[:32 * NS1].reshape((NS1, 32)).T), T1,
                      f"{lab}: $\\lambda = {lam.real:+.3f}$")
    for a in ax[1]: a.set_xlabel("$t$")
    for a in ax[:, 0]: a.set_ylabel("$x$")
    save(fig, "ch_coarse_modes")

# ---------------------------------------------------------------- exact two-grid analysis: r and t_c
if "exact" in which:
    rs = (4, 6, 8, 10, 12, 14, 16, 20)
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.7), gridspec_kw=dict(wspace=0.4))
    cols = {1: BLUE, 2: PURPLE, 4: ORANGE, 8: RED}
    for tc in (1, 2, 4, 8):
        for (pre, ls) in ((1, "--"), (2, "-")):
            rho = [np.abs(eigE(f"sine_r{r}_tc{tc}_V{pre}{pre}")).max() for r in rs]
            cnt = [np.sum(np.abs(eigE(f"sine_r{r}_tc{tc}_V{pre}{pre}")) > 0.1) for r in rs]
            ax[0].semilogy(rs, rho, ls, color=cols[tc], marker="o", ms=2.5, lw=1.1, label=f"$t_c = {tc}$" if pre == 2 else None)
            ax[1].plot(rs, cnt, ls, color=cols[tc], marker="o", ms=2.5, lw=1.1)
    ax[0].axhline(1, color=GREY, lw=0.5, ls=":"); ax[0].set_ylim(1e-3, 1e6)
    ax[0].set_xlabel("sine modes $r$"); ax[0].set_ylabel("spectral radius of $E$"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("solid V(2,2), dashed V(1,1)", fontsize=8.5)
    ax[1].set_xlabel("sine modes $r$"); ax[1].set_ylabel("eigenvalues of $E$ with $|\\lambda| > 0.1$")
    # eigenvalues of E for r = 8, 12, 16, t_c = 2, V(2,2)
    th = np.linspace(0, 2 * np.pi, 200); a = ax[2]
    a.plot(np.cos(th), np.sin(th), color=GREY, lw=0.6, ls="--")
    for r, c, mk in ((8, RED, "o"), (12, ORANGE, "s"), (16, BLUE, "^")):
        l = eigE(f"sine_r{r}_tc2_V22"); l = l[np.abs(l) > 0.02]
        a.plot(l.real, l.imag, mk, color=c, ms=3, mfc="none", mew=0.8, label=f"$r = {r}$")
    a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.set_aspect("equal"); a.legend(frameon=False, fontsize=7)
    a.set_title("eigenvalues of $E$, $t_c = 2$, V(2,2)", fontsize=8.5); a.set_xlabel("Re $\\lambda$")
    save(fig, "ch_exact_twogrid")

# ---------------------------------------------------------------- preconditioned spectra
if "preconditioned" in which:
    fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.5), sharey=True, gridspec_kw=dict(wspace=0.08))
    for case, c, ms in (("M1", BLUE, 3), ("Q2", RED, 1.6)):
        lb = K.read_eig(f"{case}_eigJBinv.txt")
        ax[0].plot(lb.real, lb.imag, ".", color=c, ms=ms)
        for a, r in zip(ax[1:], (8, 12, 16)):
            l = 1 - eigE(f"sine_r{r}_tc2_V22", case)
            a.plot(l.real, l.imag, ".", color=c, ms=ms, label=("$T = 25$" if case == "M1" else "$T = 51$"))
            a.set_title(f"$\\mathcal{{J}}M^{{-1}}$, $r = {r}$, $t_c = 2$, V(2,2)", fontsize=8)
    ax[0].set_title("$\\mathcal{J}B^{-1}$", fontsize=8.5)
    for a in ax:
        a.set_xlim(-1, 3); a.set_ylim(-2, 2); a.axvline(0, color=GREY, lw=0.4); a.set_xlabel("Re $\\lambda$")
        a.plot([1], [0], "+", color="k", ms=6)
    ax[0].set_ylabel("Im $\\lambda$"); ax[3].legend(frameon=False, fontsize=7, markerscale=2)
    save(fig, "ch_preconditioned")

# ---------------------------------------------------------------- cycles
if "cycles" in which:
    cfgs = [("sine_r16_tc1_additive", "additive*"), ("sine_r16_tc2_V10", "V(1,0)"), ("sine_r16_tc2_V01", "V(0,1)"),
            ("sine_r16_tc2_V11", "V(1,1)"), ("sine_r16_tc2_V22", "V(2,2)"), ("sine_r16_tc1_V33", "V(3,3)*")]
    fig, ax = plt.subplots(1, 6, figsize=(8.4, 1.9), sharey=True, gridspec_kw=dict(wspace=0.06))
    th = np.linspace(0, 2 * np.pi, 200)
    for a, (cfg, t) in zip(ax, cfgs):
        l = eigE(cfg); big = np.abs(l) > 0.02
        a.plot(np.cos(th), np.sin(th), color=GREY, lw=0.6, ls="--")
        a.plot(l.real[big], l.imag[big], ".", color=BLUE, ms=2.5)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect("equal")
        a.set_title(f"{t}: $\\rho = {np.abs(l).max():.2g}$", fontsize=7.5); a.set_xticks([-1, 0, 1])
    ax[0].set_ylabel("Im $\\lambda(E)$")
    save(fig, "ch_cycles_spectra")
    # time derivative of the coarse operator, t_c = 1, V(1,1)
    for h in (1, 2, 4, 8):
        print(f"FD order {2 * h}: ρ(E) = {np.abs(eigE(f'sine_r16_tc1_V11_h{h}')).max():.3g}, |λ| > 0.1: {np.sum(np.abs(eigE(f'sine_r16_tc1_V11_h{h}')) > 0.1)}")
    for r in (12, 16):
        print(f"no neutral profile, r = {r}: ρ(E) = {np.abs(eigE(f'sine_r{r}_tc2_V22_noneutral')).max():.3g}; with: {np.abs(eigE(f'sine_r{r}_tc2_V22')).max():.3g}")

# ---------------------------------------------------------------- what escapes an under-resolved coarse space
if "escape" in which:
    J = K.read_dense("M1_J.bin", N1); Minv = K.read_dense("M1_Minv_sine_r8_tc2_V22.bin", N1)
    l, V = np.linalg.eig(np.eye(N1) - Minv @ J)
    o = np.argsort(-np.abs(l)); A1, om1 = K.orbit(1); T1 = 2 * np.pi / om1
    nord, _ = K.n_order(NS1); oo = np.argsort(nord)
    fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.4), gridspec_kw=dict(wspace=0.35, width_ratios=[1.3, 1, 1, 1]))
    sel = o[:np.sum(np.abs(l) > 0.5)]
    Vh = K.to_fourier(V[:, sel], NS1)
    E = np.sum(np.abs(Vh) ** 2, axis=1).reshape((32, NS1))
    im = ax[0].pcolormesh(om1 * nord[oo], np.arange(1, 33), E[:, oo], norm=LogNorm(1e-4 * E.max(), E.max()), cmap="Blues", shading="nearest", rasterized=True)
    ax[0].axhline(8.5, color=RED, lw=0.7, ls="--"); ax[0].set_xlim(-4, 4)
    ax[0].set_xlabel("$\\omega n$"); ax[0].set_ylabel("sine mode $k$"); ax[0].set_title(f"{len(sel)} eigenvectors of $E$ with $|\\lambda| > 0.5$", fontsize=8)
    for a, i in zip(ax[1:], o[:3]):
        f = np.real(V[:, i] * np.exp(-1j * np.angle(V[np.argmax(np.abs(V[:-1, i])), i])))
        spacetime(a, K.full_field(f[:32 * NS1].reshape((NS1, 32)).T), T1, f"$|\\lambda| = {abs(l[i]):.1f}$"); a.set_xlabel("$t$")
    save(fig, "ch_escape")

# ---------------------------------------------------------------- spatial bases in the exact analysis
if "bases" in which:
    names = [("sine", "sine modes", GREY, "o"), ("podA", "POD, attractor", BLUE, "s"), ("podO", "POD, orbit", GREEN, "D"),
             ("svd", "SVD of slow directions", ORANGE, "v"), ("test", "smoothed test vectors", PURPLE, "^"),
             ("resS", "resolvent of $S$", RED, "P"), ("resF", "frozen-time resolvent", "#c07090", "X"),
             ("resM", "mean-flow resolvent", "#2a9d8f", "*")]
    rs = (4, 6, 8, 10, 12, 14, 16, 20)
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 2.8), gridspec_kw=dict(wspace=0.3))
    for b, lab, c, mk in names:
        try:
            ls = [eigE(f"{b}_r{r}_tc2_V22") for r in rs]
        except FileNotFoundError:
            continue
        ax[0].semilogy(rs, [np.abs(l).max() for l in ls], "-", marker=mk, color=c, ms=3.5, lw=1.1, mfc="none", label=lab)
        ax[1].plot(rs, [np.sum(np.abs(l) > 0.1) for l in ls], "-", marker=mk, color=c, ms=3.5, lw=1.1, mfc="none")
    ax[0].axhline(1, color=GREY, lw=0.5, ls=":"); ax[0].set_ylim(1e-3, 1e6)
    ax[0].set_xlabel("modes $r$"); ax[0].set_ylabel("spectral radius of $E$"); ax[1].set_xlabel("modes $r$"); ax[1].set_ylabel("eigenvalues with $|\\lambda| > 0.1$")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    fig.suptitle("exact two-grid analysis, $T = 25$, $t_c = 2$, V(2,2)", fontsize=9, y=1.0)
    save(fig, "ch_bases_exact")

# ---------------------------------------------------------------- GMRES and Newton histories
if "histories" in which:
    rows = []
    for line in open(os.path.join(K.RES, "histories.txt")):
        v = line.split()
        rows.append(dict(case=v[0], q=int(v[1]), T=float(v[2]), method=v[3], r=int(v[4]), tc=int(v[5]), pre=int(v[6]),
                         step=int(v[8]), F=float(v[9]), res=[float(x) for x in v[10].split(",")] if len(v) > 10 and v[10] else []))
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.7), gridspec_kw=dict(wspace=0.35))
    Ts = sorted({d["T"] for d in rows if d["case"] == "repeat1"})
    cmap = plt.cm.viridis
    for d in rows:
        if d["case"] != "repeat1" or d["step"] != 1 or not d["res"]: continue
        c = cmap(Ts.index(d["T"]) / (len(Ts) - 1))
        if d["method"] == "plain":
            ax[0].semilogy(np.arange(1, len(d["res"]) + 1), d["res"], color=c, lw=1.1, label=f"$T = {d['T']:.0f}$")
            ax[1].semilogy(np.arange(1, len(d["res"]) + 1) / d["T"], d["res"], color=c, lw=1.1)
        elif d["r"] == 16 and d["tc"] == 2:
            ax[0].semilogy(np.arange(1, len(d["res"]) + 1), d["res"], "o-", color=c, lw=1.1, ms=3,
                           label=f"$T = {d['T']:.0f}$" if d["T"] > 450 else None)
    ax[0].set_xscale("log"); ax[0].set_xlabel("GMRES iteration"); ax[0].set_ylabel("relative residual"); ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("first Newton step: $B$ (lines), two-grid (dots)", fontsize=8)
    ax[1].set_xlabel("iteration / $T$"); ax[1].set_title("$B$ alone: the plateau scales with $T$", fontsize=8)
    for d in rows:
        if d["case"] != "repeat1" or d["method"] != "twogrid" or d["r"] != 16 or d["tc"] != 2: continue
    for T in Ts:
        dd = sorted([d for d in rows if d["case"] == "repeat1" and d["method"] == "twogrid" and d["r"] == 16 and d["tc"] == 2 and d["T"] == T], key=lambda d: d["step"])
        ax[2].semilogy([d["step"] - 1 for d in dd], [d["F"] for d in dd], "o-", color=cmap(Ts.index(T) / (len(Ts) - 1)), ms=3, lw=1.1)
    ax[2].set_xlabel("Newton step"); ax[2].set_ylabel("$\\|F\\|$"); ax[2].set_title("Newton with the two-grid preconditioner", fontsize=8)
    save(fig, "ch_histories")

# ---------------------------------------------------------------- Newton searches with POD and resolvent bases
if "podscan" in which:
    rows = []
    for line in open(os.path.join(K.RES, "bases.txt")):
        v = line.split()
        rows.append(dict(case=v[0], q=int(v[1]), T=float(v[2]), method=v[4], r=int(v[5]), res=float(v[18]),
                         its=[int(x) for x in v[19].split(",")]))
    names = [("sine", "sine modes", GREY, "o"), ("podA", "POD, attractor", BLUE, "s"), ("resS", "resolvent of $S$", RED, "P"),
             ("resF", "frozen-time resolvent", "#c07090", "X"), ("resM", "mean-flow resolvent", "#2a9d8f", "*")]
    cases = [("repeat1", 8), ("prime", 8), ("repeat8", 8), ("repeat1", 64)]
    fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.4), sharey=True, gridspec_kw=dict(wspace=0.1))
    for a, (case, q) in zip(ax, cases):
        T = 0
        for m, lab, c, mk in names:
            dd = sorted([d for d in rows if d["case"] == case and d["q"] == q and d["method"] == m], key=lambda d: d["r"])
            if not dd: continue
            T = dd[0]["T"]
            a.plot([d["r"] for d in dd], [np.mean(d["its"]) if d["res"] < 1e-10 else np.nan for d in dd], "-" + mk, color=c, ms=3.5, lw=1.1, mfc="none", label=lab)
            bad = [d for d in dd if d["res"] >= 1e-10]
            a.plot([d["r"] for d in bad], [70] * len(bad), "x", color=c, ms=4)
        a.set_title(f"$T = {T:.0f}$" + (" (prime)" if case == "prime" else ""), fontsize=8.5)
        a.set_yscale("log"); a.set_ylim(1, 90); a.set_xlabel("modes per slice $r$")
    ax[0].set_ylabel("GMRES iterations\nper Newton step")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=5)
    save(fig, "ch_bases_newton")

# ---------------------------------------------------------------- B-splines against Fourier modes in time
if "timefourier" in which:
    D = np.load(os.path.join(K.HERE, "timecoarse.npz"))
    sp, fo = D["spline"], D["fourier"]
    fig = plt.figure(figsize=(8.6, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.55)
    a = fig.add_subplot(gs[0, 0])
    a.semilogy(sp[:, 0], sp[:, 1], "o-", color=PURPLE, ms=3, lw=1.1, label="B-splines")
    a.semilogy(fo[:, 0], fo[:, 1], "s-", color=GREEN, ms=3, lw=1.1, label="Fourier, $|n| \\leq N$")
    for (x, y, tc) in zip(sp[:, 0], sp[:, 1], (1, 2, 3, 4, 6, 8)):
        a.annotate(f"$t_c={tc}$", (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6, color=PURPLE)
    a.set_xlabel("time functions"); a.set_ylabel("spectral radius of $E$"); a.legend(frameon=False, fontsize=7)
    a.set_title("$r = 16$, V(2,2), $T = 25$", fontsize=8.5)
    a = fig.add_subplot(gs[0, 1])
    a.plot(sp[:, 0], sp[:, 2], "o-", color=PURPLE, ms=3, lw=1.1); a.plot(fo[:, 0], fo[:, 2], "s-", color=GREEN, ms=3, lw=1.1)
    a.set_xlabel("time functions"); a.set_ylabel("eigenvalues of $E$, $|\\lambda| > 0.1$"); a.set_xlim(0, 40)
    a = fig.add_subplot(gs[0, 2])
    a.semilogy(D["bins"] + 0.125, D["prof"], color=BLUE, lw=1.3)
    a.set_xlabel("$\\omega |n - n'|$"); a.set_ylabel("coupling / diagonal")
    a.set_title("$\\|\\hat{\\mathcal{J}}_{n n'}\\|$, $k, k' \\leq 16$", fontsize=8.5)
    for j, (Kc, t) in enumerate(((D["Ks"], "$K$, B-splines, 36 functions"), (D["Kf"], "$K$, Fourier $|n| \\leq 18$, 37 functions"))):
        a = fig.add_subplot(gs[1, j])
        z = np.abs(Kc); im = a.imshow(z, norm=LogNorm(1e-8 * z.max(), z.max()), cmap="magma_r", interpolation="nearest", rasterized=True)
        a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.045, pad=0.03)
    # memory of the coarse factors against T: banded (splines) and dense in the coupled band (Fourier)
    a = fig.add_subplot(gs[1, 2])
    Ts = np.logspace(np.log10(25), np.log10(3000), 30); dt = 25.37 / 73; r = 16
    Nc = Ts / dt / 2; memS = 31 * (r + 1) ** 2 * Nc * 8 / 1e6
    NF = 2 * 2.2 * Ts / (2 * np.pi) + 1; memF = (r * NF) ** 2 * 8 / 1e6
    a.loglog(Ts, memS, color=PURPLE, lw=1.3, label="B-splines, $t_c = 2$: $\\propto T$")
    a.loglog(Ts, memF, color=GREEN, lw=1.3, label="Fourier $|\\omega n| \\leq 2.2$: $\\propto T^2$")
    a.axhline(8000, color=GREY, lw=0.5, ls=":"); a.text(30, 3000, "8 GB", fontsize=7, color=GREY)
    a.set_xlabel("period $T$"); a.set_ylabel("coarse factors (MB)"); a.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, "ch_time_fourier")

# ---------------------------------------------------------------- convergence of the POD of the attractor
if "podconv" in which:
    C = np.loadtxt(os.path.join(K.HERE, "pod", "convergence.txt"))
    full = np.loadtxt(os.path.join(K.HERE, "pod", "energy_attractor.txt"))
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.5), gridspec_kw=dict(wspace=0.35))
    L_ = C[C[:, 0] > 0]; Q = C[C[:, 0] < 0]
    for j, (r, c) in enumerate(zip((4, 8, 12, 16), (GREY, BLUE, PURPLE, RED))):
        ax[0].loglog(L_[:, 0], L_[:, -4 + j], "o-", color=c, ms=3, lw=1.1, label=f"{r} modes")
        ax[0].plot([40000 / 4] * len(Q), Q[:, -4 + j], "x", color=c, ms=4)
    ax[0].set_xlabel("record length (time units)"); ax[0].set_ylabel("largest principal angle (deg)")
    ax[0].set_title("POD subspaces against the full record\n(crosses: the four disjoint quarters)", fontsize=8)
    ax[0].legend(frameon=False, fontsize=7)
    for row, c in zip(L_, plt.cm.viridis(np.linspace(0, 1, len(L_)))):
        ax[1].semilogy(np.arange(1, 17), row[1:17], "-", color=c, lw=1.0)
    ax[1].semilogy(np.arange(1, 17), full[:16], "k--", lw=1.0, label="full record, 40 000")
    ax[1].set_xlabel("POD mode"); ax[1].set_ylabel("energy fraction"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("POD energies, records of 625 to 20 000", fontsize=8)
    save(fig, "ch_pod_convergence")

# ---------------------------------------------------------------- the Fourier mode of TwoGrid (time = :fourier)
if "fourierimpl" in which:
    def rho(case, cfg):
        l = np.abs(eigE(cfg, case))
        return l.max(), int(np.sum(l > 0.1))
    tcs = (2, 3, 4, 6, 8)
    fig = plt.figure(figsize=(8.6, 5.6))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42)
    a0, a1, a2 = (fig.add_subplot(gs[0, j]) for j in range(3))
    for time, c, mk, lab in (("splines", PURPLE, "P", "B-splines"), ("fourier", RED, "D", "Fourier + compact")):
        for r, ls in ((16, "-"), (12, "--")):
            v = np.array([rho("M1", f"sine_r{r}_tc{tc}_V22_{time}") for tc in tcs])
            a0.semilogy(tcs, v[:, 0], ls, marker=mk, color=c, ms=4, mfc="none", lw=1.1, label=f"{lab}, $r = {r}$")
            a1.plot(tcs, v[:, 1], ls, marker=mk, color=c, ms=4, mfc="none", lw=1.1)
        q = np.array([rho("Q2", f"sine_r16_tc{tc}_V22_{time}") for tc in (2, 4)])
        a0.plot((2, 4), q[:, 0], mk, color=c, ms=7, alpha=0.5, label=f"{lab}, $r = 16$, $T = 51$")
    a0.set_xlabel("$t_c$"); a0.set_ylabel("spectral radius of $E$"); a0.set_xticks(tcs)
    a1.set_xlabel("$t_c$"); a1.set_ylabel("eigenvalues of $E$ with $|\\lambda| > 0.1$"); a1.set_xticks(tcs)
    h, l = a0.get_legend_handles_labels()
    a1.legend(h, l, frameon=False, fontsize=6.3, loc="upper left")
    a0.set_title("cycle V(2,2), $T = 25$", fontsize=8); a1.set_title("unresolved eigenvalues", fontsize=8)
    th = np.linspace(0, 2 * np.pi, 200)
    for time, c, mk in (("splines", PURPLE, "P"), ("fourier", RED, "D")):
        l = eigE(f"sine_r16_tc4_V22_{time}")
        l = l[np.abs(l) > 1e-3]
        a2.plot(l.real, l.imag, mk, color=c, ms=2.6, mfc="none", mew=0.6, label=time)
    a2.plot(0.1 * np.cos(th), 0.1 * np.sin(th), color=GREY, lw=0.6, ls=":")
    a2.set_aspect("equal"); a2.set_xlabel("Re $\\lambda$"); a2.set_ylabel("Im $\\lambda$")
    a2.set_title("eigenvalues of $E$ above $10^{-3}$, $t_c = 4$", fontsize=8); a2.legend(frameon=False, fontsize=7)
    # the coarse matrix of the Fourier mode, and how close it is to the exact Galerkin product
    cfg = "sine_r16_tc2_V22_fourier"
    n, nc, rc, Nb = dims(cfg)
    Kt = K.read_dense(f"M1_K_{cfg}.bin", nc); G = K.read_dense(f"M1_G_{cfg}.bin", nc)
    m = np.array([1, 120, 1191, 2416, 1191, 120, 1]) / 5040
    Cm = sum(m[j + 3] * np.roll(np.eye(Nb), j, axis=1) for j in range(-3, 4))
    Mf = np.eye(nc); Mf[:rc * Nb, :rc * Nb] = np.kron(Cm, np.eye(rc))
    Kc = np.linalg.solve(Mf, Kt)
    # PᵀJP is in the units of the dense basis: compare up to the scale that fits best, on the field
    # part, in total and by coarse temporal harmonic (unitary DFT over the coarse times)
    f = slice(0, rc * Nb)
    c = np.sum(Kc[f, f] * G[f, f]) / np.sum(G[f, f] ** 2)
    E, Gs = (Kc - c * G)[f, f], c * G[f, f]
    Fk = np.kron(np.fft.fft(np.eye(Nb)) / np.sqrt(Nb), np.eye(rc))
    Eh, Gh = Fk @ E @ Fk.conj().T, Fk @ Gs @ Fk.conj().T
    nh = np.repeat(np.abs(np.fft.fftfreq(Nb, 1 / Nb)), rc); N = (Nb - 1) // 2
    low = nh <= N // 2
    print(cfg, f"‖M⁻¹K - c PᵀJP‖/‖c PᵀJP‖ = {np.linalg.norm(E) / np.linalg.norm(Gs):.3f}; share of the three "
          f"highest coarse harmonics {np.linalg.norm(Eh[nh >= N - 2, :]) ** 2 / np.linalg.norm(Eh) ** 2:.2f}; "
          f"on |n| <= N/2 {np.linalg.norm(Eh[np.ix_(low, low)]) / np.linalg.norm(Gh[np.ix_(low, low)]):.3f}")
    a = fig.add_subplot(gs[1, 0])
    a.spy(np.abs(Kt) > 1e-9 * np.abs(Kt).max(), markersize=0.15, color="k", rasterized=True)
    a.set_title(f"$K$, Fourier, $t_c = 2$: {nc} unknowns", fontsize=8); a.tick_params(labelsize=7)
    a = fig.add_subplot(gs[1, 1])
    z = np.abs(G[:6 * rc, :6 * rc])
    a.imshow(z, norm=LogNorm(1e-4 * z.max(), z.max()), cmap="magma_r", interpolation="nearest")
    for bb in range(1, 6): a.axhline(bb * rc - 0.5, color=GREY, lw=0.3); a.axvline(bb * rc - 0.5, color=GREY, lw=0.3)
    a.set_title("exact Galerkin $P^{T}\\mathcal{J}P$, first 6 block rows", fontsize=8); a.tick_params(labelsize=7)
    # Newton searches against the period
    a = fig.add_subplot(gs[1, 2])
    rows = [l.split() for l in open(os.path.join(K.HERE, "..", "results", "fourier_newton.txt"))]
    for time, c, mk in (("splines", PURPLE, "P"), ("fourier", RED, "D")):
        for tc, ls in ((2, "-"), (4, "--"), (6, ":")):
            sel = sorted((float(r[2]), np.mean([int(v) for v in r[19].split(",")])) for r in rows
                         if r[0] == "repeat1" and r[4] == time and int(r[6]) == tc)
            if sel:
                a.semilogx(*zip(*sel), ls, marker=mk, color=c, ms=3.5, mfc="none", lw=1.0, label=f"{time}, $t_c = {tc}$")
    a.set_xlabel("$T$"); a.set_ylabel("GMRES iterations per Newton step"); a.set_title("searches, $r = 16$, V(2,2)", fontsize=8)
    a.legend(frameon=False, fontsize=6.3)
    save(fig, "ch_fourier_impl")

# ---------------------------------------------------------------- Fourier restriction, coarse problem in time
if "fouriertransfer" in which:
    D = np.load(os.path.join(K.HERE, "fouriertransfer.npz"))
    Nc = D["Nc"]; Ns = 73
    labs = [("galerkin", "Fourier, exact Galerkin (dense)", GREEN, "s", "-"),
            ("colloc-spec", "Fourier + collocation, spectral $\\partial_t$ (dense)", BLUE, "o", "--"),
            ("colloc-fd8", "Fourier + collocation, FD8 (banded)", GREY, "v", "-"),
            ("colloc-c4", "Fourier + collocation, compact 4th order (banded)", ORANGE, "^", "-"),
            ("colloc-cg", "Fourier + collocation, spline-Galerkin $\\partial_t$ (banded)", RED, "D", "-"),
            ("splines", "B-splines, exact Galerkin (banded, TwoGrid)", PURPLE, "P", "-")]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.8), gridspec_kw=dict(wspace=0.3))
    for k, lab, c, mk, ls in labs:
        v = D[k]
        ax[0].semilogy(Ns / Nc, v[:, 0], ls, marker=mk, color=c, ms=3.5, lw=1.1, mfc="none", label=lab)
        ax[1].plot(Ns / Nc, v[:, 1], ls, marker=mk, color=c, ms=3.5, lw=1.1, mfc="none")
    ax[0].axhline(1, color=GREY, lw=0.5, ls=":")
    for a in ax: a.set_xlabel("coarsening in time $N_s / N_c$")
    ax[0].set_ylabel("spectral radius of $E$"); ax[1].set_ylabel("eigenvalues of $E$ with $|\\lambda| > 0.1$")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    fig.suptitle("$r = 16$, V(2,2), $T = 25$", fontsize=9, y=1.0)
    save(fig, "ch_fourier_transfer")
