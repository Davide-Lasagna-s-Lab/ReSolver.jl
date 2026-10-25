"""Figures of the documentation of PeriodicBSplines.jl, from the data written by figures.jl."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#285aaa", "#be3232", "#288c50", "#6e6e6e", "#e08a00", "#7b3fa0"
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "assets")
D = os.path.join(A, "data")
ld = lambda n: np.loadtxt(os.path.join(D, n))
def save(fig, name):
    fig.savefig(os.path.join(A, name + ".png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# 1. basis
c = ld("cardinal.txt"); p = ld("periodic.txt")
fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.4]))
ax[0].plot(c[:, 0], c[:, 1], color=BLUE, lw=1.4, label="$B(u)$")
ax[0].plot(c[:, 0], c[:, 2], color=ORANGE, lw=1.1, label="$B'(u)$")
ax[0].plot(c[:, 0], c[:, 3] / 4, color=GREY, lw=1.0, ls="--", label="$B''(u)/4$")
ax[0].axhline(0, color=GREY, lw=0.4); ax[0].set_xlabel("$u$"); ax[0].legend(frameon=False, fontsize=7.5)
ax[0].set_title("cardinal cubic B-spline", fontsize=9)
s = p[:, 0]
for m in range(8):
    ax[1].plot(s, p[:, 1 + m], lw=1.1, color=plt.cm.viridis(m / 8))
ax[1].plot(s, p[:, -1], color="k", lw=1.0, ls="--", label="$\\sum_m \\phi_m = 1$")
ax[1].set_xlabel("$s$"); ax[1].set_xticks([0, np.pi, 2 * np.pi]); ax[1].set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
ax[1].set_title("periodic splines $\\phi_m$, $N_c = 8$", fontsize=9); ax[1].set_ylim(0, 1.25); ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")
save(fig, "basis")

# 2. matrices
Phi = ld("collocation.txt"); M = ld("gram.txt"); Dm = ld("derivative_gram.txt")
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw=dict(wspace=0.35, width_ratios=[0.9, 1, 1]))
ax[0].imshow(Phi, cmap="Greys", aspect="auto", interpolation="nearest"); ax[0].set_title("$\\Phi$: $\\phi_m(s_l)$, $N_s = 40$", fontsize=9)
ax[0].set_xlabel("spline $m$"); ax[0].set_ylabel("point $l$")
v = np.abs(M).max(); ax[1].imshow(M, cmap="Blues", interpolation="nearest"); ax[1].set_title("Gram $M$ (circulant)", fontsize=9)
v = np.abs(Dm).max(); ax[2].imshow(Dm, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest"); ax[2].set_title("$D_{mm'} = \\int \\phi_m \\phi'_{m'}$", fontsize=9)
for a in ax[1:]: a.set_xlabel("$m'$"); a.set_ylabel("$m$")
save(fig, "matrices")

# 3. accuracy and resolution
cv = ld("convergence.txt"); hm = ld("harmonics.txt"); wn = ld("wavenumber.txt")
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.6), gridspec_kw=dict(wspace=0.42))
ax[0].loglog(cv[:, 0], cv[:, 1], "o-", color=BLUE, ms=3, lw=1.1, label="$e^{\\sin s}$")
ax[0].loglog(cv[:, 0], cv[0, 1] * (cv[:, 0] / cv[0, 0]) ** -4, ":", color=GREY, lw=0.9, label="$N_c^{-4}$")
ax[0].set_xlabel("$N_c$"); ax[0].set_ylabel("max interpolation error"); ax[0].legend(frameon=False, fontsize=7.5)
ax[0].set_title("fourth order", fontsize=9)
ax[1].semilogy(hm[:, 0], hm[:, 1], "-", color=BLUE, lw=1.3)
for y in (0.1, 0.01): ax[1].axhline(y, color=GREY, lw=0.5, ls=":")
ax[1].set_xlabel("splines per period of $\\cos ns$"); ax[1].set_ylabel("max interpolation error")
ax[1].set_title("resolution of a harmonic", fontsize=9)
th = wn[:, 0]
ax[2].plot(th, th, color="k", lw=0.8, label="exact")
ax[2].plot(th, wn[:, 1], color=BLUE, lw=1.4, label="splines, Galerkin")
for j, (lab, c_) in enumerate((("FD2", GREY), ("FD4", ORANGE), ("FD8", GREEN))):
    ax[2].plot(th, wn[:, 2 + j], color=c_, lw=1.0, ls="--", label=lab)
ax[2].set_xlabel("$\\theta = 2\\pi/(\\mathrm{points\\ per\\ period})$"); ax[2].set_ylabel("modified wavenumber $\\theta^*$")
ax[2].set_xticks([0, np.pi / 2, np.pi]); ax[2].set_xticklabels(["0", "$\\pi/2$", "$\\pi$"])
ax[2].legend(frameon=False, fontsize=7); ax[2].set_title("derivative of a harmonic", fontsize=9)
save(fig, "accuracy")
print("figures written to", A)
