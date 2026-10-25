"""Figures of the two-grid study, from results/*.txt.      python3 figures.py [outdir]"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.colors import LogNorm

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "figs")
os.makedirs(OUT, exist_ok=True)
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#285aaa", "#be3232", "#288c50", "#6e6e6e", "#e08a00", "#7b3fa0"
T0 = 25.3706
DT = None

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=130, bbox_inches="tight")

def load(name):
    rows = []
    path = os.path.join(RES, name)
    if not os.path.exists(path): return rows
    for line in open(path):
        v = line.split()
        if len(v) < 19: continue
        d = dict(case=v[0], q=int(v[1]), T=float(v[2]), Ns=int(v[3]), method=v[4], r=int(v[5]), tc=int(v[6]),
                 pre=int(v[7]), post=int(v[8]), h=int(v[9]), steps=int(v[10]), actions=int(v[11]), jac=int(v[12]),
                 builds=int(v[13]), sec=float(v[14]), kryl=float(v[15]), coarse=float(v[16]), nodes=int(v[17]),
                 res=float(v[18]), its=[int(x) for x in v[19].split(",")] if len(v) > 19 and v[19] else [])
        d["conv"] = d["res"] < 1e-10
        rows.append(d)
    return rows

which = sys.argv[2:] or ["map", "T", "r", "cycle"]

# ---------------------------------------------------------------- smoothing map
if "map" in which:
    fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.6), gridspec_kw=dict(wspace=0.7, width_ratios=[1.3, 1, 1]))
    D = {M: np.loadtxt(os.path.join(RES, f"smoothing_map_M{M}.txt")) for M in (1, 8)}
    wn, amp = D[1][0], D[1][1:]
    a = ax[0]
    k = np.arange(1, amp.shape[0] + 1)
    pc = a.pcolormesh(wn, k, amp, norm=LogNorm(vmin=0.01, vmax=10), cmap="Blues", shading="nearest")
    cs = a.contour(wn, k, amp, levels=[0.5, 1.0], colors=[ORANGE, RED], linewidths=1.0)
    a.clabel(cs, fmt={0.5: "0.5", 1.0: "1"}, fontsize=6.5)
    cb = fig.colorbar(pc, ax=a, pad=0.02)
    a.set_xlabel(r"physical frequency $\omega n$"); a.set_ylabel("sine mode $k$")
    a.set_xlim(0, 6); a.set_title("amplification, base orbit", fontsize=8.5)
    a = ax[1]
    for M, c in ((1, BLUE), (8, RED)):
        wn, amp = D[M][0], D[M][1:]
        low = wn <= 0.5
        a.semilogy(np.arange(1, amp.shape[0] + 1), amp[:, low].max(1), color=c, lw=1.3, label=f"$T = {25.37 if M == 1 else 187.5:.0f}$")
    a.axhline(1, color=GREY, lw=0.7, ls=":"); a.set_xlabel("sine mode $k$"); a.set_ylabel("amplification")
    a.legend(frameon=False, fontsize=7); a.set_title(r"in space, $|\omega n| \leq 0.5$", fontsize=8.5)
    a = ax[2]
    for M, c in ((1, BLUE), (8, RED)):
        wn, amp = D[M][0], D[M][1:]
        a.loglog(np.maximum(wn, 1e-2), amp[:16].max(0), "o" if M == 1 else "-", color=c, ms=2.5, lw=1.2,
                 label=f"$T = {25.37 if M == 1 else 187.5:.0f}$")
    a.axhline(1, color=GREY, lw=0.7, ls=":"); a.set_xlabel(r"physical frequency $\omega n$"); a.set_ylabel("amplification")
    a.set_xlim(0.08, 30); a.legend(frameon=False, fontsize=7); a.set_title(r"in time, $k \leq 16$", fontsize=8.5)
    save(fig, "tg_smoothing_map")

# ---------------------------------------------------------------- cost against T
if "T" in which:
    R = load("T_scan.txt")
    if os.path.exists(os.path.join(RES, "T_twogrid.txt")):    # two-grid rows rerun with the final code
        R = [d for d in R if d["method"] != "twogrid"] + load("T_twogrid.txt")
    cfg = [("twogrid", 12, 1, 1, "two-grid, $r=12$, every slice, V(1,1)", GREEN, "^", "-"),
           ("twogrid", 16, 1, 1, "two-grid, $r=16$, every slice, V(1,1)", BLUE, "s", "-"),
           ("twogrid", 16, 2, 2, "two-grid, $r=16$, $t_c=2$, V(2,2)", PURPLE, "D", "-"),
           ("twogrid", 16, 4, 2, "two-grid, $r=16$, $t_c=4$, V(2,2)", ORANGE, "v", "-")]
    def sel(method, r=None, tc=None, pre=None):
        out = [d for d in R if d["method"] in method and (r is None or d["r"] == r) and (tc is None or d["tc"] == tc) and (pre is None or d["pre"] == pre)]
        return sorted(out, key=lambda d: d["T"])
    fig, ax = plt.subplots(1, 4, figsize=(8.0, 2.55), gridspec_kw=dict(wspace=0.45))
    plain = sel(("plain", "plain1"))
    methods = [("plain", None, None, None, "$B$ only", GREY, "o")] + [(m, r, tc, pre, l, c, mk) for (m, r, tc, pre, l, c, mk, ls) in cfg]
    def val(d, key):
        if key == "its": return np.mean(d["its"])
        if key == "act": return d["actions"] / d["steps"]
        if key == "sec": return d["sec"] / d["steps"]
        return d["kryl"] + d["coarse"]
    for a, key, lab in zip(ax, ("its", "act", "sec", "mem"),
                           ("GMRES iterations\nper Newton step", "Jacobian actions\nper Newton step",
                            "seconds per Newton step", "memory (MB)")):
        for (m, r, tc, pre, l, c, mk) in methods:
            dd = plain if m == "plain" else sel((m,), r, tc, pre)
            for case, fill, ls in (("repeat1", "none", "-"), ("repeat8", "none", ""), ("prime", c, "")):
                p = sorted([d for d in dd if d["case"] == case], key=lambda d: d["T"])
                if not p: continue
                a.plot([d["T"] for d in p], [val(d, key) for d in p], marker=mk if case != "repeat8" else "D",
                       ls=ls if ls else "none", color=c, ms=3.2, mfc=fill, lw=0.9,
                       label=l if case == "repeat1" else None)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("period $T$"); a.set_title(lab, fontsize=8)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    save(fig, "tg_vs_T")

    # total Jacobian actions to convergence, two-grid only, and B only where converged
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw=dict(wspace=0.35))
    for (m, r, tc, pre, l, c, mk) in methods:
        dd = plain if m == "plain" else sel((m,), r, tc, pre)
        for case, fill, ls in (("repeat1", "none", "-"), ("repeat8", "none", ""), ("prime", c, "")):
            p = sorted([d for d in dd if d["case"] == case and d["conv"] and d["method"] != "plain1"], key=lambda d: d["T"])
            if p:
                ax[0].plot([d["T"] for d in p], [d["actions"] for d in p], marker=mk if case != "repeat8" else "D",
                           ls=ls if ls else "none", color=c, ms=3.2, mfc=fill, lw=0.9, label=l if case == "repeat1" else None)
            p = sorted([d for d in dd if d["case"] == case], key=lambda d: d["T"])
            if p:
                ax[1].plot([d["T"] for d in p], [d["kryl"] if m == "plain" else d["coarse"] for d in p],
                           marker=mk if case != "repeat8" else "D", ls=ls if ls else "none", color=c, ms=3.2,
                           mfc=fill, lw=0.9)
    for a in ax: a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("period $T$")
    ax[0].set_ylabel("Jacobian actions, whole search")
    ax[1].set_ylabel("MB: Krylov basis ($B$ only)\nor coarse factors (two-grid)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    save(fig, "tg_total")

# ---------------------------------------------------------------- truncation level
if "r" in which:
    R = load("r_scan.txt")
    fig, ax = plt.subplots(2, 3, figsize=(7.9, 4.6), gridspec_kw=dict(wspace=0.35, hspace=0.55), sharex=True)
    cols = {1: BLUE, 2: PURPLE, 4: ORANGE, 8: RED}
    for j, q in enumerate((1, 8, 64)):
        for i, nu in enumerate((1, 2)):
            a = ax[i, j]
            for tc in (1, 2, 4, 8):
                dd = sorted([d for d in R if d["q"] == q and d["tc"] == tc and d["pre"] == nu], key=lambda d: d["r"])
                if not dd: continue
                y = [np.mean(d["its"]) if d["conv"] else np.nan for d in dd]
                a.plot([d["r"] for d in dd], y, "o-", color=cols[tc], ms=3, lw=1.1, label=f"$t_c = {tc}$")
                bad = [d for d in dd if not d["conv"]]
                a.plot([d["r"] for d in bad], [62] * len(bad), "x", color=cols[tc], ms=4)
            a.set_yscale("log"); a.set_ylim(1, 80)
            a.set_title(f"$T = {q * T0:.0f}$, V({nu},{nu})", fontsize=8.5)
            a.axvspan(15.5, 25, color=GREY, alpha=0.08, lw=0)
            if j == 0: a.set_ylabel("GMRES iterations\nper Newton step")
            if i == 1: a.set_xlabel("sine modes per slice $r$")
    ax[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "tg_truncation")

    # cost and memory against r at the longest period
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw=dict(wspace=0.35))
    for nu, ls in ((1, "-"), (2, "--")):
        for tc in (1, 2, 4, 8):
            dd = sorted([d for d in R if d["q"] == 64 and d["tc"] == tc and d["pre"] == nu and d["conv"]], key=lambda d: d["r"])
            if not dd: continue
            ax[0].plot([d["r"] for d in dd], [d["actions"] for d in dd], "o" + ls, color=cols[tc], ms=3, lw=1.1,
                       label=f"$t_c = {tc}$, V({nu},{nu})")
            if nu == 2:
                ax[1].plot([d["r"] for d in dd], [d["coarse"] for d in dd], "o-", color=cols[tc], ms=3, lw=1.1, label=f"$t_c = {tc}$")
    ax[0].set_yscale("log"); ax[0].set_xlabel("sine modes per slice $r$"); ax[0].set_ylabel("Jacobian actions to convergence")
    ax[1].set_yscale("log"); ax[1].set_xlabel("sine modes per slice $r$"); ax[1].set_ylabel("coarse factors (MB)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    ax[1].legend(frameon=False, fontsize=7)
    ax[0].set_title(f"$T = {64 * T0:.0f}$", fontsize=8.5); ax[1].set_title(f"$T = {64 * T0:.0f}$", fontsize=8.5)
    save(fig, "tg_truncation_cost")

# ---------------------------------------------------------------- cycles
if "cycle" in which:
    R = load("cycle_scan.txt")
    if R:
        labs, acts, its, conv = [], [], [], []
        for d in R:
            name = ("additive" if d["method"] == "additive" else f"V({d['pre']},{d['post']})") + \
                   (f", $t_c={d['tc']}$" if d["tc"] > 1 else "") + (f", FD{2 * d['h']}" if d["tc"] == 1 else "")
            labs.append(name); acts.append(d["actions"]); its.append(np.mean(d["its"])); conv.append(d["conv"])
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 0.25 * len(R) + 0.8), gridspec_kw=dict(wspace=0.12), sharey=True)
        y = np.arange(len(R))[::-1]
        ax[0].barh(y, its, color=[BLUE if c else RED for c in conv])
        ax[0].set_xlabel("GMRES iterations per step"); ax[0].set_xlim(0, 62)
        ax[1].barh(y, acts, color=[BLUE if c else RED for c in conv]); ax[1].set_xscale("log")
        ax[1].xaxis.set_minor_formatter(mt.NullFormatter())
        ax[1].set_xlabel("Jacobian actions per search")
        ax[0].set_yticks(y); ax[0].set_yticklabels(labs, fontsize=7.5)
        save(fig, "tg_cycles")
