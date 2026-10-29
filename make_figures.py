"""Build every figure in figures/ from runs/ and results/."""
import json
import pathlib
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = pathlib.Path("figures")
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

STYLE = {
    "pursuit": ("Pure pursuit", "#b8b8b8", ":"),
    "pn": ("Proportional navigation (tuned)", "#555555", "--"),
    "direct": ("PPO from scratch", "#e07b39", "-"),
    "residual": ("Residual PPO on PN", "#2f6fb0", "-"),
    "residual_priv": ("Residual PPO + privileged critic", "#2e8b57", "-"),
    "residual_nodr": ("Residual PPO, fixed easy sensing (no randomisation)", "#9b59b6", "-"),
}


def config_of(stem):
    m = re.match(r"guid_(\w+?)_dr-(\w+)_sh[\d.]+_(\w+)_s\d+", stem)
    mode, dr, critic = m.groups()
    if mode == "direct":
        return "direct"
    if dr == "none":
        return "residual_nodr"
    return "residual_priv" if critic == "priv" else "residual"


def training_curves(runs, ev):
    groups = defaultdict(list)
    for f in sorted(runs.glob("guid_*.json")):
        groups[config_of(f.stem)].append(json.loads(f.read_text()))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, curves in groups.items():
        label, color, ls = STYLE[key]
        steps = np.array([c["steps"] for c in curves[0]]) / 1e6
        y = np.array([[c["intercept"] for c in cur] for cur in curves])
        k = 5
        y = np.array([np.convolve(r, np.ones(k) / k, mode="valid") for r in y])
        x = steps[k - 1:]
        ax.plot(x, y.mean(0), color=color, ls=ls, label=label)
        ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=0.15, lw=0)
    if ev:
        ax.axhline(ev["baselines"]["pn"]["train_dist"]["intercept"], color=STYLE["pn"][1], ls="--",
                   label=STYLE["pn"][0])
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("interception rate (training)")
    ax.set_title("Learning to intercept under randomised sensing", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)


def by_config(ev):
    out = defaultdict(list)
    for stem, r in ev["policies"].items():
        out[config_of(stem)].append(r)
    return out


def robustness(ev):
    cfgs = by_config(ev)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, axis, xlab, train_hi in [(axes[0], "noise", "sensor noise std at 20 m (m)", 1.0),
                                     (axes[1], "delay", "measurement delay (steps of 50 ms)", 3)]:
        xs = ev["axes"][axis]
        for key in ["pursuit", "pn"]:
            label, color, ls = STYLE[key]
            ax.plot(xs, [r["intercept"] for r in ev["baselines"][key][axis]], color=color, ls=ls,
                    marker="o", ms=3, label=label)
        for key, runs in cfgs.items():
            label, color, ls = STYLE[key]
            y = np.array([[r["intercept"] for r in run[axis]] for run in runs])
            ax.plot(xs, y.mean(0), color=color, ls=ls, marker="o", ms=3, label=label)
            ax.fill_between(xs, y.min(0), y.max(0), color=color, alpha=0.15, lw=0)
        ax.axvspan(xs[0], train_hi, color="#2f6fb0", alpha=0.06, lw=0)
        ax.text(train_hi, 0.98, " training range ", ha="right", va="top", fontsize=7, color="#2f6fb0")
        ax.set_xlabel(xlab)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("interception rate")
    axes[0].set_title("Robustness outside the training range", loc="left")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIG / "robustness.png")
    plt.close(fig)


def attackers(ev):
    cfgs = by_config(ev)
    names = ev["axes"]["attackers"]
    rows = [("pn", [ev["baselines"]["pn"]["attacker"][a]["intercept"] for a in names])]
    for key, runs in cfgs.items():
        rows.append((key, [np.mean([r["attacker"][a]["intercept"] for r in runs]) for a in names]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.8 / len(rows)
    for i, (key, vals) in enumerate(rows):
        label, color, _ = STYLE[key]
        ax.bar(np.arange(len(names)) + i * w - 0.4 + w / 2, vals, w, color=color, label=label)
    ax.set_xticks(range(len(names)), ["straight\n(easier)", "jinking\n(trained on)", "evasive\n(never seen)"])
    ax.set_ylabel("interception rate")
    ax.set_ylim(0, 1)
    ax.set_title("Generalisation to attacker behaviours", loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "attackers.png")
    plt.close(fig)


def selfplay(path, out="selfplay.png", title_suffix=""):
    if not path.exists():
        return
    sp = json.loads(path.read_text())
    labels, score = sp["labels"], np.array(sp["score"])
    de, ae = np.array(sp["defender_elo"]), np.array(sp["attacker_elo"])
    order = sorted(range(len(labels)), key=lambda i: (labels[i]["method"], labels[i]["seed"], labels[i]["chunk"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    im = axes[0].imshow(score[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
    ticks = [i for i, o in enumerate(order) if i == 0 or labels[o]["method"] != labels[order[i - 1]]["method"]]
    for t in ticks[1:]:
        axes[0].axhline(t - 0.5, color="white", lw=1)
        axes[0].axvline(t - 0.5, color="white", lw=1)
    axes[0].set_xticks(ticks, [labels[order[t]]["method"].upper() for t in ticks], ha="left")
    axes[0].set_yticks(ticks, [labels[order[t]]["method"].upper() for t in ticks], va="top")
    axes[0].set_xlabel("attacker checkpoint (seed, then training time)")
    axes[0].set_ylabel("defender checkpoint")
    axes[0].set_title("Defender score in cross-play" + title_suffix, loc="left")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for m, color in [("ippo", "#e07b39"), ("sippo", "#2f6fb0")]:
        for side, elo_v, ls in [("defender", de, "-"), ("attacker", ae, ":")]:
            idx = [i for i in order if labels[i]["method"] == m]
            chunks = sorted({labels[i]["chunk"] for i in idx})
            y = [np.mean([elo_v[i] for i in idx if labels[i]["chunk"] == c]) for c in chunks]
            axes[1].plot(np.array(chunks) + 1, y, color=color, ls=ls, marker="o", ms=3,
                         label=f"{m.upper()} {side}")
    axes[1].set_xlabel("training chunk")
    axes[1].set_ylabel("Elo (pooled population)")
    axes[1].set_title("Elo of checkpoints over training", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / out)
    plt.close(fig)


def main():
    runs, res = pathlib.Path("runs"), pathlib.Path("results")
    ev_path = res / "guidance_eval.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else None
    training_curves(runs, ev)
    if ev:
        robustness(ev)
        attackers(ev)
    selfplay(res / "selfplay_v1_eval.json", "selfplay_v1.png", " (v1 reward)")
    selfplay(res / "selfplay_v2_eval.json", "selfplay_v2.png", " (v2 reward)")
    print("figures written to", FIG.resolve())


if __name__ == "__main__":
    main()
