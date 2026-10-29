"""Render the same engagements flown by tuned PN and by a learned policy.

Writes figures/demo_trajectories.png (static, several engagements) and figures/demo.gif
(one engagement animated side by side). Both controllers face identical attacker spawns
and jinking parameters; only their decisions differ.
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from evaluate_guidance import condition, load
from skyguard.env import OUTCOMES, RUNNING, make

FIG = pathlib.Path("figures")


def fly(mode, agent, seed, n, **cond):
    env = make(n_envs=n, defender_mode=mode, seed=seed, **cond)
    obs, _ = env.reset()
    pd, pa, est, outcome = [env.pd.copy()], [env.pa.copy()], [env.pd + env.xh], np.full(n, RUNNING)
    for _ in range(env.cfg.max_steps):
        act = np.zeros((n, 3), np.float32) if agent is None else agent.act(obs, deterministic=True,
                                                                             update_norm=False)[0]
        obs, _, _, _, done, info = env.step(act)
        newly = done & (outcome == RUNNING)
        outcome[newly] = info["outcome"][newly]
        pd.append(np.where(outcome[:, None] == RUNNING, env.pd, pd[-1]))
        pa.append(np.where(outcome[:, None] == RUNNING, env.pa, pa[-1]))
        est.append(np.where(outcome[:, None] == RUNNING, env.pd + env.xh, est[-1]))
        if (outcome != RUNNING).all():
            break
    return np.array(pd), np.array(pa), np.array(est), outcome


def draw(ax, pd, pa, est, k, t, title, outcome, box):
    ax.cla()
    ax.plot(*pa[:t, k].T, color="#c0392b", lw=1.5, label="attacker")
    ax.plot(*pd[:t, k].T, color="#2f6fb0", lw=1.8, label="interceptor")
    ax.scatter(*est[max(0, t - 60):t, k].T, color="#e67e22", s=5, alpha=0.5, label="noisy sensor track")
    ax.scatter([0], [0], [0], marker="s", s=60, color="#333333", label="protected asset")
    ax.scatter(*pa[t - 1, k], color="#c0392b", s=18)
    ax.scatter(*pd[t - 1, k], color="#2f6fb0", s=18)
    (x0, y0, _), (x1, y1, z1) = box
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_zlim(0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1))
    ax.view_init(elev=22, azim=-60)
    ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    done = t >= len(pd) - 1 or (pd[t - 1, k] == pd[-1, k]).all()
    ax.set_title(f"{title}\n{OUTCOMES.get(int(outcome[k]), '…') if done else '…'}", fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="path to a runs/guid_*.pt checkpoint")
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--delay", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--n", type=int, default=64)
    a = p.parse_args()
    cond = condition(noise=a.noise, delay=a.delay, dropout=a.dropout)
    mode, agent = load(a.policy)
    learned = "Residual PPO on PN" if mode == "residual" else "PPO from scratch"
    runs = {"Tuned proportional navigation": fly("base", None, a.seed, a.n, **cond),
            learned: fly(mode, agent, a.seed, a.n, **cond)}

    def box(k):
        pts = np.concatenate([np.concatenate([v[0][:, k], v[1][:, k]]) for v in runs.values()] + [np.zeros((1, 3))])
        lo, hi = pts.min(0) - 3, pts.max(0) + 3
        return np.array([lo[0], lo[1], 0]), np.array([hi[0], hi[1], max(hi[2], 5)])
    (_, _, _, o_pn), (_, _, _, o_rl) = runs.values()
    # Prefer engagements PN loses and the policy wins: that is what the demo should show.
    picks = [k for k in range(a.n) if o_pn[k] != 1 and o_rl[k] == 1][:3] or list(range(3))
    print("PN intercept", (o_pn == 1).mean(), "RL intercept", (o_rl == 1).mean(), "picked", picks)

    fig = plt.figure(figsize=(10, 3.6 * len(picks)))
    for r, k in enumerate(picks):
        for c, (title, (pd, pa, est, out)) in enumerate(runs.items()):
            ax = fig.add_subplot(len(picks), 2, 2 * r + c + 1, projection="3d")
            draw(ax, pd, pa, est, k, len(pd), title, out, box(k))
    fig.axes[0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.02, wspace=0, hspace=0.12)
    fig.savefig(FIG / "demo_trajectories.png")
    plt.close(fig)

    k = picks[0]
    T = max(len(v[0]) for v in runs.values())
    fig = plt.figure(figsize=(10, 4.2))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]

    def frame(t):
        for ax, (title, (pd, pa, est, out)) in zip(axes, runs.items()):
            draw(ax, pd, pa, est, k, min(t + 1, len(pd)), title, out, box(k))
        return axes

    anim = animation.FuncAnimation(fig, frame, frames=range(0, T, 2), interval=60)
    anim.save(FIG / "demo.gif", writer=animation.PillowWriter(fps=16))
    print("wrote", FIG / "demo_trajectories.png", FIG / "demo.gif")


if __name__ == "__main__":
    main()
