"""Cross-play every saved defender against every saved attacker, then rate them with Elo.

Matches one defender against all attackers at once: each attacker checkpoint controls its own
slice of a large vectorised arena. Win = intercept for the defender, leak for the attacker,
timeout = draw. Elo is computed the way Hansen et al. (2025) do it: sequential updates over
randomly ordered matches, averaged over several orderings.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from skyguard.env import ATT_OBS_DIM, DEF_OBS_DIM, INTERCEPT, LEAK, TIMEOUT, make
from skyguard.ppo import Agent
from evaluate_guidance import ATTACKERS, condition, episodes


def agents_from(states, obs_dim):
    return [Agent.from_state(s, obs_dim) for s in states]


def crossplay(defenders, attackers, mode, perfect, per_pair=128, seed=7):
    na = len(attackers)
    n = na * per_pair
    score = np.zeros((len(defenders), na))
    for i, dfd in enumerate(defenders):
        env = make(n_envs=n, defender_mode=mode, attacker="policy", perfect_info=perfect, seed=seed + i)
        dobs, aobs = env.reset()
        res = np.zeros(n, np.int64)
        alive = np.ones(n, bool)
        for _ in range(env.cfg.max_steps):
            da = dfd.act(dobs, deterministic=True, update_norm=False)[0]
            aa = np.concatenate([att.act(aobs[j * per_pair:(j + 1) * per_pair], deterministic=True,
                                         update_norm=False)[0] for j, att in enumerate(attackers)])
            dobs, aobs, _, _, done, info = env.step(da, aa)
            first = done & alive
            res[first] = info["outcome"][first]
            alive &= ~done
            if not alive.any():
                break
        r = res.reshape(na, per_pair)
        # Defender score: 1 intercept, 0.5 timeout (draw), 0 leak or crash.
        score[i] = (r == INTERCEPT).mean(1) + 0.5 * (r == TIMEOUT).mean(1)
        print(f"defender {i+1}/{len(defenders)} mean score {score[i].mean():.3f}", flush=True)
    return score


def elo(score, k=16.0, orders=8, matches_per_pair=20, seed=0):
    """score[i, j]: defender i's expected score vs attacker j. Returns (defender_elo, attacker_elo)."""
    rng = np.random.default_rng(seed)
    nd, na = score.shape
    res_d, res_a = np.zeros(nd), np.zeros(na)
    for _ in range(orders):
        rd, ra = np.full(nd, 1200.0), np.full(na, 1200.0)
        pairs = np.array([(i, j) for i in range(nd) for j in range(na)] * matches_per_pair)
        rng.shuffle(pairs)
        outcomes = rng.random(len(pairs)) < score[pairs[:, 0], pairs[:, 1]]
        for (i, j), win in zip(pairs, outcomes):
            e = 1.0 / (1.0 + 10 ** ((ra[j] - rd[i]) / 400.0))
            rd[i] += k * (win - e)
            ra[j] -= k * (win - e)
        res_d += rd / orders
        res_a += ra / orders
    return res_d, res_a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--glob", default="sp_*_direct_noisy_s*.pt")
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--out", default="results/selfplay_eval.json")
    a = p.parse_args()

    files = sorted(pathlib.Path(a.runs).glob(a.glob))
    d_states, a_states, labels = [], [], []
    mode = perfect = None
    for f in files:
        ck = torch.load(f, weights_only=False)
        mode, perfect = ck["args"]["defender_mode"], ck["args"]["sensing"] == "perfect"
        for c, s in enumerate(ck["ckpts"]):
            if (c + 1) % a.every == 0:
                d_states.append(s["defender"])
                a_states.append(s["attacker"])
                labels.append({"run": f.stem, "method": ck["args"]["method"], "seed": ck["args"]["seed"],
                               "chunk": c})
    print(f"{len(files)} runs, {len(labels)} checkpoints per side")
    score = crossplay(agents_from(d_states, DEF_OBS_DIM), agents_from(a_states, ATT_OBS_DIM), mode, perfect)
    de, ae = elo(score)
    # Final defender of each run against the scripted attackers of experiment 1 (same fixed sensing
    # condition as there): does training against a learning opponent generalise to new behaviours?
    scripted = {}
    for f in files:
        ck = torch.load(f, weights_only=False)
        agent = Agent.from_state(ck["ckpts"][-1]["defender"], DEF_OBS_DIM)
        scripted[f.stem] = {k: episodes(mode, agent, **condition(attacker=k)) for k in ATTACKERS}
        print(f.stem, {k: round(v["intercept"], 3) for k, v in scripted[f.stem].items()}, flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps({"labels": labels, "score": score.tolist(), "scripted": scripted,
                                               "defender_elo": de.tolist(), "attacker_elo": ae.tolist()}))
    for m in ("ippo", "sippo"):
        idx = [i for i, l in enumerate(labels) if l["method"] == m]
        if idx:
            print(m, "defender Elo (final ckpts):",
                  np.round([de[i] for i in idx if labels[i]["chunk"] == max(l["chunk"] for l in labels)], 0),
                  "mean score vs all attackers:", round(float(score[idx].mean()), 3))


if __name__ == "__main__":
    main()
