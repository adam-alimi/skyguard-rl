"""Evaluate experiment-1 policies and classical baselines across sensing conditions.

Each condition fixes sensor noise, delay and dropout (no randomisation) and runs one episode
per environment. Conditions beyond the training range test out-of-distribution robustness.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from skyguard.env import DEF_OBS_DIM, INTERCEPT, LEAK, CRASH, make
from skyguard.ppo import Agent

NOISE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
DELAY = [0, 2, 4, 6, 8]
ATTACKERS = ["straight", "jink", "evader"]


def episodes(mode, agent, n=2048, seed=123, **cond):
    env = make(n_envs=n, defender_mode=mode, seed=seed, max_steps=300, **cond)
    obs, _ = env.reset()
    result = np.zeros(n, np.int64)
    alive = np.ones(n, bool)
    for _ in range(env.cfg.max_steps):
        act = np.zeros((n, 3), np.float32)
        if agent is not None:
            act = agent.act(obs, deterministic=True, update_norm=False)[0]
        obs, _, _, _, done, info = env.step(act)
        first = done & alive
        result[first] = info["outcome"][first]
        alive &= ~done
        if not alive.any():
            break
    return {"intercept": float((result == INTERCEPT).mean()), "leak": float((result == LEAK).mean()),
            "crash": float((result == CRASH).mean())}


def condition(noise=0.5, delay=1, dropout=0.1, attacker="jink"):
    return dict(noise_std=(noise, noise), delay_steps=(delay, delay), dropout=(dropout, dropout),
                lag_tau=(0.05, 0.05), attacker=attacker)


def load(path):
    ck = torch.load(path, weights_only=False)
    return ck["mode"], Agent.from_state(ck["agent"], DEF_OBS_DIM)


def evaluate(mode, agent):
    return {
        "noise": [episodes(mode, agent, **condition(noise=s)) for s in NOISE],
        "delay": [episodes(mode, agent, **condition(delay=d)) for d in DELAY],
        "attacker": {k: episodes(mode, agent, **condition(attacker=k)) for k in ATTACKERS},
        "nominal": episodes(mode, agent, **condition()),
        "hard": episodes(mode, agent, **condition(noise=2.0, delay=4, dropout=0.3)),
        "train_dist": episodes(mode, agent, attacker="jink"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results/guidance_eval.json")
    a = p.parse_args()
    res = {"axes": {"noise": NOISE, "delay": DELAY, "attackers": ATTACKERS},
           "baselines": {"pursuit": evaluate("pursuit", None), "pn": evaluate("base", None)}, "policies": {}}
    for f in sorted(pathlib.Path(a.runs).glob("guid_*.pt")):
        mode, agent = load(f)
        res["policies"][f.stem] = evaluate(mode, agent)
        print(f.stem, res["policies"][f.stem]["nominal"], res["policies"][f.stem]["hard"], flush=True)
    for k, v in res["baselines"].items():
        print(k, v["nominal"], v["hard"])
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
