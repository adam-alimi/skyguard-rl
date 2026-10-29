"""Experiment 1: train an interceptor against scripted jinking attackers under sensor noise.

  --mode residual   policy outputs a correction on top of proportional navigation (PN)
  --mode direct     policy outputs the full acceleration command (PPO from scratch)
  --dr full|none    domain randomisation of sensing/actuation, or nominal conditions only
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from skyguard.env import CRASH, DEF_OBS_DIM, INTERCEPT, PRIV_DIM, make
from skyguard.ppo import Agent, PPOConfig, Rollout

NOMINAL = dict(noise_std=(0.3, 0.3), delay_steps=(1, 1), dropout=(0.05, 0.05), lag_tau=(0.05, 0.05))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="residual", choices=["residual", "direct"])
    p.add_argument("--dr", default="full", choices=["full", "none"])
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--shaping", type=float, default=0.1)
    p.add_argument("--lr-max", type=float, default=1e-3)
    p.add_argument("--critic", default="sym", choices=["sym", "priv"])
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    env_kw = {} if a.dr == "full" else NOMINAL
    env = make(n_envs=a.envs, defender_mode=a.mode, attacker="jink", seed=a.seed, shaping=a.shaping, **env_kw)
    cfg = PPOConfig(init_log_std=-1.0 if a.mode == "residual" else -0.5, lr=min(3e-4, a.lr_max), lr_max=a.lr_max)
    priv = PRIV_DIM if a.critic == "priv" else 0
    agent = Agent(DEF_OBS_DIM, 3, cfg, priv_dim=priv)
    buf = Rollout(a.horizon, a.envs, DEF_OBS_DIM, 3, DEF_OBS_DIM + priv)
    name = f"guid_{a.mode}_dr-{a.dr}_sh{a.shaping:g}_{a.critic}_s{a.seed}"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    curve, t0 = [], time.time()
    for it in range(a.iters):
        wins = crashes = done_n = 0
        for t in range(a.horizon):
            act, logp, val, nobs, cobs = agent.act(obs, env.privileged_defender())
            obs, _, rew, _, done, info = env.step(act)
            buf.put(t, nobs, cobs, act, logp, val, rew, done)
            done_n += done.sum()
            wins += (info["outcome"][done] == INTERCEPT).sum()
            crashes += (info["outcome"][done] == CRASH).sum()
        stats = agent.update(buf.b, agent.value(obs, env.privileged_defender()))
        rate = wins / max(done_n, 1)
        steps = (it + 1) * a.envs * a.horizon
        curve.append({"iter": it, "steps": steps, "intercept": float(rate), "crash": float(crashes / max(done_n, 1))})
        if it % 10 == 0 or it == a.iters - 1:
            print(f"{name} it {it:4d} steps {steps/1e6:6.2f}M intercept {rate:.3f} "
                  f"crash {crashes / max(done_n, 1):.3f} kl {stats[2]:.4f} lr {agent.lr:.1e} {time.time()-t0:5.0f}s", flush=True)

    torch.save({"agent": agent.state(), "args": vars(a), "mode": a.mode, "dr": a.dr}, out / f"{name}.pt")
    (out / f"{name}.json").write_text(json.dumps(curve))


if __name__ == "__main__":
    main()
