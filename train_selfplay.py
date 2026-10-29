"""Experiment 2: attacker and defender learn against each other.

  --method ippo    both agents learn simultaneously with independent PPO
  --method sippo   asymmetric self-play (Hansen et al., 2025): at the start of each chunk, with
                   probability p the defender faces a frozen past attacker instead of the live one
Checkpoints of both agents are saved at the end of every chunk for cross-play / Elo evaluation.
"""
import argparse
import pathlib
import time

import numpy as np
import torch

from skyguard.env import ATT_OBS_DIM, DEF_OBS_DIM, INTERCEPT, LEAK, make
from skyguard.ppo import Agent, PPOConfig, Rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="sippo", choices=["ippo", "sippo"])
    p.add_argument("--defender-mode", default="direct", choices=["direct", "residual"])
    p.add_argument("--sensing", default="perfect", choices=["perfect", "noisy"])
    p.add_argument("--chunks", type=int, default=20)
    p.add_argument("--chunk-iters", type=int, default=10)
    p.add_argument("--p-frozen", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--att-timeout", type=float, default=-5.0, help="attacker reward on timeout")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(a.seed + 100)
    env = make(n_envs=a.envs, defender_mode=a.defender_mode, attacker="policy", seed=a.seed,
               perfect_info=a.sensing == "perfect", att_timeout_penalty=a.att_timeout)
    dcfg = PPOConfig(init_log_std=-1.0 if a.defender_mode == "residual" else -0.5)
    defender, attacker = Agent(DEF_OBS_DIM, 3, dcfg), Agent(ATT_OBS_DIM, 3, PPOConfig())
    frozen = Agent(ATT_OBS_DIM, 3, PPOConfig())
    dbuf = Rollout(a.horizon, a.envs, DEF_OBS_DIM, 3)
    abuf = Rollout(a.horizon, a.envs, ATT_OBS_DIM, 3)
    name = f"sp_{a.method}_{a.defender_mode}_{a.sensing}{'_v2' if a.att_timeout <= -10 else ''}_s{a.seed}"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pool, ckpts, log = [], [], []
    dobs, aobs = env.reset()
    t0 = time.time()
    p_frozen = a.p_frozen if a.method == "sippo" else 0.0
    for chunk in range(a.chunks):
        use_frozen = bool(pool) and rng.random() < p_frozen
        if use_frozen:
            frozen.load(pool[rng.integers(len(pool))])
        opp = frozen if use_frozen else attacker
        for _ in range(a.chunk_iters):
            wins = leaks = n_done = 0
            for t in range(a.horizon):
                da, dlp, dv, dno, dco = defender.act(dobs)
                aa, alp, av, ano, aco = opp.act(aobs, update_norm=not use_frozen)
                dobs, aobs, dr, ar, done, info = env.step(da, aa)
                dbuf.put(t, dno, dco, da, dlp, dv, dr, done)
                if not use_frozen:
                    abuf.put(t, ano, aco, aa, alp, av, ar, done)
                o = info["outcome"][done]
                n_done += done.sum()
                wins += (o == INTERCEPT).sum()
                leaks += (o == LEAK).sum()
            defender.update(dbuf.b, defender.value(dobs))
            if not use_frozen:
                attacker.update(abuf.b, attacker.value(aobs))
            log.append({"chunk": chunk, "frozen": use_frozen, "intercept": wins / max(n_done, 1),
                        "leak": leaks / max(n_done, 1)})
        pool.append(attacker.state())
        ckpts.append({"defender": defender.state(), "attacker": attacker.state()})
        last = log[-1]
        print(f"{name} chunk {chunk:3d} {'FROZEN' if use_frozen else 'live  '} intercept "
              f"{last['intercept']:.3f} leak {last['leak']:.3f} {time.time()-t0:5.0f}s", flush=True)
        # Save after every chunk so an interrupted run keeps its checkpoints.
        torch.save({"ckpts": ckpts, "log": log, "args": vars(a)}, out / f"{name}.pt")


if __name__ == "__main__":
    main()
