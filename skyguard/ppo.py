"""PPO with the network and hyperparameters of Hansen, Louette, Leroy & Ernst (2025):
separate actor/critic MLPs (256-128, ELU, orthogonal init), Gaussian policy with a
state-independent log std, clip 0.2, value clipping, entropy 0.001, KL-adaptive learning rate.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    lr: float = 3e-4
    lr_max: float = 1e-3
    kl_target: float = 0.008
    entropy: float = 0.001
    value_coef: float = 1.0
    value_clip: float = 5.0
    max_grad_norm: float = 1.0
    init_log_std: float = -0.5
    actor_out_gain: float = 0.01


def _mlp(inp, out, out_gain):
    layers, dims = [], [inp, 256, 128]
    for a, b in zip(dims[:-1], dims[1:]):
        lin = nn.Linear(a, b)
        nn.init.orthogonal_(lin.weight, np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
    head = nn.Linear(dims[-1], out)
    nn.init.orthogonal_(head.weight, out_gain)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)


class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg: PPOConfig, critic_dim=None):
        super().__init__()
        self.actor = _mlp(obs_dim, act_dim, cfg.actor_out_gain)
        self.critic = _mlp(critic_dim or obs_dim, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.init_log_std))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())


class Agent:
    """PPO agent. With priv_dim > 0 the critic is asymmetric: it reads the actor's observation
    (which holds the tracker state, i.e. the actor's memory) plus privileged simulator state."""

    def __init__(self, obs_dim, act_dim, cfg: PPOConfig = PPOConfig(), priv_dim=0):
        self.cfg = cfg
        self.priv_dim = priv_dim
        self.model = ActorCritic(obs_dim, act_dim, cfg, obs_dim + priv_dim)
        self.norm = RunningNorm(obs_dim)
        self.pnorm = RunningNorm(priv_dim) if priv_dim else None
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.lr = cfg.lr

    def _critic_input(self, o, priv, update_norm):
        if not self.priv_dim:
            return o
        if priv is None:  # acting only (evaluation): the critic output is unused
            return np.concatenate([o, np.zeros((len(o), self.priv_dim), np.float32)], -1)
        if update_norm:
            self.pnorm.update(priv)
        return np.concatenate([o, self.pnorm(priv)], -1)

    @torch.no_grad()
    def act(self, obs, priv=None, deterministic=False, update_norm=True):
        """Returns action, log-prob, value, normalised obs, critic input."""
        if update_norm:
            self.norm.update(obs)
        o = self.norm(obs)
        c = self._critic_input(o, priv, update_norm)
        d = self.model.dist(torch.from_numpy(o))
        a = d.mean if deterministic else d.sample()
        v = self.model.critic(torch.from_numpy(c)).squeeze(-1)
        return a.numpy(), d.log_prob(a).sum(-1).numpy(), v.numpy(), o, c

    @torch.no_grad()
    def value(self, obs, priv=None):
        c = self._critic_input(self.norm(obs), priv, False)
        return self.model.critic(torch.from_numpy(c)).squeeze(-1).numpy()

    def update(self, buf, last_value):
        """buf: dict of arrays shaped [T, N, ...] with obs, act, logp, val, rew, done."""
        c = self.cfg
        T = len(buf["rew"])
        adv = np.zeros_like(buf["rew"])
        gae, nxt = 0.0, last_value
        for t in reversed(range(T)):
            nonterm = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + c.gamma * nxt * nonterm - buf["val"][t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t] = gae
            nxt = buf["val"][t]
        ret = adv + buf["val"]
        flat = lambda x: torch.from_numpy(np.ascontiguousarray(x.reshape(T * x.shape[1], *x.shape[2:])))
        obs, cobs, act = flat(buf["obs"]), flat(buf["cobs"]), flat(buf["act"])
        logp0, val0 = flat(buf["logp"]), flat(buf["val"])
        adv_t, ret_t = flat(adv), flat(ret)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n = len(obs)
        mb = n // c.minibatches
        stats = []
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for i in range(c.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                d = self.model.dist(obs[j])
                logp = d.log_prob(act[j]).sum(-1)
                ratio = (logp - logp0[j]).exp()
                pg = -torch.min(ratio * adv_t[j], ratio.clamp(1 - c.clip, 1 + c.clip) * adv_t[j]).mean()
                v = self.model.critic(cobs[j]).squeeze(-1)
                v_clip = val0[j] + (v - val0[j]).clamp(-c.value_clip, c.value_clip)
                vloss = torch.max((v - ret_t[j]) ** 2, (v_clip - ret_t[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + c.value_coef * vloss - c.entropy * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = (logp0[j] - logp).mean().item()
                if kl > 2 * c.kl_target:
                    self.lr = max(1e-5, self.lr / 1.5)
                elif kl < c.kl_target / 2:
                    self.lr = min(c.lr_max, self.lr * 1.5)
                for g in self.opt.param_groups:
                    g["lr"] = self.lr
                stats.append((pg.item(), vloss.item(), kl))
        return np.mean(stats, 0)

    def state(self):
        return {"model": {k: v.clone() for k, v in self.model.state_dict().items()},
                "norm": (self.norm.mean.copy(), self.norm.var.copy(), self.norm.count),
                "priv_dim": self.priv_dim,
                "pnorm": (self.pnorm.mean.copy(), self.pnorm.var.copy(), self.pnorm.count) if self.priv_dim else None}

    def load(self, state):
        self.model.load_state_dict(state["model"])
        self.norm.mean, self.norm.var, self.norm.count = (np.copy(state["norm"][0]),
                                                          np.copy(state["norm"][1]), state["norm"][2])
        if self.priv_dim:
            self.pnorm.mean, self.pnorm.var, self.pnorm.count = state["pnorm"]

    @classmethod
    def from_state(cls, state, obs_dim, act_dim=3):
        agent = cls(obs_dim, act_dim, PPOConfig(), priv_dim=state.get("priv_dim", 0))
        agent.load(state)
        return agent


class Rollout:
    def __init__(self, T, N, obs_dim, act_dim, critic_dim=None):
        f = np.float32
        self.b = {"obs": np.zeros((T, N, obs_dim), f), "cobs": np.zeros((T, N, critic_dim or obs_dim), f),
                  "act": np.zeros((T, N, act_dim), f),
                  "logp": np.zeros((T, N), f), "val": np.zeros((T, N), f),
                  "rew": np.zeros((T, N), f), "done": np.zeros((T, N), f)}

    def put(self, t, obs, cobs, act, logp, val, rew, done):
        for k, v in zip(("obs", "cobs", "act", "logp", "val", "rew", "done"), (obs, cobs, act, logp, val, rew, done)):
            self.b[k][t] = v
