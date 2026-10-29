"""Vectorized 3D counter-drone arena.

One interceptor drone defends a ground asset (at the origin) against one attacker drone.
Drones are point masses commanded in acceleration, as if an inner attitude/thrust loop
tracks the command with a first-order lag. All environments step together in numpy.

The defender does not see the attacker directly: it gets noisy, delayed, sometimes missing
position measurements, and an alpha-beta tracker turns them into position/velocity estimates.
Sensor and actuator parameters are re-sampled every episode (domain randomisation).
"""
from dataclasses import dataclass, replace

import numpy as np

RUNNING, INTERCEPT, LEAK, CRASH, TIMEOUT = 0, 1, 2, 3, 4
OUTCOMES = {INTERCEPT: "intercept", LEAK: "leak", CRASH: "crash", TIMEOUT: "timeout"}

DEF_OBS_DIM = 19
ATT_OBS_DIM = 12
PRIV_DIM = 19


@dataclass
class ArenaConfig:
    n_envs: int = 512
    dt: float = 0.05
    max_steps: int = 300
    capture_radius: float = 1.0
    target_radius: float = 2.0
    arena_radius: float = 120.0
    def_vmax: float = 14.0
    def_amax: float = 25.0
    att_vmax: float = 10.0
    att_amax: float = 15.0
    spawn_dist: tuple = (40.0, 55.0)
    spawn_alt: tuple = (10.0, 25.0)
    # Ranges sampled uniformly per episode. Noise std is in metres at 20 m range.
    noise_std: tuple = (0.0, 1.0)
    delay_steps: tuple = (0, 3)
    dropout: tuple = (0.0, 0.2)
    lag_tau: tuple = (0.02, 0.10)
    perfect_info: bool = False
    # Attacker behaviour: jink | straight | evader | policy
    attacker: str = "jink"
    jink_amp: tuple = (0.3, 1.0)
    threat_radius: float = 15.0
    # How the defender action is used: residual (PN + correction) | direct | base (PN only) | pursuit
    defender_mode: str = "residual"
    residual_scale: float = 0.5
    pn_gain: float = 4.0
    tracker_alpha: float = 0.3
    tracker_beta: float = 0.25
    shaping: float = 0.1
    att_timeout_penalty: float = -5.0  # v2 self-play uses -10: stalling must not beat attacking
    seed: int = 0


def _norm(x):
    return np.linalg.norm(x, axis=-1)


def _unit(x):
    return x / (_norm(x)[..., None] + 1e-6)


def _clip_norm(x, max_norm=1.0):
    n = _norm(x)[..., None]
    return np.where(n > max_norm, x * (max_norm / (n + 1e-9)), x)


def _segment_min_dist(r0, r1):
    """Closest approach between two points moving linearly over one step (no tunnelling)."""
    d = r1 - r0
    t = np.clip(-(r0 * d).sum(-1) / ((d * d).sum(-1) + 1e-9), 0.0, 1.0)
    return _norm(r0 + t[:, None] * d)


def guidance_pn(r, v, gain, amax=25.0):
    """Proportional navigation plus a pursuit term, normalised to unit max acceleration.

    r: attacker position relative to defender, v: attacker velocity relative to defender.
    PN accelerates across the line of sight to null its rotation rate; the pursuit term
    builds closing speed from a standstill.
    """
    dist = _norm(r)[:, None] + 1e-6
    rhat = r / dist
    vc = np.maximum(-(r * v).sum(-1, keepdims=True) / dist, 1.0)
    v_perp = v - (v * rhat).sum(-1, keepdims=True) * rhat
    a_pn = gain * vc * v_perp / dist
    return _clip_norm(a_pn / amax + rhat)


def guidance_pursuit(r):
    return _unit(r)


class InterceptArena:
    def __init__(self, cfg: ArenaConfig):
        self.cfg = cfg
        self.n = cfg.n_envs
        self.rng = np.random.default_rng(cfg.seed)
        self.max_delay = max(1, int(cfg.delay_steps[1]))
        n, f = self.n, np.float32
        self.pd, self.vd, self.ad = (np.zeros((n, 3), f) for _ in range(3))
        self.pa, self.va, self.aa = (np.zeros((n, 3), f) for _ in range(3))
        self.xh, self.vh = np.zeros((n, 3), f), np.zeros((n, 3), f)
        self.hist = np.zeros((n, self.max_delay + 1, 3), f)
        self.t = np.zeros(n, np.int64)
        self.since = np.zeros(n, f)
        self.base = np.zeros((n, 3), f)
        self.sigma = np.zeros(n, f)
        self.delay = np.zeros(n, np.int64)
        self.pdrop = np.zeros(n, f)
        self.tau = np.zeros(n, f)
        self.amp = np.zeros(n, f)
        self.w = np.zeros((n, 2), f)
        self.ph = np.zeros((n, 2), f)
        self.d_def = np.zeros(n, f)
        self.d_att = np.zeros(n, f)
        self.reset_idx(np.arange(n))

    # ------------------------------------------------------------------ reset
    def reset_idx(self, idx):
        c, rng, k = self.cfg, self.rng, len(idx)
        if k == 0:
            return
        ang = rng.uniform(0, 2 * np.pi, k)
        rad = rng.uniform(*c.spawn_dist, k)
        self.pa[idx] = np.stack([rad * np.cos(ang), rad * np.sin(ang), rng.uniform(*c.spawn_alt, k)], -1)
        self.va[idx] = 0.0
        self.aa[idx] = 0.0
        da = rng.uniform(0, 2 * np.pi, k)
        dr = rng.uniform(0, 3, k)
        self.pd[idx] = np.stack([dr * np.cos(da), dr * np.sin(da), rng.uniform(1, 3, k)], -1)
        self.vd[idx] = 0.0
        self.ad[idx] = 0.0
        self.t[idx] = 0
        self.sigma[idx] = rng.uniform(*c.noise_std, k)
        self.delay[idx] = rng.integers(c.delay_steps[0], c.delay_steps[1] + 1, k)
        self.pdrop[idx] = rng.uniform(*c.dropout, k)
        self.tau[idx] = rng.uniform(*c.lag_tau, k)
        self.amp[idx] = rng.uniform(*c.jink_amp, k)
        self.w[idx] = rng.uniform(1.0, 4.0, (k, 2))
        self.ph[idx] = rng.uniform(0, 2 * np.pi, (k, 2))
        rel = self.pa[idx] - self.pd[idx]
        self.hist[idx] = rel[:, None, :]
        self.xh[idx] = rel + self._noise(idx, rel)
        self.vh[idx] = 0.0
        self.since[idx] = 0.0
        self.d_def[idx] = _norm(rel)
        self.d_att[idx] = _norm(self.pa[idx])
        self._update_base(idx)

    def reset(self):
        self.reset_idx(np.arange(self.n))
        return self.obs_defender(), self.obs_attacker()

    # ----------------------------------------------------------------- sensing
    def _noise(self, idx, rel):
        scale = self.sigma[idx] * np.maximum(1.0, _norm(rel) / 20.0)
        return (self.rng.standard_normal((len(idx), 3)) * scale[:, None]).astype(np.float32)

    def _sense(self):
        """Alpha-beta tracker on delayed, noisy, intermittent relative-position measurements."""
        c, dt = self.cfg, self.cfg.dt
        rel_true = self.pa - self.pd
        if c.perfect_info:
            self.xh[:] = rel_true
            self.vh[:] = self.va - self.vd
            self.since[:] = 0.0
            return
        L = self.max_delay + 1
        self.hist[np.arange(self.n), self.t % L] = rel_true
        meas_idx = (self.t - self.delay) % L
        z = self.hist[np.arange(self.n), meas_idx]
        z = z + self._noise(np.arange(self.n), z)
        got = self.rng.random(self.n) >= self.pdrop
        self.xh += self.vh * dt
        res = (z - self.xh) * got[:, None]
        self.xh += c.tracker_alpha * res
        self.vh += (c.tracker_beta / dt) * res
        self.since = np.where(got, 0.0, self.since + dt).astype(np.float32)

    def _update_base(self, idx=None):
        idx = np.arange(self.n) if idx is None else idx
        mode = self.cfg.defender_mode
        if mode == "pursuit":
            self.base[idx] = guidance_pursuit(self.xh[idx])
        else:
            self.base[idx] = guidance_pn(self.xh[idx], self.vh[idx], self.cfg.pn_gain, self.cfg.def_amax)

    # ------------------------------------------------------------ observations
    def obs_defender(self):
        return np.concatenate([
            self.xh / 30.0, self.vh / 20.0, self.vd / 14.0, self.pd / 30.0,
            (self.pd + self.xh) / 30.0, self.base, np.minimum(self.since, 1.0)[:, None],
        ], -1).astype(np.float32)

    def privileged_defender(self):
        """True state plus hidden simulator parameters. Only the critic sees this, during training."""
        return np.concatenate([
            (self.pa - self.pd) / 30.0, (self.va - self.vd) / 20.0, self.va / 10.0, self.pa / 30.0,
            self.aa / self.cfg.att_amax, self.sigma[:, None], self.delay[:, None] / 4.0,
            self.pdrop[:, None] * 5.0, self.tau[:, None] * 10.0,
        ], -1).astype(np.float32)

    def obs_attacker(self):
        return np.concatenate([
            self.pa / 30.0, self.va / 10.0, (self.pd - self.pa) / 30.0, (self.vd - self.va) / 20.0,
        ], -1).astype(np.float32)

    # ---------------------------------------------------------------- attacker
    def _scripted_attacker(self):
        c = self.cfg
        u = _unit(-self.pa)
        e1 = np.cross(u, np.array([0.0, 0.0, 1.0], np.float32))
        e1 = np.where(_norm(e1)[:, None] < 1e-3, np.array([1.0, 0.0, 0.0], np.float32), e1)
        e1 = _unit(e1)
        e2 = np.cross(e1, u)
        rel_def = self.pa - self.pd
        threat = (_norm(rel_def) < c.threat_radius)[:, None]
        if c.attacker == "straight":
            a = u
        elif c.attacker == "evader":
            away = rel_def - (rel_def * u).sum(-1, keepdims=True) * u
            a = u + np.where(threat, 1.5 * _unit(away), 0.0)
        else:
            ts = self.t[:, None] * c.dt
            s = np.sin(self.w * ts + self.ph)
            lat = self.amp[:, None] * (s[:, :1] * e1 + s[:, 1:] * e2)
            a = u + lat * np.where(threat, 2.5, 1.0)
        return _clip_norm(a)

    # -------------------------------------------------------------------- step
    def step(self, def_action, att_action=None):
        c, dt = self.cfg, self.cfg.dt
        def_action = np.asarray(def_action, np.float32)
        mode = c.defender_mode
        if mode == "residual":
            cmd = self.base + c.residual_scale * def_action
        elif mode == "direct":
            cmd = def_action
        else:
            cmd = self.base
        cmd = _clip_norm(np.clip(cmd, -1.0, 1.0)) * c.def_amax

        if c.attacker == "policy":
            acmd = _clip_norm(np.clip(np.asarray(att_action, np.float32), -1.0, 1.0))
        else:
            acmd = self._scripted_attacker()
        acmd = acmd * c.att_amax

        # First-order actuator lag, integrate, saturate speed.
        self.ad += (cmd - self.ad) * np.minimum(dt / self.tau, 1.0)[:, None]
        self.aa += (acmd - self.aa) * min(dt / 0.05, 1.0)
        rel0 = self.pa - self.pd
        att0 = self.pa.copy()
        self.vd = _clip_norm(self.vd + self.ad * dt, c.def_vmax)
        self.va = _clip_norm(self.va + self.aa * dt, c.att_vmax)
        self.pd = self.pd + self.vd * dt
        self.pa = self.pa + self.va * dt
        self.pa[:, 2] = np.maximum(self.pa[:, 2], 0.3)
        self.t += 1

        rel1 = self.pa - self.pd
        miss = _segment_min_dist(rel0, rel1)
        to_target = _segment_min_dist(att0, self.pa)
        d_def, d_att = _norm(rel1), _norm(self.pa)

        outcome = np.full(self.n, RUNNING)
        outcome[self.t >= c.max_steps] = TIMEOUT
        outcome[(self.pd[:, 2] < 0) | (_norm(self.pd[:, :2]) > c.arena_radius)] = CRASH
        outcome[to_target < c.target_radius] = LEAK
        outcome[miss < c.capture_radius] = INTERCEPT

        r_def = c.shaping * (self.d_def - d_def) - 0.01 - 0.001 * (def_action ** 2).sum(-1)
        r_def += np.select([outcome == INTERCEPT, (outcome == LEAK) | (outcome == CRASH)], [10.0, -10.0], 0.0)
        r_att = c.shaping * (self.d_att - d_att)
        r_att += np.select([outcome == LEAK, outcome == INTERCEPT, outcome == TIMEOUT], [10.0, -10.0, c.att_timeout_penalty], 0.0)
        self.d_def, self.d_att = d_def.astype(np.float32), d_att.astype(np.float32)

        info = {"outcome": outcome.copy(), "length": self.t.copy(), "miss": miss}
        done = outcome != RUNNING
        self._sense()
        self._update_base()
        self.reset_idx(np.flatnonzero(done))
        return (self.obs_defender(), self.obs_attacker(), r_def.astype(np.float32),
                r_att.astype(np.float32), done, info)


def make(**overrides):
    return InterceptArena(replace(ArenaConfig(), **overrides))
