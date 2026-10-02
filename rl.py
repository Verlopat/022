"""Deep reinforcement learning baseline DQN-C.

Independent double deep Q-learning with parameter sharing across the three junctions.
Every junction selects the arm to serve next (acyclic phase selection) at the end of the
minimum green and then every `delta` seconds. The observation is built from camera frames
(same detector profiles as the other camera controllers); the reward uses the simulator's
true queue during training only, as is standard for simulation-trained signal controllers.

Usage:
  python3 rl.py train <delta> <episodes> <seed> <tag>
  python3 rl.py select <tag>            # pick the checkpoint with the lowest tuning-seed delay
"""
import json
import os
import pickle
import sys
import time

import numpy as np

import simcore as sc
from simcore import G_MIN, snapshot

OBS = 52
NA = 4
Q_NORM = 40.0


class QNet:
    """Two hidden ReLU layers, manual backpropagation, Adam."""

    def __init__(self, nin=OBS, h=128, nout=NA, seed=0):
        rng = np.random.default_rng(seed)

        def init(fi, fo):
            lim = np.sqrt(6.0 / (fi + fo))
            return rng.uniform(-lim, lim, (fi, fo))
        self.P = [init(nin, h), np.zeros(h), init(h, h), np.zeros(h), init(h, nout), np.zeros(nout)]
        self.m = [np.zeros_like(p) for p in self.P]
        self.v = [np.zeros_like(p) for p in self.P]
        self.t = 0

    def forward(self, X):
        W1, b1, W2, b2, W3, b3 = self.P
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ W2 + b2
        a2 = np.maximum(z2, 0.0)
        return a2 @ W3 + b3, (X, z1, a1, z2, a2)

    def q(self, X):
        return self.forward(X)[0]

    def update(self, X, acts, targets, lr, clip=10.0, kappa=1.0):
        q, (X, z1, a1, z2, a2) = self.forward(X)
        B = len(X)
        ar = np.arange(B)
        r = q[ar, acts] - targets
        g = np.where(np.abs(r) <= kappa, r, kappa * np.sign(r)) / B
        dq = np.zeros_like(q)
        dq[ar, acts] = g
        W1, b1, W2, b2, W3, b3 = self.P
        gW3 = a2.T @ dq
        gb3 = dq.sum(0)
        dz2 = (dq @ W3.T) * (z2 > 0)
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(0)
        dz1 = (dz2 @ W2.T) * (z1 > 0)
        gW1 = X.T @ dz1
        gb1 = dz1.sum(0)
        grads = [gW1, gb1, gW2, gb2, gW3, gb3]
        norm = np.sqrt(sum(float((x ** 2).sum()) for x in grads))
        if norm > clip:
            grads = [x * (clip / norm) for x in grads]
        self.t += 1
        for i, (p, gr) in enumerate(zip(self.P, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * gr
            self.v[i] = 0.999 * self.v[i] + 0.001 * gr * gr
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        a = np.abs(r)
        return float(np.where(a <= kappa, 0.5 * r * r, kappa * (a - 0.5 * kappa)).mean())

    def copy_from(self, other):
        self.P = [p.copy() for p in other.P]

    def get_state(self):
        return [p.copy() for p in self.P]

    def set_state(self, st):
        self.P = [p.copy() for p in st]

    def n_params(self):
        return int(sum(p.size for p in self.P))


class Learner:
    def __init__(self, qnet, cap=100000, bs=64, gamma=0.99, lr=5e-4, sync=500, start=2000, seed=0, every=2):
        # gamma is the discount per second; transitions store their duration in seconds (semi-MDP)
        self.q = qnet
        self.tq = QNet(nin=qnet.P[0].shape[0])
        self.tq.copy_from(qnet)
        self.cap, self.bs, self.gamma, self.lr, self.sync, self.start = cap, bs, gamma, lr, sync, start
        nin = qnet.P[0].shape[0]
        self.S = np.zeros((cap, nin))
        self.A = np.zeros(cap, dtype=int)
        self.R = np.zeros(cap)
        self.S2 = np.zeros((cap, nin))
        self.K = np.zeros(cap)
        self.size = 0
        self.ptr = 0
        self.updates = 0
        self.rng = np.random.default_rng([seed, 5])
        self.losses = []
        self.every = every
        self.n_push = 0

    def push(self, s, a, r, s2, k):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.K[i] = s, a, r, s2, k
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        self.n_push += 1
        if self.size >= self.start and self.n_push % self.every == 0:
            self.train_step()

    def train_step(self):
        idx = self.rng.integers(0, self.size, self.bs)
        S, A, R, S2, K = self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.K[idx]
        a2 = np.argmax(self.q.q(S2), axis=1)
        qt = self.tq.q(S2)[np.arange(self.bs), a2]
        y = R + (self.gamma ** K) * qt
        self.losses.append(self.q.update(S, A, y, self.lr))
        self.updates += 1
        if self.updates % self.sync == 0:
            self.tq.copy_from(self.q)


class DQNController:
    """Acyclic phase selection by a shared Q-network; frames of the junction's four arms and of its
    downstream internal arms at every decision."""
    cyclic = False
    name = 'DQN-C'

    def __init__(self, profile, qnet, delta=5, gmax=90, eps=0.0, learner=None, seed=0, alpha=0.5, gmin=G_MIN,
                 reward='queue', oracle=False, obs_v1=False, backlog=True):
        self.oracle = oracle
        self.obs_v1 = obs_v1
        self.backlog = backlog
        self.alpha = alpha
        self.gmin = gmin
        self.reward = reward
        self.prev_scores = [None, None, None]
        self.prof = profile
        self.qnet = qnet
        self.delta = delta
        self.gmax = gmax
        self.eps = eps
        self.learner = learner
        self.rng = np.random.default_rng([seed, 17])
        self.pending = [None, None, None]
        self.last = [None, None, None]
        self.decisions = 0

    def reset(self, sim):
        self.pending = [None, None, None]
        self.last = [None, None, None]
        self.prev_scores = [None, None, None]

    def junction_queue(self, sim, j):
        """True waiting demand of junction j (PCU): lane queues plus, unless disabled, vehicles held at the
        boundary entry."""
        q = 0.0
        for d in range(4):
            a = 4 * j + d
            q += sim.lane_pcu[a][0] + sim.lane_pcu[a][1]
            if self.backlog:
                buf = sim.entry_buf.get(a)
                if buf:
                    q += sum(sim.v_w[v] for v in buf)
        return q

    @staticmethod
    def arm_queue(sim, a):
        q = sim.lane_pcu[a][0] + sim.lane_pcu[a][1]
        buf = sim.entry_buf.get(a)
        if buf:
            q += sum(sim.v_w[v] for v in buf)
        return q

    def pressure(self, sim, j):
        """Junction pressure (PressLight form): inbound queue minus turning-weighted receiving-arm queue."""
        p = 0.0
        net = sim.net
        for d in range(4):
            a = 4 * j + d
            p += self.arm_queue(sim, a)
            for m, pm in zip(('L', 'T', 'R'), net.turn[a]):
                typ, tgt = net.dest[a][m]
                if typ == 'app':
                    p -= pm * self.arm_queue(sim, tgt)
        return abs(p)

    def reward_queue(self, sim, j):
        if self.reward == 'pressure':
            return self.pressure(sim, j)
        q = self.junction_queue(sim, j)
        for n in (j - 1, j + 1):
            if 0 <= n < 3:
                q += self.alpha * self.junction_queue(sim, n)
        return q

    def zone_state(self, sim, a, t):
        """Camera check of the 20 m stop-line zone of arm a (as in the tactical layer)."""
        present = 0.0
        for lane in (0, 1):
            pos = 0.0
            for vid in sim.lanes[a][lane]:
                if pos >= 20.0:
                    break
                if sim.rng_p.random() < self.prof.recall[sim.v_cls[vid]] * self.prof.recall_mult:
                    present = 1.0
                    break
                pos += sim.v_space[vid] * sc.CAR_SPACE_M
            if present:
                break
        return present

    def obs(self, sim, j, t):
        f = []
        s = sim.sig[j]
        scores = []
        for d in range(4):
            a = 4 * j + d
            sn = snapshot(sim, a, t, self.prof, sim.rng_p)
            scores.append(sn['score'])
            gnow = 1.0 if (s.cur == d and s.stage == 'G') else 0.0
            lg = sim.last_green_end[a]
            since = 0.0 if gnow else (min(300.0, t - lg) if lg > 0 else 150.0)
            g = sn['groups']
            f += [sn['score'] / 15.0, g[0] / 10.0, g[1] / 10.0, g[2] / 10.0, g[3] / 5.0, sn['fill'], gnow,
                  since / 150.0]
        downs = [4 * (j + 1) + 0 if j < 2 else None, 4 * (j - 1) + 2 if j > 0 else None]
        for tgt in downs:
            if tgt is None:
                f += [0.0, 0.0, 0.0]
            else:
                sn = snapshot(sim, tgt, t, self.prof, sim.rng_p)
                f += [1.0, sn['score'] / 15.0, sn['fill']]
        prev = self.prev_scores[j] if self.prev_scores[j] is not None else scores
        f += [(x - y) / 15.0 for x, y in zip(scores, prev)]
        self.prev_scores[j] = scores
        ph = [0.0] * 4
        if s.cur >= 0:
            ph[s.cur] = 1.0
        el = (t - s.g_start) / 60.0 if s.stage == 'G' else 0.0
        if s.stage == 'G' and s.cur >= 0:
            a = 4 * j + s.cur
            last = max(sim.last_dis_time[a], s.g_start + sc.STARTUP_LOST)
            since_cross = min(30.0, max(0.0, t - last)) / 10.0
            zone = 0.0 if self.obs_v1 else self.zone_state(sim, a, t)
        else:
            since_cross, zone = 0.0, 0.0
        jj = [0.0] * 3
        jj[j] = 1.0
        return np.array(f + ph + [el, since_cross, zone] + jj)

    def obs_oracle(self, sim, j, t):
        """Exact simulator state in the same 52-dimensional layout (reference variant DQN-O)."""
        f = []
        s = sim.sig[j]
        qs = []
        net = sim.net
        for d in range(4):
            a = 4 * j + d
            q = sim.lane_pcu[a][0] + sim.lane_pcu[a][1]
            qs.append(q)
            mov = sum(sim.v_w[v] for _, v in sim.moving[a])
            buf = sim.entry_buf.get(a)
            back = sum(sim.v_w[v] for v in buf) if buf else 0.0
            gnow = 1.0 if (s.cur == d and s.stage == 'G') else 0.0
            lg = sim.last_green_end[a]
            since = 0.0 if gnow else (min(300.0, t - lg) if lg > 0 else 150.0)
            f += [q / 40.0, mov / 40.0, min(back, 400.0) / 40.0, sim.occ[a] / net.storage[a], 0.0, 0.0, gnow,
                  since / 150.0]
        downs = [4 * (j + 1) + 0 if j < 2 else None, 4 * (j - 1) + 2 if j > 0 else None]
        for tgt in downs:
            if tgt is None:
                f += [0.0, 0.0, 0.0]
            else:
                f += [1.0, (sim.lane_pcu[tgt][0] + sim.lane_pcu[tgt][1]) / 40.0, sim.occ[tgt] / net.storage[tgt]]
        prev = self.prev_scores[j] if self.prev_scores[j] is not None else qs
        f += [(x - y) / 40.0 for x, y in zip(qs, prev)]
        self.prev_scores[j] = qs
        ph = [0.0] * 4
        if s.cur >= 0:
            ph[s.cur] = 1.0
        el = (t - s.g_start) / 60.0 if s.stage == 'G' else 0.0
        if s.stage == 'G' and s.cur >= 0:
            a = 4 * j + s.cur
            last = max(sim.last_dis_time[a], s.g_start + sc.STARTUP_LOST)
            since_cross = min(30.0, max(0.0, t - last)) / 10.0
            zone = 1.0 if (sim.lanes[a][0] or sim.lanes[a][1]) else 0.0
        else:
            since_cross, zone = 0.0, 0.0
        jj = [0.0] * 3
        jj[j] = 1.0
        return np.array(f + ph + [el, since_cross, zone] + jj)

    def _decide(self, sim, j, t, forced):
        s = sim.sig[j]
        o = self.obs_oracle(sim, j, t) if self.oracle else self.obs(sim, j, t)
        if self.obs_v1:
            # basic 46-feature layout: arms, downstream arms, phase, elapsed green, junction indicator
            o = np.concatenate([o[0:38], o[42:46], o[46:47], o[49:52]])
        self.decisions += 1
        if self.learner is not None:
            qj = self.reward_queue(sim, j)
            if self.last[j] is not None:
                o0, a0, t0, q0 = self.last[j]
                dt = t - t0
                r = -0.5 * (q0 + qj) * dt / (5.0 * Q_NORM)
                self.learner.push(o0, a0, r, o, float(dt))
        if self.rng.random() < self.eps:
            act = int(self.rng.integers(0, 4))
            if forced and act == s.cur:
                act = (act + 1 + int(self.rng.integers(0, 3))) % 4
        else:
            qv = self.qnet.q(o[None])[0]
            if forced:
                qv[s.cur] = -1e9
            act = int(np.argmax(qv))
        if self.learner is not None:
            self.last[j] = (o, act, t, qj)
        return act

    def end_green(self, sim, j, t):
        s = sim.sig[j]
        el = t - s.g_start
        if el < self.gmin:
            return False
        forced = el >= self.gmax
        if not forced and (el - self.gmin) % self.delta != 0:
            return False
        act = self._decide(sim, j, t, forced)
        if act == s.cur:
            return False
        self.pending[j] = act
        return True

    def next_phase(self, sim, j, t):
        if self.pending[j] is not None:
            d = self.pending[j]
            self.pending[j] = None
            return d
        return (sim.sig[j].last_served + 1) % 4


def random_scenario(rng):
    scen = dict(name='train', factor=float(rng.uniform(0.5, 1.35)), eb=float(rng.uniform(0.6, 1.6)),
                wb=float(rng.uniform(0.6, 1.4)), incident=None, mix='hv' if rng.random() < 0.2 else 'base')
    if rng.random() < 0.3:
        a = int(rng.integers(0, 12))
        t0 = int(rng.uniform(900, 2400))
        scen['incident'] = (a, int(rng.integers(0, 2)), t0, t0 + int(rng.uniform(600, 1500)))
    return scen


def train(delta, episodes, seed, tag, eps_frac=0.5, eps_min=0.05, ckpt_every=25, alpha=0.5, lr=2.5e-4, gmin=G_MIN,
          reward=0.0, gamma=0.99, oracle=0.0, obs_v1=0.0, backlog=1.0):
    gmin = int(gmin)
    oracle = bool(oracle)
    obs_v1 = bool(obs_v1)
    backlog = bool(backlog)
    reward = 'pressure' if reward == 1.0 else 'queue'
    q = QNet(nin=46 if obs_v1 else OBS, seed=seed)
    L = Learner(q, seed=seed, lr=lr, gamma=gamma)
    profs = sc.make_profiles()
    os.makedirs('models', exist_ok=True)
    hist = []
    t0 = time.time()
    ckpts = {}
    for ep in range(episodes):
        eps = max(eps_min, 1.0 - (1.0 - eps_min) * ep / max(1, eps_frac * episodes))
        rng = np.random.default_rng([seed, ep, 7])
        scen = random_scenario(rng)
        pk = str(rng.choice(['P0', 'P1', 'P2', 'P3'], p=[0.10, 0.60, 0.15, 0.15]))
        ctrl = DQNController(profs[pk], q, delta=delta, eps=eps, learner=L, seed=seed * 100000 + ep, alpha=alpha,
                             gmin=gmin, reward=reward, oracle=oracle, obs_v1=obs_v1, backlog=backlog)
        sim = sc.Sim(scen, 30000 + ep, ctrl, perception=profs[pk])
        m = sim.run()
        nl = len(L.losses)
        hist.append(dict(episode=ep + 1, eps=eps, delay=m['delay'], factor=scen['factor'],
                         incident=scen['incident'] is not None, updates=L.updates,
                         loss=float(np.mean(L.losses[-2000:])) if nl else None, time=time.time() - t0))
        if (ep + 1) % ckpt_every == 0:
            ckpts[ep + 1] = q.get_state()
            print(f'{tag} ep {ep + 1} eps {eps:.2f} delay {m["delay"]:.1f} updates {L.updates} '
                  f'loss {hist[-1]["loss"]} ({time.time() - t0:.0f}s)', flush=True)
    with open(f'models/{tag}.pkl', 'wb') as f:
        pickle.dump(dict(delta=delta, seed=seed, ckpts=ckpts, final=q.get_state(), n_params=q.n_params(),
                         alpha=alpha, lr=lr, episodes=episodes, gmin=gmin, reward=reward, gamma=gamma,
                         oracle=oracle, obs_v1=obs_v1, backlog=backlog), f)
    with open(f'models/{tag}_hist.json', 'w') as f:
        json.dump(dict(hist=hist, delta=delta, seed=seed, train_time_s=time.time() - t0), f, indent=1)


def load_policy(tag, ckpt=None):
    d = pickle.load(open(f'models/{tag}.pkl', 'rb'))
    st = d['final'] if ckpt is None else d['ckpts'][ckpt]
    q = QNet(nin=st[0].shape[0])
    q.set_state(st)
    return q, d


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        extra = {}
        for kv in sys.argv[6:]:
            k, v = kv.split('=')
            extra[k] = float(v)
        train(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], **extra)
