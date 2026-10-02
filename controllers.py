"""
Signal controllers for the three-junction corridor.

FT-W   : fixed-time plan from Webster's delay-minimising cycle and flow-ratio splits
ACT    : fully actuated, presence-mode stop-line detection (loops), gap-out
MP-C   : cyclic max-pressure (pressure-proportional splits, camera snapshots)
MP-A   : acyclic max-pressure (phase chosen every delta s, camera snapshots)
LPS    : local proportional split (the original lane-score controller)
GATC   : graph-attention coordinated controller (proposed): strategic plan from the
         GAT estimator, tactical camera gap-out within the cycle
All cycle-based controllers plan each junction when its previous cycle ends.
"""
import numpy as np

import simcore as sc
from simcore import (G_MIN, INTERGREEN, N_PHASE, L_CYCLE, STARTUP_LOST, W_DESIGN, MIX_BASE, SPACE,
                     CAR_SPACE_M, integer_split, snapshot, webster_plan)

NODE_F = 14
LANE_IDX = list(range(12))
JUNC_IDX = [12, 13, 14]
H_TARGET = 120.0          # prediction horizon (s)
Y_SCALE = 40.0            # normalisation of lane demand (PCU)
PCU_PER_SPACE = float((MIX_BASE * W_DESIGN).sum() / (MIX_BASE * SPACE).sum())
STOPLINE_ZONE = 20.0      # metres inspected by the tactical camera check


def build_mask(net, variant='full'):
    """Attention mask (N x N, True where p may attend to q)."""
    N = 15
    M = np.eye(N, dtype=bool)
    if variant == 'none':
        return M
    for j in range(3):
        lanes = [4 * j + d for d in range(4)]
        for p in lanes:
            for q in lanes:
                M[p, q] = True
            M[p, 12 + j] = True
            M[12 + j, p] = True
    if variant == 'full':
        for j in range(3):
            for k in range(3):
                M[12 + j, 12 + k] = True
        for tgt in range(12):
            for (a, m) in net.feeders[tgt]:
                M[tgt, a] = True
                M[a, tgt] = True
    return M


def order_phases(scores, last_served):
    order = list(np.argsort(-np.asarray(scores), kind='stable'))
    if len(order) > 1 and order[0] == last_served:
        order[0], order[1] = order[1], order[0]
    return order


def split_greens(weights, C):
    w = np.maximum(np.asarray(weights, dtype=float), 1e-6)
    total = C - INTERGREEN * N_PHASE
    extra = total - N_PHASE * G_MIN
    g = G_MIN + extra * w / w.sum()
    return integer_split(g, total)


class TacticalCamera:
    """Within-cycle camera check on the approach holding green (every 2 s after minimum green).

    Ends the phase when (a) the 20 m stop-line zone shows no queued vehicle in two
    consecutive checks, or (b) vehicles are present but none has crossed the stop line
    for 4 s (downstream blockage or incident)."""

    def __init__(self, profile, every=2, empty_needed=2):
        self.prof = profile
        self.every = every
        self.empty_needed = empty_needed
        self.empty = [0, 0, 0]
        self.phase_id = [None, None, None]

    def reset(self):
        self.empty = [0, 0, 0]
        self.phase_id = [None, None, None]

    def check(self, sim, j, t):
        s = sim.sig[j]
        a = 4 * j + s.cur
        if self.phase_id[j] != s.g_start:
            self.phase_id[j] = s.g_start
            self.empty[j] = 0
        sim.check_count = getattr(sim, 'check_count', 0) + 1
        present = False
        rng = sim.rng_p
        for lane in (0, 1):
            pos = 0.0
            for vid in sim.lanes[a][lane]:
                if pos >= STOPLINE_ZONE:
                    break
                c = sim.v_cls[vid]
                if rng.random() < self.prof.recall[c] * self.prof.recall_mult:
                    present = True
                    break
                pos += sim.v_space[vid] * CAR_SPACE_M
            if present:
                break
        if not present:
            self.empty[j] += 1
            return self.empty[j] >= self.empty_needed
        self.empty[j] = 0
        last = max(sim.last_dis_time[a], s.g_start + STARTUP_LOST)
        return (t - last) > 4.0


# ----------------------------------------------------------------------------
class FixedTimeWebster:
    """Webster plan. plan_scenario=None uses the true mean flows of the running scenario
    (idealised reference); a scenario key uses that design-day demand instead."""
    cyclic = True
    tactical = False
    name = 'FT-W'

    def __init__(self, c_max=160, plan_scenario=None):
        self.c_max = c_max
        self.plan_scenario = plan_scenario

    def reset(self, sim):
        if self.plan_scenario is None:
            qb, mix = sim.boundary_q, sim.mix
        else:
            qb, mix = sc.scenario_boundary_flows(sc.SCENARIOS[self.plan_scenario]), MIX_BASE
        C, plans, Ys = webster_plan(sim.net, qb, mix=mix, c_max=self.c_max)
        self.C = C
        self.plans = plans
        self.Ys = Ys

    def plan_junction(self, sim, j, t):
        return self.plans[j]


class Actuated:
    cyclic = False
    name = 'ACT'

    def __init__(self, gmin=G_MIN, gmax=90, gap=2.0):
        self.gmin, self.gmax, self.gap = gmin, gmax, gap

    def reset(self, sim):
        pass

    def end_green(self, sim, j, t):
        s = sim.sig[j]
        el = t - s.g_start
        if el < self.gmin:
            return False
        if el >= self.gmax:
            return True
        a = 4 * j + s.cur
        # presence-mode detection at the stop line: occupied while vehicles wait there
        if sim.lanes[a][0] or sim.lanes[a][1]:
            return False
        last = max(sim.lane_last[a][0], sim.lane_last[a][1], s.g_start + STARTUP_LOST)
        return (t - last) > self.gap

    def next_phase(self, sim, j, t):
        return (sim.sig[j].last_served + 1) % 4


def pressures(sim, j, snaps, net):
    P = np.zeros(4)
    for d in range(4):
        a = 4 * j + d
        down = 0.0
        for m, pm in zip(('L', 'T', 'R'), net.turn[a]):
            typ, tgt = net.dest[a][m]
            if typ == 'app':
                down += pm * snaps[tgt]['score']
        P[d] = snaps[a]['score'] - down
    return P


def _needed_for_pressure(net, j):
    need = set(4 * j + d for d in range(4))
    for d in range(4):
        a = 4 * j + d
        for m in ('L', 'T', 'R'):
            typ, tgt = net.dest[a][m]
            if typ == 'app':
                need.add(tgt)
    return need


class CyclicMaxPressure:
    cyclic = True
    tactical = False
    name = 'MP-C'

    def __init__(self, profile, C=120):
        self.prof = profile
        self.C = C

    def reset(self, sim):
        pass

    def plan_junction(self, sim, j, t):
        snaps = {a: snapshot(sim, a, t, self.prof, sim.rng_p) for a in _needed_for_pressure(sim.net, j)}
        P = pressures(sim, j, snaps, sim.net)
        w = np.maximum(P, 0.0) + 0.5
        g = split_greens(w, self.C)
        return [(d, int(g[d])) for d in (0, 1, 2, 3)]


class AcyclicMaxPressure:
    cyclic = False
    name = 'MP-A'

    def __init__(self, profile, delta=20, gmin=15, gmax=90):
        self.prof = profile
        self.delta, self.gmin, self.gmax = delta, gmin, gmax
        self.pending = [None, None, None]

    def reset(self, sim):
        self.pending = [None, None, None]

    def _pressure_now(self, sim, j, t):
        snaps = {a: snapshot(sim, a, t, self.prof, sim.rng_p) for a in _needed_for_pressure(sim.net, j)}
        return pressures(sim, j, snaps, sim.net)

    def end_green(self, sim, j, t):
        s = sim.sig[j]
        el = t - s.g_start
        if el < self.gmin:
            return False
        if el >= self.gmax:
            P = self._pressure_now(sim, j, t)
            P[s.cur] = -1e9
            self.pending[j] = int(np.argmax(P))
            return True
        if (el - self.gmin) % self.delta != 0:
            return False
        P = self._pressure_now(sim, j, t)
        best = int(np.argmax(P))
        if best != s.cur and P[best] > P[s.cur]:
            self.pending[j] = best
            return True
        return False

    def next_phase(self, sim, j, t):
        if self.pending[j] is not None:
            d = self.pending[j]
            self.pending[j] = None
            return d
        P = self._pressure_now(sim, j, t)
        return int(np.argmax(P))


class LocalProportional:
    """Original controller: fixed cycle, greens proportional to the PCU lane score,
    phases served in descending score order. Optional raw-count scoring and tactical layer."""
    cyclic = True
    name = 'LPS'

    def __init__(self, profile, C=120, priority=True, weights='pcu', tactical=False, check_every=2, empty_needed=2):
        self.prof = profile
        self.C = C
        self.priority = priority
        self.weights = weights
        self.tactical = tactical
        self.tc = TacticalCamera(profile, every=check_every, empty_needed=empty_needed)

    def reset(self, sim):
        self.tc.reset()

    def plan_junction(self, sim, j, t):
        snaps = [snapshot(sim, 4 * j + d, t, self.prof, sim.rng_p) for d in range(4)]
        sc_ = [s['score'] if self.weights == 'pcu' else float(s['n']) for s in snaps]
        w = np.array(sc_) + 0.5
        g = split_greens(w, self.C)
        order = order_phases(sc_, sim.sig[j].last_served) if self.priority else [0, 1, 2, 3]
        return [(d, int(g[d])) for d in order]

    def gap_out(self, sim, j, t):
        return self.tc.check(sim, j, t)


# ----------------------------------------------------------------------------
class FeatureBuilder:
    """Builds the 15-node feature matrix from camera snapshots and signal state."""

    def __init__(self, profile):
        self.prof = profile
        self.reset()

    def reset(self):
        self.prev_score = np.full(12, np.nan)

    def build(self, sim, t):
        snaps = [snapshot(sim, a, t, self.prof, sim.rng_p) for a in range(12)]
        X = np.zeros((15, NODE_F))
        scores = np.array([s['score'] for s in snaps])
        prev = np.where(np.isnan(self.prev_score), scores, self.prev_score)
        dscore = scores - prev
        for a in range(12):
            j, d = divmod(a, 4)
            s = snaps[a]
            sig = sim.sig[j]
            gsum = sum(sig.green_given)
            gshare = sig.green_given[d] / gsum if gsum > 0 else 0.25
            Cj = sig.cycle_len if sig.cycle_len > 0 else 120.0
            green_now = 1.0 if (sig.cur == d and sig.stage == 'G') else 0.0
            if green_now:
                since = 0.0
            elif sig.cur == d:
                since = max(0.0, t - sig.y_start)
            else:
                since = sim.last_green_end[a] if hasattr(sim, 'last_green_end') else 0.0
                since = max(0.0, t - since) if since > 0 else Cj
            X[a] = [s['score'] / 15.0, s['groups'][0] / 10.0, s['groups'][1] / 10.0, s['groups'][2] / 10.0,
                    s['groups'][3] / 5.0, s['fill'], gshare, Cj / 150.0, dscore[a] / 15.0,
                    1.0 if a in sim.net.internal else 0.0, 1.0 if d in (0, 2) else 0.0,
                    green_now, min(since, 300.0) / 150.0, 0.0]
        for j in range(3):
            sj = scores[4 * j:4 * j + 4]
            sig = sim.sig[j]
            Cj = sig.cycle_len if sig.cycle_len > 0 else 120.0
            elapsed = (t - sig.cycle_start) / Cj if sig.cycle_len > 0 else 0.0
            X[12 + j] = [sj.mean() / 15.0, sj.max() / 15.0, sj.std() / 15.0,
                         sum(snaps[4 * j + d]['n'] for d in range(4)) / 40.0,
                         float(np.mean([snaps[4 * j + d]['fill'] for d in range(4)])), 0.0, 0.0,
                         Cj / 150.0, dscore[4 * j:4 * j + 4].mean() / 15.0, 1.0 if j == 1 else 0.0, 0.0,
                         min(1.5, elapsed), 0.0, 1.0]
        self.prev_score = scores
        return X, scores


class RandomizedBehaviour:
    """Exploratory behaviour policy used only to generate estimator training data."""
    cyclic = True
    name = 'RB'

    def __init__(self, profile, tactical=False, log_prob=0.5):
        self.fb = FeatureBuilder(profile)
        self.tactical = tactical
        self.tc = TacticalCamera(profile)
        self.log_prob = log_prob
        self.log_t, self.log_X = [], []

    def reset(self, sim):
        self.fb.reset()
        self.tc.reset()
        self.log_t, self.log_X = [], []

    def plan_junction(self, sim, j, t):
        rng = sim.rng_c
        X, scores = self.fb.build(sim, t)
        if rng.random() < self.log_prob:
            self.log_t.append(t)
            self.log_X.append(X)
        C = int(rng.integers(12, 33) * 5)  # 60..160
        s = scores[4 * j:4 * j + 4] + 0.5
        w = s / s.sum()
        mix = rng.uniform(0.0, 0.6)
        w = (1 - mix) * w + mix * rng.dirichlet(np.ones(4))
        g = split_greens(w, C)
        order = order_phases(w, sim.sig[j].last_served) if rng.random() < 0.5 else [0, 1, 2, 3]
        return [(d, int(g[d])) for d in order]

    def gap_out(self, sim, j, t):
        return self.tc.check(sim, j, t)


def targets_from_passages(sim, times, H=H_TARGET):
    """lane demand y (queue at t + arrivals in (t, t+H]) and junction arrival flow ratio."""
    arrs = []
    for a in range(12):
        p = list(sim.passages[a])
        for lane in (0, 1):
            for vid in sim.lanes[a][lane]:
                p.append((sim.v_tff[vid], np.inf, sim.v_w[vid]))
        arrs.append(np.array(p) if p else np.zeros((0, 3)))
    Y = np.zeros((len(times), 12))
    A = np.zeros((len(times), 12))
    for k, t in enumerate(times):
        for a in range(12):
            arr = arrs[a]
            if arr.size == 0:
                continue
            tff, tdis, w = arr[:, 0], arr[:, 1], arr[:, 2]
            Y[k, a] = w[(tff <= t + H) & (tdis > t)].sum()
            A[k, a] = w[(tff > t) & (tff <= t + H)].sum()
    psi = np.zeros((len(times), 3))
    for j in range(3):
        psi[:, j] = np.minimum(1.0, A[:, 4 * j:4 * j + 4].sum(axis=1) / H)
    return Y, psi, Y - A, A


class GATCoordinated:
    """Proposed controller.

    Strategic layer (once per cycle of junction j):
      1. camera snapshots -> PCU lane scores and class-group counts (FeatureBuilder)
      2. graph estimator -> next-cycle lane demand y_hat and junction flow ratio psi
      3. cycle length from Webster's formula at psi_j, smoothed
      4. spillback guard discounts demand of movements feeding loaded downstream links
      5. green splits proportional to guarded demand, phases in descending demand order
    Tactical layer (every 2 s): camera gap-out, spare green passed to the remaining phases.
    """
    cyclic = True
    name = 'GATC'

    def __init__(self, profile, model, c_min=60, c_max=160, kappa=0.9, rho0=0.5, smooth=0.5,
                 priority=True, adapt_cycle=True, guard=True, tactical=True, C_fixed=120, name=None,
                 max_factor=1.0, check_every=2, empty_needed=2, clear_cycle=True):
        self.clear_cycle = clear_cycle
        self.fb = FeatureBuilder(profile)
        self.tc = TacticalCamera(profile, every=check_every, empty_needed=empty_needed)
        self.max_factor = max_factor
        self.model = model
        self.c_min, self.c_max = c_min, c_max
        self.kappa, self.rho0 = kappa, rho0
        self.smooth = smooth
        self.priority = priority
        self.adapt_cycle = adapt_cycle
        self.guard = guard
        self.tactical = tactical
        self.C_fixed = C_fixed
        if name:
            self.name = name
        self.trace = []

    def reset(self, sim):
        self.fb.reset()
        self.tc.reset()
        self.C_prev = [self.C_fixed] * 3
        self.trace = []

    def plan_junction(self, sim, j, t):
        net = sim.net
        X, scores = self.fb.build(sim, t)
        out, psi = self.model.forward(X[None])
        out = out.data[0] * Y_SCALE          # (12, 2): predicted queue Q and arrivals A over H
        qhat, ahat = out[:, 0], out[:, 1]
        yhat = qhat + ahat
        psi = psi.data[0]
        lanes = slice(4 * j, 4 * j + 4)
        if self.adapt_cycle:
            Yj = float(psi[j])
            if Yj < 0.99:
                C_web = (1.5 * L_CYCLE + 5.0) / (1.0 - Yj)
                # queue-clearance cycle: green must serve the standing queue plus cycle arrivals
                C_clr = (L_CYCLE + qhat[lanes].sum()) / (1.0 - Yj)
            else:
                C_web = C_clr = self.c_max
            C_raw = self.max_factor * max(C_web, C_clr if self.clear_cycle else 0.0)
            C_raw = min(max(C_raw, self.c_min), self.c_max)
            C = self.smooth * self.C_prev[j] + (1 - self.smooth) * C_raw
            C = int(round(C / 5.0) * 5)
        else:
            C = self.C_fixed
        self.C_prev[j] = C
        # green need over the planned cycle (s of effective green at 1 PCU/s per arm)
        need = qhat[lanes] + ahat[lanes] * (C / H_TARGET)
        e = np.maximum(need, 0.5)
        delta = np.ones(4)
        if self.guard:
            for d in range(4):
                a = 4 * j + d
                pen = 0.0
                for m, pm in zip(('L', 'T', 'R'), net.turn[a]):
                    typ, tgt = net.dest[a][m]
                    if typ == 'app':
                        rho = yhat[tgt] / (net.storage[tgt] * PCU_PER_SPACE)
                        phi = min(1.0, max(0.0, (rho - self.rho0) / (1.0 - self.rho0)))
                        pen += pm * phi
                delta[d] = 1.0 - self.kappa * pen
        w = e * delta
        g = split_greens(w, C)
        order = order_phases(w, sim.sig[j].last_served) if self.priority else [0, 1, 2, 3]
        self.trace.append((t, j, C, psi.copy(), yhat.copy(), delta.copy()))
        return [(d, int(g[d])) for d in order]

    def gap_out(self, sim, j, t):
        return self.tc.check(sim, j, t)

    def attention_snapshot(self, sim, t):
        X, _ = self.fb.build(sim, t)
        self.model.forward(X[None])
        return self.model.attention()
