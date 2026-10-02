"""
Mesoscopic spatial-queue simulator for a three-junction, four-approach, dual-lane
urban corridor carrying heterogeneous (Indian) traffic.

Model summary
-------------
* Links carry vehicles at class-specific free-flow speeds (overtaking allowed).
* Each approach has two lanes; vehicles join a lane FIFO queue at their free-flow
  stop-line arrival time (vertical queue), with lane choice by movement.
* Storage: every approach link has a finite storage (car-space units, 7 m per car
  per lane). A vehicle can discharge into a downstream link only if space exists,
  which reproduces spillback and blocked green.
* Discharge: class-specific saturation headways per lane, randomised per vehicle.
  Start-up lost time 2 s; discharge continues 1 s into amber.
* Signals: split phasing (one approach served at a time), amber 3 s, all-red 2 s.
* Left-hand traffic (India): left turn is the near-side turn.
"""
import heapq
import math
from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# Vehicle classes (UVH-26's 14 classes grouped into 7 simulation groups)
# ----------------------------------------------------------------------------
CLS = ['2W', 'BIC', '3W', 'CAR', 'LCV', 'BUS', 'TRK']
NC = len(CLS)
# design PCU weights used by the controllers (lane scoring)
W_DESIGN = np.array([0.5, 0.5, 1.0, 1.0, 1.5, 3.0, 3.0])
# true per-lane saturation headways in the simulator (s)
HEADWAY = np.array([0.8, 1.2, 1.9, 2.0, 2.8, 4.4, 4.8])
# jam space in car-space units (7 m per car per lane; lateral packing for 2W/BIC)
SPACE = np.array([0.3, 0.3, 0.6, 1.0, 1.3, 2.2, 2.0])
SPEED_MU = np.array([11.1, 4.2, 8.3, 10.0, 8.9, 8.3, 7.8])
SPEED_SD = np.array([1.5, 0.6, 1.0, 1.5, 1.0, 1.0, 0.8])
MIX_BASE = np.array([0.50, 0.02, 0.15, 0.24, 0.05, 0.025, 0.015])
MIX_HV = np.array([0.40, 0.02, 0.20, 0.22, 0.06, 0.06, 0.04])
CAR_SPACE_M = 7.0

# ----------------------------------------------------------------------------
# Signal timing constants
# ----------------------------------------------------------------------------
AMBER = 3
ALLRED = 2
INTERGREEN = AMBER + ALLRED
STARTUP_LOST = 2.0
AMBER_USE = 1.0
G_MIN = 8
LOST_PER_PHASE = INTERGREEN - AMBER_USE + STARTUP_LOST  # 6 s
N_PHASE = 4
L_CYCLE = N_PHASE * LOST_PER_PHASE  # 24 s
S_APPROACH_PCU = 3600.0  # design saturation flow per 2-lane approach (PCU/h)

APP_NAMES = ['W', 'N', 'E', 'S']
HEADING = {0: 'E', 1: 'S', 2: 'W', 3: 'N'}
LEFT_OF = {'E': 'N', 'S': 'E', 'W': 'S', 'N': 'W'}   # left-hand traffic
RIGHT_OF = {'E': 'S', 'S': 'W', 'W': 'N', 'N': 'E'}
TURN_CORRIDOR = (0.15, 0.72, 0.13)  # L, T, R
TURN_SIDE = (0.30, 0.40, 0.30)
MOVES = ('L', 'T', 'R')


class Network:
    """Three junctions J1..J3 (index 0..2) along a west-east arterial."""

    def __init__(self, spacing=(300.0, 250.0), entry_len=300.0):
        self.nJ = 3
        self.nA = 12
        self.lanes = 2
        L = np.full(self.nA, entry_len)
        # internal approaches: J1-E (2) fed by J2; J2-W (4) fed by J1;
        # J2-E (6) fed by J3; J3-W (8) fed by J2
        L[4] = spacing[0]
        L[2] = spacing[0]
        L[8] = spacing[1]
        L[6] = spacing[1]
        self.length = L
        self.storage = self.lanes * self.length / CAR_SPACE_M
        self.boundary = [0, 1, 3, 5, 7, 9, 10, 11]
        self.internal = [2, 4, 6, 8]
        self.corridor = [a for a in range(12) if a % 4 in (0, 2)]
        self.turn = {}
        self.dest = {}
        for a in range(12):
            j, d = divmod(a, 4)
            self.turn[a] = TURN_CORRIDOR if d in (0, 2) else TURN_SIDE
            heading = HEADING[d]
            self.dest[a] = {}
            for m in MOVES:
                out = heading if m == 'T' else (LEFT_OF[heading] if m == 'L' else RIGHT_OF[heading])
                if out == 'E':
                    tgt = ('app', 4 * (j + 1) + 0) if j < 2 else ('exit', 'E')
                elif out == 'W':
                    tgt = ('app', 4 * (j - 1) + 2) if j > 0 else ('exit', 'W')
                else:
                    tgt = ('exit', out + str(j))
                self.dest[a][m] = tgt
        # feeders: which approaches send vehicles into each internal approach
        self.feeders = {a: [] for a in range(12)}
        for a in range(12):
            for m in MOVES:
                typ, tgt = self.dest[a][m]
                if typ == 'app':
                    self.feeders[tgt].append((a, m))

    def analytic_flows(self, boundary_q):
        """Propagate boundary flows (veh/h) with the design turning shares."""
        q = np.zeros(12)
        for a, v in boundary_q.items():
            q[a] = v
        # eastbound chain then westbound chain (acyclic along each direction)
        order = [4, 8, 6, 2]
        for tgt in order:
            s = 0.0
            for (a, m) in self.feeders[tgt]:
                s += q[a] * self.turn[a][MOVES.index(m)]
            q[tgt] = s
        return q


BASE_BOUNDARY = {0: 1400.0, 1: 380.0, 3: 340.0, 5: 450.0, 7: 420.0, 9: 360.0, 10: 1250.0, 11: 400.0}
BASE_SCALE = 0.93  # calibrated so that the critical flow ratio Y at J2 is about 0.70 at factor 1.0

SCENARIOS = {
    'S1': dict(name='Off-peak', factor=0.60, eb=1.0, wb=1.0, incident=None, mix='base'),
    'S2': dict(name='Peak', factor=1.00, eb=1.0, wb=1.0, incident=None, mix='base'),
    'S3': dict(name='Oversaturated', factor=1.30, eb=1.0, wb=1.0, incident=None, mix='base'),
    'S4': dict(name='Tidal surge', factor=1.00, eb=1.5, wb=0.6, incident=None, mix='base'),
    'S5': dict(name='Lane-blocking incident', factor=1.00, eb=1.0, wb=1.0,
               incident=(8, 0, 1500, 2700), mix='base'),
    'S2HV': dict(name='Peak, heavy-vehicle mix', factor=1.00, eb=1.0, wb=1.0, incident=None, mix='hv'),
}


def scenario_boundary_flows(sc, factor_override=None):
    f = sc['factor'] if factor_override is None else factor_override
    q = {}
    for a, v in BASE_BOUNDARY.items():
        mult = 1.0
        if a == 0:
            mult = sc.get('eb', 1.0)
        elif a == 10:
            mult = sc.get('wb', 1.0)
        q[a] = v * BASE_SCALE * f * mult
    return q


def demand_profile(t):
    """Time-of-day multiplier: mean 1.0 over the 600-3600 s measurement window."""
    if t < 600:
        return 0.85
    if t < 2100:
        return 0.85 + 0.30 * (t - 600) / 1500.0
    if t < 3600:
        return 1.15 - 0.30 * (t - 2100) / 1500.0
    return 0.85


PROFILE_MAX = 1.15


class Signal:
    __slots__ = ('cur', 'stage', 'stage_end', 'g_start', 'y_start', 'plan', 'last_served',
                 'g_len', 'cycle_start', 'cycle_len', 'green_given')

    def __init__(self):
        self.cur = -1
        self.stage = 'R'
        self.stage_end = 0
        self.g_start = -1e9
        self.y_start = -1e9
        self.plan = deque()
        self.last_served = -1
        self.g_len = 0
        self.cycle_start = 0
        self.cycle_len = 0
        self.green_given = [0, 0, 0, 0]


class Sim:
    def __init__(self, scenario, seed, controller, perception=None, net=None,
                 T_end=4800, warm=600, meas_end=3600, record_passages=False,
                 demand_override=None):
        self.net = net if net is not None else Network()
        self.sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.seed = seed
        self.T_end = T_end
        self.warm = warm
        self.meas_end = meas_end
        self.ctrl = controller
        self.perception = perception
        self.rng_d = np.random.default_rng([seed, 11])
        self.rng_p = np.random.default_rng([seed, 22])
        self.rng_c = np.random.default_rng([seed, 33])
        self.record_passages = record_passages
        self.demand_override = demand_override
        self._generate_demand()
        self._init_state()

    # ------------------------------------------------------------------
    def _generate_demand(self):
        sc = self.sc
        if self.demand_override is not None:
            qb = self.demand_override
        else:
            qb = scenario_boundary_flows(sc)
        self.boundary_q = qb
        mixspec = sc.get('mix', 'base')
        if isinstance(mixspec, np.ndarray):
            mix = mixspec
        else:
            mix = MIX_HV if mixspec == 'hv' else MIX_BASE
        self.mix = mix
        flat = sc.get('flat', False)
        pmax = 1.0 if flat else PROFILE_MAX
        rng = self.rng_d
        veh_t, veh_app = [], []
        for a in sorted(qb.keys()):
            lam = qb[a] / 3600.0 * pmax
            if lam <= 0:
                continue
            n = rng.poisson(lam * self.T_end * 1.1) + 50
            gaps = rng.exponential(1.0 / lam, size=n)
            times = np.cumsum(gaps)
            times = times[times < self.T_end]
            if not flat:
                acc = rng.random(times.size) < np.array([demand_profile(t) for t in times]) / PROFILE_MAX
                times = times[acc]
            veh_t.append(times)
            veh_app.append(np.full(times.size, a))
        t_all = np.concatenate(veh_t)
        a_all = np.concatenate(veh_app)
        order = np.argsort(t_all, kind='stable')
        t_all = t_all[order]
        a_all = a_all[order]
        n = t_all.size
        cls = rng.choice(NC, size=n, p=mix)
        spd = np.maximum(SPEED_MU[cls] + SPEED_SD[cls] * rng.standard_normal(n), 0.5 * SPEED_MU[cls])
        hs = sc.get('hw_sigma', 0.12)
        hwf = np.exp(hs * rng.standard_normal(n) - 0.5 * hs ** 2)
        urand = rng.random((n, 3))
        self.nV = n
        self.v_tgen = t_all.tolist()
        self.v_app0 = a_all.astype(int).tolist()
        self.v_cls = cls.astype(int).tolist()
        self.v_spd = spd.tolist()
        self.v_hw = (HEADWAY[cls] * hwf).tolist()
        self.v_u = urand.tolist()
        self.v_w = W_DESIGN[cls].tolist()
        self.v_space = SPACE[cls].tolist()

    def _choose_move(self, vid, a):
        k = self.v_nj[vid]
        u = self.v_u[vid][min(k, 2)]
        pL, pT, _ = self.net.turn[a]
        if u < pL:
            return 'L'
        if u < pL + pT:
            return 'T'
        return 'R'

    def _init_state(self):
        n = self.nV
        self.v_nj = [0] * n
        self.v_move = [''] * n
        self.v_tff = [0.0] * n
        self.v_delay = [0.0] * n
        self.v_entrywait = [0.0] * n
        self.v_texit = [-1.0] * n
        self.v_cur = [-1] * n
        self.v_state = [0] * n  # 0 not entered, 1 moving, 2 queued, 3 exited
        self.v_exitdir = [''] * n
        nA = self.net.nA
        self.moving = [[] for _ in range(nA)]           # heaps of (t_ff, vid)
        self.lanes = [[deque(), deque()] for _ in range(nA)]
        self.lane_space = [[0.0, 0.0] for _ in range(nA)]
        self.lane_pcu = [[0.0, 0.0] for _ in range(nA)]
        self.lane_last = [[-1e9, -1e9] for _ in range(nA)]
        self.win_open = [-1] * nA
        self.occ = [0.0] * nA
        self.entry_buf = {a: deque() for a in self.net.boundary}
        self.next_vid = 0
        self.sig = [Signal() for _ in range(self.net.nJ)]
        self.last_dis_time = [-1e9] * nA
        # statistics
        self.stat_q_veh = np.zeros(nA)
        self.stat_q_pcu = np.zeros(nA)
        self.stat_q_max = np.zeros(nA)
        self.stat_occ = np.zeros(nA)
        self.stat_spill_any = 0
        self.stat_spill_link = np.zeros(nA)
        self.stat_seconds = 0
        self.g_used = 0
        self.g_idle = 0
        self.g_blocked = 0
        self.app_delay_sum = np.zeros(nA)
        self.app_delay_n = np.zeros(nA)
        self.exits_in_window = 0
        self.q_series = []   # per-second junction queue (PCU) for plotting
        self.cycle_log = []  # (t, C, greens per junction)
        self.passages = [[] for _ in range(nA)] if self.record_passages else None
        self.junction_q_acc = np.zeros(self.net.nJ)
        self.incident = self.sc.get('incident', None)
        self.dis_count_window = np.zeros(nA)
        self.gapouts = 0
        self.last_green_end = [0.0] * nA

    # ------------------------------------------------------------------
    def lane_blocked_by_incident(self, a, lane, t):
        inc = self.incident
        if inc is None:
            return False
        ia, il, t0, t1 = inc
        return a == ia and lane == il and t0 <= t < t1

    def discharge_allowed(self, a, t):
        """True during effective green seconds of approach a."""
        j, d = divmod(a, 4)
        s = self.sig[j]
        if s.cur != d:
            return False
        if s.stage == 'G':
            return t >= s.g_start + STARTUP_LOST
        if s.stage == 'Y':
            return t < s.y_start + AMBER_USE
        return False

    def window_end(self, a, t):
        """End of the effective green of approach a (inf while green), or None.

        A vehicle may cross if more than half of its discharge headway lies inside the
        effective green (midpoint rule), which keeps the expected capacity equal to
        saturation flow x effective green."""
        j, d = divmod(a, 4)
        s = self.sig[j]
        if s.cur != d:
            return None
        if s.stage == 'G':
            return math.inf if t >= s.g_start + STARTUP_LOST else None
        if s.y_start >= s.g_start:
            return s.y_start + AMBER_USE
        return None

    # ------------------------------------------------------------------
    def _enter_link(self, vid, a, t_in):
        """vehicle enters approach link a at time t_in"""
        self.v_cur[vid] = a
        self.v_state[vid] = 1
        tff = t_in + self.net.length[a] / self.v_spd[vid]
        self.v_tff[vid] = tff
        self.v_move[vid] = self._choose_move(vid, a)
        heapq.heappush(self.moving[a], (tff, vid))
        self.occ[a] += self.v_space[vid]

    def _join_queue(self, vid, a, t):
        m = self.v_move[vid]
        if self.lane_blocked_by_incident(a, 0, t):
            lane = 1
        elif m == 'L':
            lane = 0
        elif m == 'R':
            lane = 1
        else:
            ls = self.lane_space[a]
            lane = 0 if ls[0] <= ls[1] else 1
        self.lanes[a][lane].append(vid)
        self.lane_space[a][lane] += self.v_space[vid]
        self.lane_pcu[a][lane] += self.v_w[vid]
        self.v_state[vid] = 2

    def _downstream_ok(self, vid, a):
        typ, tgt = self.net.dest[a][self.v_move[vid]]
        if typ == 'exit':
            return True
        return self.occ[tgt] + self.v_space[vid] <= self.net.storage[tgt] + 1e-9

    def _discharge(self, vid, a, lane, tau):
        self.lanes[a][lane].popleft()
        sp = self.v_space[vid]
        self.lane_space[a][lane] -= sp
        self.lane_pcu[a][lane] -= self.v_w[vid]
        self.occ[a] -= sp
        d = max(0.0, tau - self.v_tff[vid])
        self.v_delay[vid] += d
        if self.warm <= self.v_tgen[vid] < self.meas_end:
            self.app_delay_sum[a] += d
            self.app_delay_n[a] += 1
        if self.record_passages:
            self.passages[a].append((self.v_tff[vid], tau, self.v_w[vid]))
        self.last_dis_time[a] = tau
        if self.warm <= tau < self.meas_end:
            self.dis_count_window[a] += 1
        self.v_nj[vid] += 1
        typ, tgt = self.net.dest[a][self.v_move[vid]]
        if typ == 'exit':
            self.v_state[vid] = 3
            self.v_texit[vid] = tau + 2.0 + 150.0 / self.v_spd[vid]
            self.v_exitdir[vid] = tgt
            if self.warm <= tau < self.meas_end:
                self.exits_in_window += 1
        else:
            self._enter_link(vid, tgt, tau + 2.0)

    # ------------------------------------------------------------------
    def _signal_update(self, t):
        ctrl = self.ctrl
        if ctrl.cyclic:
            # cycle-based control, planned per junction when its previous cycle ends
            for j in range(self.net.nJ):
                s = self.sig[j]
                if s.stage == 'R' and t >= s.stage_end and not s.plan:
                    plan = ctrl.plan_junction(self, j, t)
                    s.plan = deque(plan)
                    s.cycle_start = t
                    s.cycle_len = sum(g for _, g in plan) + INTERGREEN * len(plan)
                    s.green_given = [0, 0, 0, 0]
                    for d, g in plan:
                        s.green_given[d] += g
                    self.cycle_log.append((t, j, s.cycle_len, list(plan)))
                if s.stage == 'G' and getattr(ctrl, 'tactical', False) and t < s.stage_end:
                    el = t - s.g_start
                    every = ctrl.tc.every if hasattr(ctrl, 'tc') else 2
                    if el >= G_MIN and (el - G_MIN) % every == 0 and ctrl.gap_out(self, j, t):
                        spare = s.stage_end - t
                        s.green_given[s.cur] -= spare
                        if s.plan and spare > 0:
                            tot = sum(g for _, g in s.plan)
                            newp = []
                            acc = 0
                            for k, (d, g) in enumerate(s.plan):
                                add = int(round(spare * g / tot)) if k < len(s.plan) - 1 else spare - acc
                                acc += add
                                newp.append((d, g + add))
                                s.green_given[d] += add
                            s.plan = deque(newp)
                        s.stage_end = t
                        self.gapouts += 1
            for j in range(self.net.nJ):
                s = self.sig[j]
                if t >= s.stage_end:
                    if s.stage == 'G':
                        s.stage = 'Y'
                        s.y_start = t
                        s.stage_end = t + AMBER
                        self.last_green_end[4 * j + s.cur] = t
                    elif s.stage == 'Y':
                        s.stage = 'R'
                        s.stage_end = t + ALLRED
                    elif s.stage == 'R' and s.plan:
                        d, g = s.plan.popleft()
                        s.cur = d
                        s.stage = 'G'
                        s.g_start = t
                        s.g_len = g
                        s.stage_end = t + g
                        s.last_served = d
        else:
            for j in range(self.net.nJ):
                s = self.sig[j]
                if s.stage == 'G':
                    if ctrl.end_green(self, j, t):
                        s.stage = 'Y'
                        s.y_start = t
                        s.stage_end = t + AMBER
                        self.last_green_end[4 * j + s.cur] = t
                elif s.stage == 'Y' and t >= s.stage_end:
                    s.stage = 'R'
                    s.stage_end = t + ALLRED
                elif s.stage == 'R' and t >= s.stage_end:
                    d = ctrl.next_phase(self, j, t)
                    s.cur = d
                    s.stage = 'G'
                    s.g_start = t
                    s.stage_end = 10 ** 9
                    s.last_served = d

    # ------------------------------------------------------------------
    def step(self, t):
        net = self.net
        # 1. boundary generation into entry buffers
        while self.next_vid < self.nV and self.v_tgen[self.next_vid] < t + 1:
            vid = self.next_vid
            self.entry_buf[self.v_app0[vid]].append(vid)
            self.next_vid += 1
        for a, buf in self.entry_buf.items():
            while buf:
                vid = buf[0]
                if self.occ[a] + self.v_space[vid] <= net.storage[a] + 1e-9:
                    buf.popleft()
                    t_in = max(self.v_tgen[vid], float(t))
                    self.v_entrywait[vid] = t_in - self.v_tgen[vid]
                    self._enter_link(vid, a, t_in)
                else:
                    break
        # 2. signals
        self._signal_update(t)
        in_window = self.warm <= t < self.meas_end
        # 3. per approach: join queues and discharge
        for a in range(net.nA):
            mv = self.moving[a]
            while mv and mv[0][0] < t + 1:
                tff, vid = heapq.heappop(mv)
                self._join_queue(vid, a, t)
            allowed = self.discharge_allowed(a, t)
            wend = self.window_end(a, t)
            if allowed:
                j = a // 4
                gs = self.sig[j].g_start + STARTUP_LOST
                if self.win_open[a] != self.sig[j].g_start:
                    # discharge window opens: queued vehicles need start-up lost time
                    self.win_open[a] = self.sig[j].g_start
                    for lane in (0, 1):
                        if self.lane_last[a][lane] < gs:
                            self.lane_last[a][lane] = gs
            for lane in (0, 1):
                q = self.lanes[a][lane]
                if wend is None or self.lane_blocked_by_incident(a, lane, t):
                    continue
                used = False
                blocked = False
                while q:
                    vid = q[0]
                    h = self.v_hw[vid]
                    tau = max(self.lane_last[a][lane] + h, self.v_tff[vid], float(t))
                    if tau - 0.5 * h > wend:
                        break
                    if tau >= t + 1:
                        used = True  # head vehicle is in its discharge headway
                        break
                    if not self._downstream_ok(vid, a):
                        blocked = True
                        break
                    self._discharge(vid, a, lane, tau)
                    self.lane_last[a][lane] = tau
                    used = True
                if in_window and allowed:
                    if blocked:
                        self.g_blocked += 1
                    elif used:
                        self.g_used += 1
                    else:
                        self.g_idle += 1
        # 4. statistics
        if in_window:
            self.stat_seconds += 1
            spill = False
            jq = [0.0, 0.0, 0.0]
            for a in range(net.nA):
                nq = len(self.lanes[a][0]) + len(self.lanes[a][1])
                pq = self.lane_pcu[a][0] + self.lane_pcu[a][1]
                self.stat_q_veh[a] += nq
                self.stat_q_pcu[a] += pq
                if pq > self.stat_q_max[a]:
                    self.stat_q_max[a] = pq
                r = self.occ[a] / net.storage[a]
                self.stat_occ[a] += r
                jq[a // 4] += pq
                if a in (2, 4, 6, 8) and r >= 0.95:
                    spill = True
                    self.stat_spill_link[a] += 1
            if spill:
                self.stat_spill_any += 1
            self.junction_q_acc += np.array(jq)
            self.q_series.append(jq)

    # ------------------------------------------------------------------
    def run(self):
        if hasattr(self.ctrl, 'reset'):
            self.ctrl.reset(self)
        for t in range(self.T_end):
            self.step(t)
        return self.metrics()

    # ------------------------------------------------------------------
    def _realized_cycle(self):
        diffs = []
        for j in range(self.net.nJ):
            ts = [c[0] for c in self.cycle_log if c[1] == j]
            for t0, t1 in zip(ts[:-1], ts[1:]):
                if self.warm <= t0 < self.meas_end:
                    diffs.append(t1 - t0)
        return float(np.mean(diffs)) if diffs else float('nan')

    def queue_pcu_true(self, a):
        return self.lane_pcu[a][0] + self.lane_pcu[a][1]

    def metrics(self):
        T = self.T_end
        idx = [v for v in range(self.nV) if self.warm <= self.v_tgen[v] < self.meas_end]
        delays, tts, eb_tt = [], [], []
        unfinished = 0
        for v in idx:
            d = self.v_delay[v] + self.v_entrywait[v]
            st = self.v_state[v]
            if st == 0:
                # still in the entry buffer
                d = T - self.v_tgen[v]
                unfinished += 1
            elif st in (1, 2):
                if st == 2 and self.v_tff[v] < T:
                    d += T - self.v_tff[v]
                unfinished += 1
            else:
                tt = self.v_texit[v] - self.v_tgen[v]
                tts.append(tt)
                if self.v_app0[v] == 0 and self.v_exitdir[v] == 'E':
                    eb_tt.append(tt)
            delays.append(d)
        delays = np.array(delays)
        n_sec = max(1, self.stat_seconds)
        app_d = np.where(self.app_delay_n > 0, self.app_delay_sum / np.maximum(self.app_delay_n, 1), 0.0)
        jain = (app_d.sum() ** 2) / (len(app_d) * (app_d ** 2).sum() + 1e-12)
        gtot = self.g_used + self.g_idle + self.g_blocked
        jq = self.junction_q_acc / n_sec
        occ_int = self.stat_occ[[2, 4, 6, 8]] / n_sec
        res = dict(
            delay=float(delays.mean()),
            delay_p95=float(np.percentile(delays, 95)),
            travel_time=float(np.mean(tts)) if tts else float('nan'),
            eb_travel_time=float(np.mean(eb_tt)) if eb_tt else float('nan'),
            throughput=float(self.exits_in_window * 3600.0 / (self.meas_end - self.warm)),
            completion=float(1.0 - unfinished / max(1, len(idx))),
            queue_veh=float(self.stat_q_veh.sum() / n_sec / self.net.nA),
            queue_pcu=float(self.stat_q_pcu.sum() / n_sec / self.net.nA),
            queue_max=float(self.stat_q_max.max()),
            jain=float(jain),
            gur=float(self.g_used / max(1, gtot)),
            blocked_share=float(self.g_blocked / max(1, gtot)),
            idle_share=float(self.g_idle / max(1, gtot)),
            spill_share=float(self.stat_spill_any / n_sec),
            load_dispersion=float(np.std(jq)),
            congestion_index=float(occ_int.mean()),
            n_cohort=len(idx),
            mean_cycle=self._realized_cycle(),
            gapouts=int(self.gapouts),
        )
        res['app_delay'] = app_d.tolist()
        res['delays'] = delays
        return res


# ----------------------------------------------------------------------------
# Camera snapshot and detector error model
# ----------------------------------------------------------------------------
COCO_LABELS = ['car', 'motorcycle', 'bicycle', 'bus', 'truck']
COCO_W = {'car': 1.0, 'motorcycle': 0.5, 'bicycle': 0.5, 'bus': 3.0, 'truck': 3.0}


class DetectorProfile:
    """Per-class recall, label confusion, false positives and occlusion decay.

    labels: 'uvh' -> outputs the 7 simulation groups (from UVH-26's 14 classes)
            'coco' -> outputs COCO labels (car, motorcycle, bicycle, bus, truck)
    """

    def __init__(self, name, labels, recall, confusion, fp_rate, occlusion):
        self.name = name
        self.labels = labels
        self.recall = np.array(recall, dtype=float)
        self.confusion = confusion  # list per true class of (label_list, prob_list)
        self.fp_rate = fp_rate
        self.occlusion = occlusion
        self.recall_mult = 1.0
        self.extra_fp = 0.0

    def label_space(self):
        return CLS if self.labels == 'uvh' else COCO_LABELS

    def weight_of(self, lab):
        if self.labels == 'uvh':
            return W_DESIGN[CLS.index(lab)]
        return COCO_W[lab]


def _uvh_conf(p_other):
    """UVH-style confusion: mostly correct, small confusion among neighbours."""
    nb = {'2W': ['BIC', '3W'], 'BIC': ['2W'], '3W': ['2W', 'CAR'], 'CAR': ['LCV', '3W'],
          'LCV': ['CAR', 'TRK'], 'BUS': ['TRK'], 'TRK': ['BUS', 'LCV']}
    out = []
    for c in CLS:
        labs = [c] + nb[c]
        pr = [1.0 - p_other] + [p_other / len(nb[c])] * len(nb[c])
        out.append((labs, pr))
    return out


def make_profiles():
    P0 = DetectorProfile('Oracle', 'uvh', [1.0] * NC, [([c], [1.0]) for c in CLS], 0.0, 0.0)
    P1 = DetectorProfile('UVH-26 YOLO (best.pt)', 'uvh',
                         [0.88, 0.78, 0.93, 0.95, 0.90, 0.95, 0.93], _uvh_conf(0.05), 0.3, 0.15)
    coco_m = [
        (['motorcycle', 'bicycle'], [0.97, 0.03]),            # 2W
        (['bicycle', 'motorcycle'], [0.90, 0.10]),            # BIC
        (['car', 'motorcycle', 'truck'], [0.55, 0.20, 0.25]),  # 3W (no COCO class)
        (['car', 'truck'], [0.92, 0.08]),                     # CAR (SUV/MUV sometimes truck)
        (['truck', 'car', 'bus'], [0.60, 0.35, 0.05]),        # LCV / van / tempo
        (['bus', 'truck'], [0.90, 0.10]),                     # BUS / mini-bus
        (['truck', 'bus'], [0.95, 0.05]),                     # TRK
    ]
    P2 = DetectorProfile('COCO YOLOv8m', 'coco',
                         [0.82, 0.70, 0.70, 0.93, 0.80, 0.90, 0.88], coco_m, 0.5, 0.25)
    coco_n = [
        (['motorcycle', 'bicycle'], [0.95, 0.05]),
        (['bicycle', 'motorcycle'], [0.85, 0.15]),
        (['car', 'motorcycle', 'truck'], [0.50, 0.20, 0.30]),
        (['car', 'truck'], [0.90, 0.10]),
        (['truck', 'car', 'bus'], [0.55, 0.40, 0.05]),
        (['bus', 'truck'], [0.85, 0.15]),
        (['truck', 'bus'], [0.93, 0.07]),
    ]
    P3 = DetectorProfile('COCO YOLOv8n', 'coco',
                         [0.66, 0.52, 0.55, 0.87, 0.68, 0.84, 0.80], coco_n, 0.7, 0.35)
    return {'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3}


CAM_RANGE = 60.0  # metres of approach visible to the camera


def snapshot(sim, a, t, prof, rng):
    """Detected vehicles on approach a within CAM_RANGE of the stop line.

    Returns dict with score (PCU), group counts (2W+BIC, 3W, CAR+LCV+other, BUS+TRK),
    visible-zone fill ratio, and number of detections.
    """
    sim.frame_count = getattr(sim, 'frame_count', 0) + 1
    vis = []  # (true class, depth m)
    qlen_lane = [0.0, 0.0]
    for lane in (0, 1):
        pos = 0.0
        for vid in sim.lanes[a][lane]:
            if pos >= CAM_RANGE:
                break
            vis.append((sim.v_cls[vid], pos))
            pos += sim.v_space[vid] * CAR_SPACE_M
        # full lane queue length
        qlen_lane[lane] = sim.lane_space[a][lane] * CAR_SPACE_M
    qmean = 0.5 * (qlen_lane[0] + qlen_lane[1])
    for (tff, vid) in sim.moving[a]:
        dist = sim.v_spd[vid] * (tff - t)
        p = max(dist, qmean)
        if p < CAM_RANGE:
            vis.append((sim.v_cls[vid], p))
    labs = prof.label_space()
    counts = {lab: 0 for lab in labs}
    space_det = 0.0
    rmult = prof.recall_mult
    for (c, depth) in vis:
        r = prof.recall[c] * rmult * (1.0 - prof.occlusion * min(1.0, depth / CAM_RANGE))
        if rng.random() < r:
            ll, pp = prof.confusion[c]
            if len(ll) == 1:
                lab = ll[0]
            else:
                u = rng.random()
                acc = 0.0
                lab = ll[-1]
                for L_, P_ in zip(ll, pp):
                    acc += P_
                    if u < acc:
                        lab = L_
                        break
            counts[lab] += 1
            space_det += SPACE[c]
    nfp = rng.poisson(prof.fp_rate + prof.extra_fp) if (prof.fp_rate + prof.extra_fp) > 0 else 0
    for _ in range(nfp):
        if prof.labels == 'uvh':
            c = rng.choice(NC, p=MIX_BASE)
            counts[CLS[c]] += 1
            space_det += SPACE[c]
        else:
            lab = COCO_LABELS[rng.integers(0, len(COCO_LABELS))]
            counts[lab] += 1
            space_det += 1.0
    score = sum(prof.weight_of(lab) * n for lab, n in counts.items())
    if prof.labels == 'uvh':
        g = [counts['2W'] + counts['BIC'], counts['3W'], counts['CAR'] + counts['LCV'],
             counts['BUS'] + counts['TRK']]
        raw = sum(counts.values())
    else:
        g = [counts['motorcycle'] + counts['bicycle'], 0, counts['car'], counts['bus'] + counts['truck']]
        raw = sum(counts.values())
    fill = space_det * CAR_SPACE_M / (2 * CAM_RANGE)
    return dict(score=float(score), groups=g, fill=float(fill), n=int(raw))


def webster_plan(net, qb, mix=MIX_BASE, c_min=60, c_max=180):
    """Webster fixed-time plan from expected approach flows (design PCU)."""
    q = net.analytic_flows(qb)
    pcu_per_veh = float((mix * W_DESIGN).sum())
    Cs, ys = [], []
    for j in range(net.nJ):
        y = np.array([q[4 * j + d] * pcu_per_veh / S_APPROACH_PCU for d in range(4)])
        Y = y.sum()
        C0 = (1.5 * L_CYCLE + 5.0) / max(1e-3, 1.0 - Y) if Y < 0.98 else c_max
        Cs.append(C0)
        ys.append(y)
    C = int(round(min(max(max(Cs), c_min), c_max) / 5.0) * 5)
    plans = []
    for j in range(net.nJ):
        y = ys[j]
        ge = (C - L_CYCLE) * y / y.sum()
        g = np.maximum(ge + AMBER_USE, G_MIN)
        g = integer_split(g, C - INTERGREEN * N_PHASE)
        plans.append([(d, int(g[d])) for d in (0, 1, 2, 3)])
    return C, plans, [float(y.sum()) for y in ys]


def integer_split(g, total):
    """Round a green vector to integers summing exactly to total, keeping >= G_MIN."""
    g = np.asarray(g, dtype=float)
    g = g * (total / g.sum())
    base = np.floor(g).astype(int)
    rem = total - base.sum()
    frac = g - base
    order = np.argsort(-frac)
    for k in range(int(rem)):
        base[order[k % len(base)]] += 1
    # enforce minimum
    for _ in range(20):
        low = base < G_MIN
        if not low.any():
            break
        deficit = (G_MIN - base[low]).sum()
        base[low] = G_MIN
        donors = np.argsort(-base)
        k = 0
        while deficit > 0:
            dnr = donors[k % len(base)]
            if base[dnr] > G_MIN:
                base[dnr] -= 1
                deficit -= 1
            k += 1
    return base
