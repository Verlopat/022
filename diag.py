"""Diagnostics: value of perfect link state for cycle-based control vs actuated."""
import sys
import numpy as np
import simcore as sc
import controllers as ct
import train


class LinkOracleModel:
    """Pretends to be the estimator but reads the true link state from the simulator."""

    def __init__(self, sim_ref):
        self.sim_ref = sim_ref

    def forward(self, X):
        sim = self.sim_ref[0]
        t = sim._t_now
        y = np.zeros(12)
        arr = np.zeros(12)
        for a in range(12):
            q = sim.lane_pcu[a][0] + sim.lane_pcu[a][1]
            mv = sum(sim.v_w[v] for (tff, v) in sim.moving[a] if tff <= t + ct.H_TARGET)
            y[a] = q + mv
            arr[a] = mv
        psi = np.array([min(1.0, arr[4 * j:4 * j + 4].sum() / ct.H_TARGET) for j in range(3)])

        class _T:
            def __init__(s, d):
                s.data = d
        return _T(y[None] / ct.Y_SCALE), _T(psi[None])


class Wrapped(ct.GATCoordinated):
    def plan(self, sim, t):
        sim._t_now = t
        return super().plan(sim, t)


def run(ctrl_factory, scen, seeds):
    out = []
    for s in seeds:
        ref = [None]
        c = ctrl_factory(ref)
        sim = sc.Sim(scen, s, c, perception=sc.make_profiles()['P1'])
        ref[0] = sim
        out.append(sim.run()['delay'])
    return np.mean(out)


if __name__ == '__main__':
    profs = sc.make_profiles()
    seeds = list(range(500, 506))
    gat = train.load_model('gat_full')
    for scen in ['S1', 'S2', 'S3', 'S4', 'S5']:
        r_or = run(lambda ref: Wrapped(profs['P1'], LinkOracleModel(ref), kappa=0.9, rho0=0.5), scen, seeds)
        r_or0 = run(lambda ref: Wrapped(profs['P1'], LinkOracleModel(ref), kappa=0.0, rho0=0.5), scen, seeds)
        r_gat = run(lambda ref: ct.GATCoordinated(profs['P1'], gat, kappa=0.9, rho0=0.5), scen, seeds)
        r_act = run(lambda ref: ct.Actuated(gmax=90, gap=2.0), scen, seeds)
        print(f'{scen}: link-oracle cyc {r_or:6.1f} (no guard {r_or0:6.1f})  GATC {r_gat:6.1f}  ACT {r_act:6.1f}', flush=True)
