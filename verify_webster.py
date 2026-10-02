"""Verification: simulated control delay at an isolated fixed-time junction vs Webster's formula."""
import json
import numpy as np
import simcore as sc


class FixedPlan:
    cyclic = True
    name = 'FIX'

    def __init__(self, plans):
        self.plans = plans

    def reset(self, sim):
        pass

    def plan(self, sim, t):
        return self.plans


def isolated_net():
    net = sc.Network()
    net.boundary = [0, 1, 2, 3]
    for a in range(12):
        net.turn[a] = (0.0, 1.0, 0.0)
        for m in sc.MOVES:
            net.dest[a][m] = ('exit', 'X')
    return net


def webster_delay(q, C, ge, s):
    lam = ge / C
    x = q / (lam * s)
    d1 = C * (1 - lam) ** 2 / (2 * (1 - lam * x))
    d2 = x ** 2 / (2 * q * (1 - x))
    d3 = 0.65 * (C / q ** 2) ** (1.0 / 3.0) * x ** (2 + 5 * lam)
    return d1 + d2 - d3, x


def run():
    C = 96
    g = 19
    plans = [[(d, g) for d in range(4)]] * 3
    ge = g - 1.0  # start-up lost 2 s, 1 s of amber used
    s = 1.0       # veh/s per approach (2 lanes x 1800 veh/h)
    mix = np.zeros(sc.NC)
    mix[3] = 1.0  # passenger cars only
    out = []
    for x_target in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        q = x_target * (ge / C) * s  # veh/s
        dw, x = webster_delay(q, C, ge, s)
        sims = []
        for seed in range(16):
            net = isolated_net()
            scen = dict(name='verif', factor=1.0, incident=None, mix=mix, flat=True, hw_sigma=0.0)
            sim = sc.Sim(scen, 1000 + seed, FixedPlan(plans), net=net, T_end=8 * 3600 + 900, warm=900,
                         meas_end=8 * 3600, demand_override={a: q * 3600 for a in range(4)})
            r = sim.run()
            ad = np.array(r['app_delay'][:4])
            sims.append(ad.mean())
        sims = np.array(sims)
        out.append(dict(x=float(x), q_vph=float(q * 3600), webster=float(dw), sim_mean=float(sims.mean()),
                        sim_sd=float(sims.std(ddof=1)), sim_ci=float(1.96 * sims.std(ddof=1) / np.sqrt(len(sims)))))
        print(f"x={x:.2f} q={q*3600:5.0f} veh/h  Webster {dw:6.1f} s  sim {sims.mean():6.1f} +- {sims.std(ddof=1):4.1f}")
    json.dump(out, open('results/verify_webster.json', 'w'), indent=1)


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    run()
