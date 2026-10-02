"""Hyper-parameter selection for every controller on tuning seeds (500-505), disjoint from test seeds."""
import itertools
import json
from collections import defaultdict

import numpy as np

import experiments as ex

SCEN = ['S1', 'S2', 'S3', 'S4', 'S5']
SEEDS = list(range(500, 506))

grids = {
    'FT-W': [dict()],
    'ACT': [dict(gmax=g, gap=p) for g, p in itertools.product([60, 90, 120], [1.5, 2.0, 3.0])],
    'MP-C': [dict(C=c) for c in [80, 100, 120, 150]],
    'MP-A': [dict(delta=d, gmin=g) for d, g in itertools.product([10, 20, 30], [8, 15, 20])],
    'LPS': [dict(C=c) for c in [90, 120, 150]],
    'GATC': [dict(kappa=k, rho0=r, c_max=c) for k, r, c in itertools.product([0.0, 0.5, 0.9], [0.5, 0.7],
                                                                             [120, 160])],
    'GATC3': [dict(kappa=k, rho0=r, max_factor=mf, c_max=cm, priority=True, check_every=1)
              for k, r, mf, cm in itertools.product([0.0, 0.5, 0.9], [0.5, 0.7], [2.0, 2.5, 3.0], [240, 300])],
    'ACT2': [dict(gmax=g, gap=p) for g, p in itertools.product([75, 90, 105], [1.0, 1.5])],
    'MPA2': [dict(delta=d, gmin=g) for d, g in itertools.product([15, 20], [20, 25, 30])],
    'FTW2': [dict(c_max=c) for c in [160, 200, 240]],
    'GATC4': [dict(kappa=0.5, rho0=0.7, max_factor=mf, c_max=cm, priority=True, check_every=1, empty_needed=en)
              for mf, cm, en in itertools.product([2.0, 2.5, 3.0], [200, 240], [1, 2])],
    'ACT3': [dict(gmax=g, gap=p) for g, p in itertools.product([60, 75, 90], [0.5, 1.0])],
    'LPST': [dict(C=c, tactical=True, check_every=1, empty_needed=en) for c, en in itertools.product([120, 160, 200, 240], [1, 2])],
    'GATCS': [dict(kappa=k, rho0=r, max_factor=mf, c_max=cm, tactical=False)
              for k, r, mf, cm in itertools.product([0.0, 0.5, 0.9], [0.5, 0.7], [1.0, 1.25, 1.5], [160, 200, 240])],
    'NGS': [dict(kappa=k, rho0=r, max_factor=mf, c_max=cm, tactical=False, clear_cycle=cc)
            for k, r, mf, cm, cc in itertools.product([0.0, 0.5, 0.9], [0.5, 0.7], [1.0, 1.25], [160, 200, 240], [True, False])],
    'NGT': [dict(kappa=k, rho0=0.7, max_factor=mf, c_max=cm, tactical=True, check_every=1, empty_needed=1, clear_cycle=cc)
            for k, mf, cm, cc in itertools.product([0.0, 0.5], [1.5, 2.0, 2.5, 3.0], [200, 240, 300], [True, False])],
    'LPST2': [dict(C=c, tactical=True, check_every=1, empty_needed=1) for c in [240, 300]],
    'FTWD': [dict(c_max=c, plan='design') for c in [160, 200, 240]],
    'FTWO': [dict(c_max=c, plan='oracle') for c in [160, 200, 240]],
    'ACT4': [dict(gmax=g, gap=p) for g, p in itertools.product([75, 90], [0.0, 0.25])],
    'NGT2': [dict(kappa=k, rho0=0.7, max_factor=mf, c_max=cm, tactical=True, check_every=1, empty_needed=2, clear_cycle=cc)
             for k, mf, cm, cc in itertools.product([0.0, 0.5], [2.0, 2.5, 3.0], [240, 300], [True, False])],
    'LPST3': [dict(C=c, tactical=True, check_every=1, empty_needed=2) for c in [200, 240, 300]],
    'ACT5': [dict(gmax=g, gap=2.0) for g in [60, 75, 90, 105]],
    'GATC2': [dict(kappa=0.5, rho0=0.7, max_factor=mf, c_max=cm, priority=pr, check_every=ce)
              for mf, cm, pr, ce in itertools.product([1.0, 1.5, 2.0], [160, 240], [True, False], [1, 2])],
}

if __name__ == '__main__':
    import sys
    which = sys.argv[1].split(',') if len(sys.argv) > 1 else list(grids.keys())
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    tag = '_'.join(w.replace('-', '') for w in which)
    grids = {k: v for k, v in grids.items() if k in which}
    specs = []
    for ctrl, grid in grids.items():
        for gi, params in enumerate(grid):
            for s in SCEN:
                for seed in SEEDS:
                    base = {'GATC2': 'GATC', 'GATC3': 'GATC', 'GATC4': 'GATC', 'GATCS': 'GATC', 'NGS': 'GATC', 'NGT': 'GATC', 'LPST2': 'LPS', 'FTWD': 'FT-W', 'FTWO': 'FT-W', 'ACT4': 'ACT', 'NGT2': 'GATC', 'LPST3': 'LPS', 'ACT5': 'ACT', 'ACT2': 'ACT', 'ACT3': 'ACT',
                            'MPA2': 'MP-A', 'FTW2': 'FT-W', 'LPST': 'LPS'}.get(ctrl, ctrl)
                    sp = dict(ctrl=base, scenario=s, seed=seed, profile='P1', cfg=gi, grid=ctrl)
                    sp.update(params)
                    specs.append(sp)
    res = ex.run_batch(specs, f'results/tuning_{tag}.jsonl', model_tags=('gat_full',), workers=workers)
    agg = defaultdict(list)
    for r in res:
        agg[(r['grid'], r['cfg'])].append(r)
    summary = {}
    for ctrl, grid in grids.items():
        rows = []
        for gi, params in enumerate(grid):
            rr = agg[(ctrl, gi)]
            per_s = {s: float(np.mean([x['delay'] for x in rr if x['scenario'] == s])) for s in SCEN}
            rows.append(dict(params=params, mean_delay=float(np.mean(list(per_s.values()))), per_scenario=per_s))
        rows.sort(key=lambda z: z['mean_delay'])
        summary[ctrl] = rows
        print(ctrl)
        for z in rows:
            print('   ', z['params'], round(z['mean_delay'], 1), {k: round(v, 1) for k, v in z['per_scenario'].items()})
    json.dump(summary, open(f'results/tuning_summary_{tag}.json', 'w'), indent=1)
