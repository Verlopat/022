"""Tuning-seed evaluation of early DQN checkpoints for the learning-curve figure."""
import json
import numpy as np
import experiments as ex

SCEN = ['S1', 'S2', 'S3', 'S4', 'S5']
jobs = [('dqn_d10_s0', [25, 50, 75, 100, 125], True), ('dqno_d10_g8', [25, 50, 75], False)]
specs = []
for tag, cks, v1 in jobs:
    for ck in cks:
        for sc in SCEN:
            for sd in range(500, 506):
                specs.append(dict(ctrl='DQN', model=tag, ckpt=ck, delta=10, tag=f'{tag}@{ck}', scenario=sc, seed=sd,
                                  profile='P1', obs_v1=v1))
res = ex.run_batch(specs, 'results/dqn_curve_extra.jsonl', model_tags=())
summ = {}
for r in res:
    summ.setdefault(r['tag'], {}).setdefault(r['scenario'], []).append(r['delay'])
tab = {k: float(np.mean([np.mean(v[s]) for s in SCEN])) for k, v in summ.items()}
json.dump(tab, open('results/dqn_curve_extra_summary.json', 'w'), indent=1)
for k in sorted(tab):
    print(k, round(tab[k], 1))
