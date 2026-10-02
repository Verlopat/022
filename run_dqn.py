"""DQN-C baseline: checkpoint selection on tuning seeds 500-505 and test-phase runs on seeds 1-20."""
import json
import os
import pickle
import sys

import numpy as np

import experiments as ex

SCEN = ['S1', 'S2', 'S3', 'S4', 'S5']
TUNE = list(range(500, 506))
TEST = list(range(1, 21))


def ckpts(tag):
    d = pickle.load(open(f'models/{tag}.pkl', 'rb'))
    return sorted(d['ckpts'].keys()), d['delta']


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'select':
        tags = sys.argv[2].split(',')
        first = int(sys.argv[3]) if len(sys.argv) > 3 else 150
        specs = []
        for tag in tags:
            cks, delta = ckpts(tag)
            for ck in cks:
                if ck < first:
                    continue
                for sc in SCEN:
                    for sd in TUNE:
                        specs.append(dict(ctrl='DQN', model=tag, ckpt=ck, delta=delta, tag=f'{tag}@{ck}',
                                          scenario=sc, seed=sd, profile='P1'))
        out = f'results/dqn_select_{"_".join(tags)}.jsonl'
        res = ex.run_batch(specs, out, model_tags=())
        summ = {}
        for r in res:
            summ.setdefault(r['tag'], {}).setdefault(r['scenario'], []).append(r['delay'])
        table = {k: float(np.mean([np.mean(v[s]) for s in SCEN])) for k, v in summ.items()}
        for k in sorted(table, key=table.get):
            print(k, round(table[k], 1), {s: round(float(np.mean(summ[k][s])), 1) for s in SCEN})
        json.dump(table, open(out.replace('.jsonl', '_summary.json'), 'w'), indent=1)
    elif cmd == 'test':
        # python3 run_dqn.py test <tag> <ckpt> <out_tag> [profiles]
        tag, ck, out_tag = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        profiles = sys.argv[5].split(',') if len(sys.argv) > 5 else ['P1']
        obs_v1 = len(sys.argv) > 6 and sys.argv[6] == 'v1'
        _, delta = ckpts(tag)
        specs = []
        for p in profiles:
            for sc in SCEN:
                for sd in TEST:
                    specs.append(dict(ctrl='DQN', model=tag, ckpt=ck, delta=delta, tag=out_tag, scenario=sc, seed=sd,
                                      profile=p, keep_delays=(sc in ('S2', 'S3') and p == 'P1'), obs_v1=obs_v1))
            if p == 'P1':
                for sd in TEST:
                    specs.append(dict(ctrl='DQN', model=tag, ckpt=ck, delta=delta, tag=out_tag, scenario='S2HV',
                                      seed=sd, profile='P1', obs_v1=obs_v1))
        out = f'results/dqn_test_{out_tag.replace("-", "")}_{"".join(profiles)}.jsonl'
        ex.run_batch(specs, out, model_tags=())
    elif cmd == 'series':
        # python3 run_dqn.py series tag:ckpt:out_tag[:v1],...
        specs = []
        for item in sys.argv[2].split(','):
            parts = item.split(':')
            tag, ck, out_tag = parts[0], int(parts[1]), parts[2]
            v1 = len(parts) > 3 and parts[3] == 'v1'
            _, delta = ckpts(tag)
            specs += [dict(ctrl='DQN', model=tag, ckpt=ck, delta=delta, tag=out_tag, scenario=sc, seed=1, profile='P1',
                           keep_series=True, obs_v1=v1) for sc in ['S3', 'S4', 'S5']]
        ex.run_batch(specs, 'results/dqn_series.jsonl', model_tags=())
