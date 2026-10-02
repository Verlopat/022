"""Control outcome of every estimator and training seed (GATC-S and GATC modes, test seeds 1-20)."""
import json
import sys

import numpy as np

import experiments as ex

SCEN = ['S1', 'S2', 'S3', 'S4', 'S5']
CFG = json.load(open('results/final_configs.json'))
KINDS = ['gat_full', 'gat_local', 'gcn_full', 'mlp_none']
TAGS = [k + s for k in KINDS for s in ('', '_s1', '_s2')]

if __name__ == '__main__':
    if sys.argv[1] == 'run':
        specs = []
        for mode in ['GATC-S', 'GATC']:
            for tag in TAGS:
                for sc in SCEN:
                    for sd in range(1, 21):
                        s = dict(CFG[mode])
                        s.update(model=tag, tag=f'{mode}|{tag}', scenario=sc, seed=sd, profile='P1')
                        specs.append(s)
        ex.run_batch(specs, 'results/estctrl.jsonl', model_tags=TAGS)
    else:
        res = ex.load_results('results/estctrl.jsonl')
        out = {}
        for mode in ['GATC-S', 'GATC']:
            for k in KINDS:
                per_seed = []
                for s in ('', '_s1', '_s2'):
                    tg = f'{mode}|{k}{s}'
                    m = np.mean([np.mean([r['delay'] for r in res if r['tag'] == tg and r['scenario'] == sc])
                                 for sc in SCEN])
                    per_seed.append(float(m))
                out[f'{mode}|{k}'] = dict(mean=float(np.mean(per_seed)), sd=float(np.std(per_seed, ddof=1)),
                                          seeds=per_seed)
                print(mode, k, round(np.mean(per_seed), 1), '+-', round(np.std(per_seed, ddof=1), 1), per_seed)
        json.dump(out, open('results/estctrl_summary.json', 'w'), indent=1)
