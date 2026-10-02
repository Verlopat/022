"""Extra training seeds for every estimator; aggregates held-out accuracy as mean and SD over seeds 0-2."""
import json
import sys

import numpy as np

import train

KINDS = [('gat', 'full'), ('gat', 'local'), ('gcn', 'full'), ('mlp', 'none')]

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        which = sys.argv[2].split(',')
        for w in which:
            kind, variant, s = w.split(':')
            train.train(kind, variant, seed=int(s), tag=f'{kind}_{variant}_s{s}', verbose=False)
    else:
        out = {}
        for kind, variant in KINDS:
            tags = [f'{kind}_{variant}'] + [f'{kind}_{variant}_s{s}' for s in (1, 2)]
            ms = [json.load(open(f'models/{t}_hist.json'))['metrics'] for t in tags]
            out[f'{kind}_{variant}'] = {k: dict(mean=float(np.mean([m[k] for m in ms])),
                                                sd=float(np.std([m[k] for m in ms], ddof=1)))
                                        for k in ms[0] if isinstance(ms[0][k], (int, float))}
        json.dump(out, open('results/estimator_seeds.json', 'w'), indent=1)
        for k, v in out.items():
            print(k, {m: round(v[m]['mean'], 3) for m in ['mae_y', 'r2_y', 'r2_y_internal', 'mae_psi', 'r2_psi']})
