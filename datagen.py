"""Generate estimator training data with an exploratory behaviour policy."""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

import controllers as ct
import simcore as sc


def episode(seed):
    rng = np.random.default_rng([seed, 99])
    scen = dict(name='train', factor=float(rng.uniform(0.5, 1.35)), eb=float(rng.uniform(0.6, 1.6)),
                wb=float(rng.uniform(0.6, 1.4)), incident=None,
                mix='hv' if rng.random() < 0.2 else 'base')
    if rng.random() < 0.3:
        a = int(rng.integers(0, 12))
        t0 = int(rng.uniform(900, 2400))
        scen['incident'] = (a, int(rng.integers(0, 2)), t0, t0 + int(rng.uniform(600, 1500)))
    profs = sc.make_profiles()
    pk = rng.choice(['P0', 'P1', 'P2', 'P3'], p=[0.10, 0.60, 0.15, 0.15])
    prof = profs[pk]
    ctrl = ct.RandomizedBehaviour(prof, tactical=bool(rng.random() < 0.6), log_prob=0.45)
    sim = sc.Sim(scen, seed, ctrl, perception=prof, record_passages=True)
    sim.run()
    times = np.array(ctrl.log_t, dtype=float)
    X = np.array(ctrl.log_X)
    keep = times + ct.H_TARGET < sim.T_end - 60
    Y, psi, Q, A = ct.targets_from_passages(sim, times[keep])
    return dict(seed=seed, X=X[keep], Y=Y, psi=psi, Q=Q, A=A, t=times[keep], profile=str(pk), scen=scen)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    t0 = time.time()
    with Pool(2) as pool:
        eps = pool.map(episode, [10000 + i for i in range(n)], chunksize=4)
    os.makedirs('data', exist_ok=True)
    with open('data/train_episodes.pkl', 'wb') as f:
        pickle.dump(eps, f)
    ns = sum(e['X'].shape[0] for e in eps)
    print(f'{n} episodes, {ns} samples, {time.time() - t0:.0f}s')
