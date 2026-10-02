"""Batch experiment runner (multiprocessing) for all controller/scenario/seed combinations."""
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

import controllers as ct
import simcore as sc

_MODELS = {}
_PROFS = None


def _init_worker(model_tags):
    global _MODELS, _PROFS
    import train
    _PROFS = sc.make_profiles()
    for tag in model_tags:
        if os.path.exists(f'models/{tag}.pkl'):
            _MODELS[tag] = train.load_model(tag)


def build_controller(spec):
    name = spec['ctrl']
    prof = _PROFS[spec.get('profile', 'P1')]
    if 'recall_mult' in spec or 'extra_fp' in spec:
        import copy
        prof = copy.deepcopy(prof)
        prof.recall_mult = spec.get('recall_mult', 1.0)
        prof.extra_fp = spec.get('extra_fp', 0.0)
    if name == 'FT-W':
        plan = spec.get('plan', 'design')
        if plan == 'design':
            ps = 'S1' if spec['scenario'] == 'S1' else 'S2'
        else:
            ps = None
        return ct.FixedTimeWebster(c_max=spec.get('c_max', 160), plan_scenario=ps), prof
    if name == 'ACT':
        return ct.Actuated(gmax=spec.get('gmax', 90), gap=spec.get('gap', 2.0)), prof
    if name == 'MP-C':
        return ct.CyclicMaxPressure(prof, C=spec.get('C', 120)), prof
    if name == 'MP-A':
        return ct.AcyclicMaxPressure(prof, delta=spec.get('delta', 20), gmin=spec.get('gmin', 15),
                                     gmax=spec.get('gmax', 90)), prof
    if name == 'LPS':
        return ct.LocalProportional(prof, C=spec.get('C', 120), priority=spec.get('priority', True),
                                    weights=spec.get('weights', 'pcu'), tactical=spec.get('tactical', False),
                                    check_every=spec.get('check_every', 2),
                                    empty_needed=spec.get('empty_needed', 2)), prof
    if name == 'GATC':
        model = _MODELS[spec.get('model', 'gat_full')]
        return ct.GATCoordinated(prof, model, c_min=spec.get('c_min', 60), c_max=spec.get('c_max', 160),
                                 kappa=spec.get('kappa', 0.9), rho0=spec.get('rho0', 0.5),
                                 smooth=spec.get('smooth', 0.5), priority=spec.get('priority', True),
                                 adapt_cycle=spec.get('adapt_cycle', True), guard=spec.get('guard', True),
                                 tactical=spec.get('tactical', True), C_fixed=spec.get('C_fixed', 120),
                                 max_factor=spec.get('max_factor', 1.0), check_every=spec.get('check_every', 2),
                                 empty_needed=spec.get('empty_needed', 2),
                                 clear_cycle=spec.get('clear_cycle', True)), prof
    if name == 'DQN':
        import rl
        q, meta = rl.load_policy(spec['model'], spec.get('ckpt'))
        return rl.DQNController(prof, q, delta=spec['delta'], gmax=spec.get('gmax', 90),
                                gmin=meta.get('gmin', 8), oracle=meta.get('oracle', False),
                                obs_v1=spec.get('obs_v1', meta.get('obs_v1', False))), prof
    raise ValueError(name)


def run_spec(spec):
    t0 = time.time()
    ctrl, prof = build_controller(spec)
    sim = sc.Sim(spec['scenario'], spec['seed'], ctrl, perception=prof)
    r = sim.run()
    delays = r.pop('delays')
    out = dict(spec)
    out.update(r)
    out['runtime_s'] = time.time() - t0
    if spec.get('keep_series'):
        out['q_series'] = np.array(sim.q_series).tolist()
        out['delays'] = delays.tolist()
        out['cycle_log'] = [(c[0], c[1], c[2]) for c in sim.cycle_log]
        if hasattr(ctrl, 'trace'):
            out['trace'] = [(t, j, C, psi.tolist(), yh.tolist(), dl.tolist()) for (t, j, C, psi, yh, dl) in ctrl.trace]
    if spec.get('keep_delays'):
        out['delays'] = delays.tolist()
    return out


def run_batch(specs, out_path, model_tags=('gat_full',), workers=2, chunksize=2):
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(list(model_tags),)) as pool:
        res = pool.map(run_spec, specs, chunksize=chunksize)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    print(f'{len(specs)} runs -> {out_path} in {time.time() - t0:.0f}s', flush=True)
    return res


def load_results(path):
    return [json.loads(l) for l in open(path)]
