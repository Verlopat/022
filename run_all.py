"""Test-phase experiments (seeds 1-20) with configurations frozen after tuning."""
import json
import sys

import experiments as ex

SCEN = ['S1', 'S2', 'S3', 'S4', 'S5']
SEEDS = list(range(1, 21))
CFG = json.load(open('results/final_configs.json'))
MODELS = ('gat_full', 'gat_local', 'gcn_full', 'mlp_none')


def spec(tag, base, scen, seed, profile='P1', **extra):
    s = dict(CFG[base])
    s.update(extra)
    s.update(tag=tag, scenario=scen, seed=seed, profile=profile)
    return s


def main_specs():
    out = []
    for tag in ['FT-W', 'FT-W*', 'ACT', 'MP-C', 'MP-A', 'LPS', 'LPS-T', 'GATC-S', 'GATC']:
        for sc in SCEN:
            for sd in SEEDS:
                out.append(spec(tag, tag, sc, sd, keep_delays=(sc in ('S2', 'S3'))))
    return out


def ablation_specs():
    variants = {
        'GATC (GCN)': ('GATC', dict(model='gcn_full')),
        'GATC (MLP)': ('GATC', dict(model='mlp_none')),
        'GATC (GAT-local)': ('GATC', dict(model='gat_local')),
        'GATC no guard': ('GATC', dict(kappa=0.0)),
        'GATC fixed cycle': ('GATC', dict(adapt_cycle=False, C_fixed=CFG['LPS-T']['C'])),
        'GATC fixed order': ('GATC', dict(priority=False)),
        'GATC-S fixed cycle': ('GATC-S', dict(adapt_cycle=False, C_fixed=CFG['LPS']['C'])),
        'GATC-S fixed order': ('GATC-S', dict(priority=False)),
        'GATC-S (GCN)': ('GATC-S', dict(model='gcn_full')),
        'GATC-S (MLP)': ('GATC-S', dict(model='mlp_none')),
        'GATC-S (GAT-local)': ('GATC-S', dict(model='gat_local')),
        'GATC-S no guard': ('GATC-S', dict(kappa=0.0)),
        'LPS count': ('LPS', dict(weights='count')),
    }
    out = []
    for tag, (base, extra) in variants.items():
        for sc in SCEN:
            for sd in SEEDS:
                out.append(spec(tag, base, sc, sd, **extra))
    for tag, base, extra in [('LPS', 'LPS', {}), ('LPS count', 'LPS', dict(weights='count')),
                             ('GATC', 'GATC', {}), ('GATC-S', 'GATC-S', {}), ('FT-W', 'FT-W', {})]:
        for sd in SEEDS:
            out.append(spec(tag, base, 'S2HV', sd, **extra))
    return out


def perception_specs():
    out = []
    for prof in ['P0', 'P2', 'P3']:
        for tag in ['MP-A', 'LPS', 'LPS-T', 'GATC-S', 'GATC']:
            for sc in SCEN:
                for sd in SEEDS:
                    out.append(spec(tag, tag, sc, sd, profile=prof))
    return out


def sweep_specs():
    out = []
    for tag in ['GATC', 'GATC-S', 'LPS']:
        for rm in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            for fp in [0.0, 1.0, 2.0, 3.0]:
                for sd in range(1, 11):
                    out.append(spec(tag, tag, 'S2', sd, recall_mult=rm, extra_fp=fp))
    return out


def series_specs():
    out = []
    for tag in ['FT-W', 'ACT', 'MP-A', 'LPS', 'GATC-S', 'GATC']:
        for sc in ['S3', 'S4', 'S5']:
            out.append(spec(tag, tag, sc, 1, keep_series=True))
    return out


if __name__ == '__main__':
    phase = sys.argv[1]
    fn = dict(main=main_specs, ablation=ablation_specs, perception=perception_specs, sweep=sweep_specs,
              series=series_specs)[phase]
    ex.run_batch(fn(), f'results/{phase}.jsonl', model_tags=MODELS)
