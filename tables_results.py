"""Result tables and a numbers file for the manuscript text."""
import json
import os

import numpy as np

import stats as S
from tables_static import md_table

SCEN = ['S1', 'S2', 'S3', 'S4', 'S5']
CTRLS = ['FT-W', 'FT-W*', 'MP-C', 'LPS', 'GATC-S', 'ACT', 'MP-A', 'DQN-C', 'LPS-T', 'GATC', 'DQN-O']
REGIME = {'FT-W': 'None', 'FT-W*': 'None (true flows)', 'MP-C': 'Snapshot', 'LPS': 'Snapshot', 'GATC-S': 'Snapshot',
          'ACT': 'Continuous (loops)', 'MP-A': 'Continuous', 'DQN-C': 'Continuous', 'LPS-T': 'Continuous',
          'GATC': 'Continuous', 'DQN-O': 'Exact state'}
REFERENCE = ('FT-W*', 'DQN-O')
DQN_P1 = 'results/dqn_test_DQNC_P1.jsonl'
DQNO_P1 = 'results/dqn_test_DQNO_P1.jsonl'
DQN_PX = 'results/dqn_test_DQNC_P0P2P3.jsonl'
N = {}


def load_main():
    res = S.load('results/main.jsonl')
    for f in (DQN_P1, DQNO_P1):
        if os.path.exists(f):
            res += [r for r in S.load(f) if r['scenario'] in SCEN]
    return res


def pct(a, b):
    return 100.0 * (a - b) / b


def main_tables():
    res = load_main()
    tab = S.table(res, 'delay')
    header = ['Controller', 'Sensing'] + [f'{s}' for s in SCEN] + ['Mean']
    rows = []
    best = {s: min(tab[(c, s)].mean() for c in CTRLS if c not in REFERENCE) for s in SCEN}
    for c in CTRLS:
        cells = []
        for s in SCEN:
            x = tab[(c, s)]
            txt = S.fmt_pm(x)
            if abs(x.mean() - best[s]) < 1e-9:
                txt = txt + ' †'
            cells.append(txt)
        rows.append([c, REGIME[c]] + cells + [f'{np.mean([tab[(c, s)].mean() for s in SCEN]):.1f}'])
    t8 = md_table(header, rows, 'Table 8. Average waiting time per vehicle (s/veh, mean ± SD over 20 paired seeds). † marks the lowest value among deployable controllers; FT-W* (true scenario flows) and DQN-O (exact simulator state) are references that no camera deployment can reach.')
    for c in CTRLS:
        N[f'W_{c}_mean'] = float(np.mean([tab[(c, s)].mean() for s in SCEN]))
        for s in SCEN:
            N[f'W_{c}_{s}'] = float(tab[(c, s)].mean())
    # secondary metrics for S3 and S5
    mets = [('travel_time', 'Travel time (s)', 1), ('throughput', 'Throughput (veh/h)', 0),
            ('queue_pcu', 'Queue per arm (PCU)', 1), ('delay_p95', 'P95 waiting (s)', 0), ('jain', 'Jain index', 3),
            ('gur', 'GUR', 3), ('blocked_share', 'Blocked green (%)', 1)]
    header = ['Scenario', 'Controller'] + [m[1] for m in mets]
    rows = []
    for s in ['S3', 'S5']:
        for c in ['FT-W', 'LPS', 'GATC-S', 'ACT', 'MP-A', 'DQN-C', 'LPS-T', 'GATC', 'DQN-O']:
            cells = []
            for m, lab, nd in mets:
                v = S.table(res, m)[(c, s)].mean()
                if m == 'blocked_share':
                    v *= 100
                cells.append(f'{v:,.{nd}f}')
            rows.append([s, c] + cells)
    t9 = md_table(header, rows, 'Table 10. Secondary metrics in the oversaturated (S3) and incident (S5) scenarios (means over 20 seeds).')
    for s in SCEN:
        for c in CTRLS:
            for m in ['throughput', 'jain', 'gur', 'blocked_share', 'spill_share', 'queue_pcu', 'delay_p95',
                      'travel_time', 'mean_cycle']:
                N[f'{m}_{c}_{s}'] = float(S.table(res, m)[(c, s)].mean())
    # statistical tests
    comps = [('GATC-S', 'LPS'), ('GATC-S', 'FT-W'), ('GATC-S', 'MP-C'), ('GATC', 'ACT'), ('GATC', 'MP-A'),
             ('GATC', 'DQN-C'), ('GATC', 'FT-W'), ('GATC', 'LPS'), ('GATC', 'LPS-T'), ('GATC', 'DQN-O')]
    raw = []
    for a, b in comps:
        for s in SCEN:
            x, y = S.paired(res, a, b, s)
            p, r = S.wilcoxon_rb(x, y)
            raw.append((a, b, s, pct(x.mean(), y.mean()), p, r))
    adj = S.holm([z[4] for z in raw])
    header = ['Comparison'] + SCEN
    rows = []
    k = 0
    for a, b in comps:
        cells = []
        for s in SCEN:
            _, _, _, d, p, r = raw[k]
            pa = adj[k]
            N[f'd_{a}_vs_{b}_{s}'] = d
            N[f'p_{a}_vs_{b}_{s}'] = float(pa)
            N[f'r_{a}_vs_{b}_{s}'] = r
            mark = '' if pa < 0.05 else ' ns'
            cells.append(f'{d:+.1f}% ({S.fmt_p(pa)}; {r:+.2f}){mark}')
            k += 1
        rows.append([f'{a} vs {b}'] + cells)
    t10 = md_table(header, rows, f'Table 9. Paired comparisons of average waiting time: relative difference, Holm-adjusted Wilcoxon p-value over {len(raw)} tests, and matched-pairs rank-biserial correlation (negative favours the first controller); ns marks p ≥ 0.05.')
    return t8, t9, t10


def estimator_table():
    import train
    rows = []
    snap, raw = train.snapshot_baseline()
    ctrl = json.load(open('results/estctrl_summary.json')) if os.path.exists('results/estctrl_summary.json') else {}
    header = ['Estimator', 'Edges used', 'MAE Q+A (PCU)', 'R² Q+A', 'R² internal arms', 'MAE Q (PCU)', 'MAE A (PCU)',
              'R² ψ', 'GATC-S waiting (s/veh)', 'GATC waiting (s/veh)']
    rows.append(['Raw snapshot score', 'None', f"{raw['mae_y']:.2f}", f"{raw['r2_y']:.3f}", f"{raw['r2_y_internal']:.3f}",
                 '-', '-', '-', '-', '-'])
    rows.append(['Calibrated snapshot', 'None', f"{snap['mae_y']:.2f}", f"{snap['r2_y']:.3f}", f"{snap['r2_y_internal']:.3f}",
                 '-', '-', f"{snap['r2_psi']:.3f}", '-', '-'])
    N['est_raw'] = raw
    N['est_snap'] = snap
    names = [('mlp_none', 'MLP', 'Self-loops only'), ('gat_local', 'GAT-local', 'Intra-junction and hub'),
             ('gcn_full', 'GCN', 'All four sets'), ('gat_full', 'GAT (proposed)', 'All four sets')]
    ms = json.load(open('results/estimator_seeds.json'))
    for tag, lab, edges in names:
        m = ms[tag]
        f = lambda k, nd: f"{m[k]['mean']:.{nd}f} ± {m[k]['sd']:.{nd}f}"
        cs = ctrl.get(f'GATC-S|{tag}')
        cg = ctrl.get(f'GATC|{tag}')
        fc = lambda c: f"{c['mean']:.1f} ± {c['sd']:.1f}" if c else '-'
        rows.append([lab, edges, f('mae_y', 2), f('r2_y', 3), f('r2_y_internal', 3), f('mae_q', 2), f('mae_a', 2),
                     f('r2_psi', 3), fc(cs), fc(cg)])
        N[f'est_{tag}'] = {k: v['mean'] for k, v in m.items()}
        N[f'est_{tag}_sd'] = {k: v['sd'] for k, v in m.items()}
        if cs:
            N[f'estctrl_S_{tag}'] = cs
            N[f'estctrl_T_{tag}'] = cg
    cap = ('Table 7. Accuracy of next-cycle demand estimators on 80 held-out episodes (5,349 graph snapshots) and the '
           'resulting average waiting time (mean over S1-S5 and 20 test seeds) when each estimator drives GATC-S or GATC; '
           'learned estimators are mean ± SD over three training seeds.')
    return md_table(header, rows, cap)


def ablation_table():
    res = S.load('results/ablation.jsonl') + S.load('results/main.jsonl')
    tab = S.table(res, 'delay')
    groups = [('GATC', ['GATC no guard', 'GATC fixed cycle', 'GATC fixed order', 'LPS-T']),
              ('GATC-S', ['GATC-S no guard', 'GATC-S fixed cycle', 'GATC-S fixed order', 'LPS'])]
    label = {'LPS-T': 'Camera scores instead of estimator (LPS-T)', 'LPS': 'Camera scores instead of estimator (LPS)'}
    header = ['Variant'] + SCEN + ['Mean change']
    rows = []
    abl_fig = []
    for parent, vs in groups:
        rows.append([f'{parent} (full)'] + [f'{tab[(parent, s)].mean():.1f}' for s in SCEN] + ['0.0%'])
        for v in vs:
            ch = [pct(tab[(v, s)].mean(), tab[(parent, s)].mean()) for s in SCEN]
            # paired bootstrap CI of mean relative change across scenarios
            diffs = np.array([pct(tab[(v, s)], tab[(parent, s)].mean()) for s in SCEN])  # 5 x 20
            mc = float(np.mean(ch))
            rng = np.random.default_rng(1)
            boots = []
            for _ in range(10000):
                idx = rng.integers(0, 20, 20)
                boots.append(np.mean([pct(tab[(v, s)][idx].mean(), tab[(parent, s)][idx].mean()) for s in SCEN]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            name = label.get(v, v.replace(parent + ' ', '').replace('(', 'estimator: ').replace(')', ''))
            name = name if v in label else name[0].upper() + name[1:]
            rows.append([f'  {name}'] + [f'{tab[(v, s)].mean():.1f} ({c:+.1f}%)' for s, c in zip(SCEN, ch)] + [f'{mc:+.1f}%'])
            abl_fig.append((f'{parent}: {name}', mc, lo, hi))
            N[f'abl_{v}'] = mc
            for s, c in zip(SCEN, ch):
                N[f'abl_{v}_{s}'] = c
    # PCU weights vs counts on S2 and S2HV
    extra = []
    for v, parent in [('LPS count', 'LPS')]:
        for s in ['S2', 'S2HV']:
            a = tab[(v, s)]
            b = tab[(parent, s)]
            p, r = S.wilcoxon_rb(a, b)
            N[f'count_{s}'] = pct(a.mean(), b.mean())
            N[f'count_p_{s}'] = p
            extra.append((s, a.mean(), b.mean(), pct(a.mean(), b.mean()), p))
    t = md_table(header, rows, 'Table 11. Ablation of GATC and GATC-S: average waiting time (s/veh) and change relative to the full controller, 20 seeds per cell (estimator variants are compared in Table 7).')
    return t, abl_fig, extra


def perception_table():
    res = S.load('results/perception.jsonl') + [r for r in S.load('results/main.jsonl') if r['profile'] == 'P1']
    ctrls = ['LPS', 'GATC-S', 'MP-A', 'LPS-T', 'GATC']
    if os.path.exists(DQN_PX) and os.path.exists(DQN_P1):
        res += S.load(DQN_PX) + [r for r in S.load(DQN_P1) if r['scenario'] in SCEN]
        ctrls = ['LPS', 'GATC-S', 'MP-A', 'DQN-C', 'LPS-T', 'GATC']
    header = ['Controller'] + ['P0 oracle', 'P1 UVH-26', 'P2 COCO v8m', 'P3 COCO v8n', 'P3 vs P1']
    rows = []
    for c in ctrls:
        vals = []
        for p in ['P0', 'P1', 'P2', 'P3']:
            v = np.mean([np.mean([r['delay'] for r in res if S.tag_of(r) == c and r['profile'] == p and r['scenario'] == s])
                         for s in SCEN])
            vals.append(v)
            N[f'perc_{c}_{p}'] = float(v)
        rows.append([c] + [f'{v:.1f}' for v in vals] + [f'{pct(vals[3], vals[1]):+.1f}%'])
    return md_table(header, rows, 'Table 12. Average waiting time (s/veh, mean over S1-S5 and 20 seeds) under the four detector profiles.')


if __name__ == '__main__':
    os.makedirs('paper/tables', exist_ok=True)
    t8, t9, t10 = main_tables()
    open('paper/tables/main.md', 'w').write(t8)
    open('paper/tables/secondary.md', 'w').write(t9)
    open('paper/tables/stats.md', 'w').write(t10)
    open('paper/tables/estimator.md', 'w').write(estimator_table())
    if os.path.exists('results/ablation.jsonl'):
        t, abl_fig, extra = ablation_table()
        open('paper/tables/ablation.md', 'w').write(t)
        json.dump(abl_fig, open('results/ablation_fig.json', 'w'))
        print('count vs pcu', extra)
    if os.path.exists('results/perception.jsonl'):
        open('paper/tables/perception.md', 'w').write(perception_table())
    json.dump(N, open('results/numbers.json', 'w'), indent=1, default=float)
    print(open('paper/tables/stats.md').read())
