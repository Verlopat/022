"""Result figures (one chart per image)."""
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import figstyle as fs
import stats as S

fs.setup()
PAL = fs.PAL
SCEN = ['S1', 'S2', 'S3', 'S4', 'S5']
SCEN_NAME = {'S1': 'S1 off-peak', 'S2': 'S2 peak', 'S3': 'S3 oversat.', 'S4': 'S4 tidal', 'S5': 'S5 incident'}
CTRL = ['FT-W', 'ACT', 'MP-A', 'DQN-C', 'LPS', 'LPS-T', 'GATC-S', 'GATC', 'DQN-O']
DQN_FILES = ('results/dqn_test_DQNC_P1.jsonl', 'results/dqn_test_DQNO_P1.jsonl')


def load_main():
    res = S.load('results/main.jsonl')
    for f in DQN_FILES:
        if os.path.exists(f):
            res += [r for r in S.load(f) if r['scenario'] in SCEN]
    return res


def fig_training():
    tags = [('gat_full', 'GAT (full graph)'), ('gat_local', 'GAT-local'), ('gcn_full', 'GCN'), ('mlp_none', 'MLP')]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for k, (t, lab) in enumerate(tags):
        h = json.load(open(f'models/{t}_hist.json'))['hist']
        ep = [x['epoch'] for x in h]
        ax.plot(ep, [x['val_loss'] for x in h], color=PAL[k], lw=1.9, label=f'{lab}, held-out')
    h = json.load(open('models/gat_full_hist.json'))['hist']
    ax.plot([x['epoch'] for x in h], [x['train_loss'] for x in h], color=PAL[6], lw=1.4,
            label='GAT (full graph), training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss, Eq. (8)')
    ax.legend(ncol=2, fontsize=7.6, loc='upper right')
    fs.save(fig, 'figs/fig07_training.png')


def fig_scatter():
    import train
    import controllers as ct
    (Xtr, Ytr, Ptr), (Xva, Yva, Pva), _ = train.load_data()
    model = train.load_model('gat_full')
    yh, ph = train.evaluate(model, Xva, Yva, Pva)
    y_true = Yva.sum(-1).ravel()
    y_gat = yh.sum(-1).ravel()
    s_tr = Xtr[:, :12, 0] * 15.0
    b = float((s_tr * Ytr.sum(-1)).sum() / (s_tr ** 2).sum())
    y_snap = (b * Xva[:, :12, 0] * 15.0).ravel()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_true), size=2500, replace=False)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = 160
    ax.plot([0, lim], [0, lim], color=fs.MUTED, lw=1.0, zorder=1)
    ax.scatter(y_true[idx], y_snap[idx], s=7, color=PAL[1], alpha=0.35, linewidths=0,
               label=f'calibrated snapshot (b = {b:.2f})', zorder=2)
    ax.scatter(y_true[idx], y_gat[idx], s=7, color=PAL[0], alpha=0.45, linewidths=0, label='GAT estimator', zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Observed next-cycle demand Q + A (PCU)')
    ax.set_ylabel('Predicted demand (PCU)')
    leg = ax.legend(loc='upper left', markerscale=2.5)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fs.save(fig, 'figs/fig08_scatter.png')


def fig_bars(res):
    tab = S.table(res, 'delay')
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    w = 0.095
    x = np.arange(len(SCEN))
    ymax = 270.0
    nc = len(CTRL)
    for k, c in enumerate(CTRL):
        m = np.array([tab[(c, s)].mean() for s in SCEN])
        ci = [S.boot_ci(tab[(c, s)], n=10000) for s in SCEN]
        lo = np.array([a for a, b in ci])
        hi = np.array([b for a, b in ci])
        mc = np.minimum(m, ymax)
        err = np.array([np.where(m > ymax, 0, m - lo), np.where(m > ymax, 0, hi - m)])
        xs = x + (k - (nc - 1) / 2) * w
        ax.bar(xs, mc, width=w * 0.92, color=fs.CTRL_COLOR[c], label=fs.CTRL_LABEL[c],
               yerr=err, error_kw=dict(ecolor=fs.INK2, elinewidth=0.8, capsize=1.4), zorder=2,
               hatch='///' if c == 'DQN-O' else None, edgecolor='white' if c == 'DQN-O' else None, linewidth=0)
        for xx, mm in zip(xs, m):
            if mm > ymax:
                ax.text(xx, ymax + 2, f'{mm:.0f}', ha='center', va='bottom', fontsize=6.4, color=fs.INK, rotation=90)
    ax.set_ylim(0, ymax + 40)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_NAME[s] for s in SCEN])
    ax.set_ylabel('Average waiting time (s/veh)')
    ax.legend(ncol=3, fontsize=7.0, loc='upper left', bbox_to_anchor=(0, 1.30))
    fs.save(fig, 'figs/fig09_bars.png')


def fig_heat_change(res):
    tab = S.table(res, 'delay')
    comps = [('GATC-S', 'LPS'), ('GATC-S', 'FT-W'), ('GATC', 'FT-W'), ('GATC', 'LPS'), ('GATC', 'ACT'),
             ('GATC', 'MP-A'), ('GATC', 'DQN-C'), ('GATC', 'LPS-T'), ('GATC', 'DQN-O')]
    M = np.array([[100 * (tab[(a, s)].mean() - tab[(b, s)].mean()) / tab[(b, s)].mean() for s in SCEN]
                  for a, b in comps])
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list('div', ['#104281', '#6da7ec', '#f0efec', '#f2a07b', '#b3261e'])
    vmax = 60.0
    im = ax.imshow(np.clip(M, -vmax, vmax), cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            col = 'white' if abs(v) > 0.55 * vmax else fs.INK
            ax.text(j, i, f'{v:+.1f}%', ha='center', va='center', fontsize=8, color=col)
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([SCEN_NAME[s] for s in SCEN], fontsize=8)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([f'{a} vs {b}' for a, b in comps], fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Change in average waiting time (%)', fontsize=8)
    cb.outline.set_visible(False)
    fs.save(fig, 'figs/fig10_heat_change.png')
    return M


def fig_series(series, scen='S5', fname='figs/fig11_series_s5.png'):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    show = ['FT-W', 'ACT', 'MP-A', 'DQN-C', 'LPS', 'GATC-S', 'GATC', 'DQN-O']
    cols = fs.CTRL_COLOR
    labs = {}
    if scen == 'S5':
        ax.axvspan(1500, 2700, color=PAL[7], alpha=0.10, zorder=0)
        ax.text(2100, 0.03, 'near-side lane of J3 west arm blocked', transform=ax.get_xaxis_transform(), ha='center',
                va='bottom', fontsize=7.5, color=fs.INK2)
    for tag in show:
        r = [x for x in series if S.tag_of(x) == tag and x['scenario'] == scen][0]
        q = np.array(r['q_series']).sum(axis=1)
        t = 600 + np.arange(len(q))
        k = 60
        qs = np.convolve(q, np.ones(k) / k, mode='valid')
        tv = t[k // 2: k // 2 + len(qs)]
        ax.plot(tv, qs, color=cols[tag], lw=1.7, label=fs.CTRL_LABEL.get(tag, tag))
    ax.set_xlabel('Simulation time (s)')
    ax.set_ylabel('Queued demand, all arms (PCU)')
    ax.set_xlim(600, 3600)
    ax.legend(fontsize=7.2, ncol=3, loc='upper left', bbox_to_anchor=(0.0, 1.25))
    fs.save(fig, fname)


def fig_cdf(res, scen='S3'):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for c in CTRL:
        ds = np.concatenate([np.array(r['delays']) for r in res if S.tag_of(r) == c and r['scenario'] == scen])
        xs = np.sort(ds)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=fs.CTRL_COLOR[c], lw=1.8, label=fs.CTRL_LABEL[c])
    ax.set_xlim(0, 700)
    ax.set_xlabel('Waiting time per vehicle (s)')
    ax.set_ylabel('Cumulative share of vehicles')
    ax.legend(fontsize=7.2, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fs.save(fig, 'figs/fig12_cdf_s3.png')


def fig_ablation(abl_rows):
    """abl_rows: list of (label, mean change % vs GATC, lo, hi)"""
    rows = sorted(abl_rows, key=lambda z: z[1])
    fig, ax = plt.subplots(figsize=(6.2, 0.30 * len(rows) + 0.8))
    y = np.arange(len(rows))
    cols = [PAL[2] if r[1] <= 0 else PAL[1] for r in rows]
    ax.barh(y, [r[1] for r in rows], color=cols, height=0.62, zorder=2)
    ax.errorbar([r[1] for r in rows], y, xerr=np.array([[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]]),
                fmt='none', ecolor=fs.INK2, elinewidth=0.8, capsize=1.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0, color=fs.INK2, lw=0.9)
    ax.set_xlabel('Change in average waiting time vs the full mode (%), mean of S1-S5')
    ax.grid(axis='y', visible=False)
    fs.save(fig, 'figs/fig13_ablation.png')


def fig_sweep(sweep, tag='GATC'):
    rms = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    fps = [0.0, 1.0, 2.0, 3.0]
    M = np.zeros((len(rms), len(fps)))
    for i, rm in enumerate(rms):
        for j, fp in enumerate(fps):
            v = [r['delay'] for r in sweep if S.tag_of(r) == tag and abs(r['recall_mult'] - rm) < 1e-9
                 and abs(r['extra_fp'] - fp) < 1e-9]
            M[i, j] = np.mean(v)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list('seq', ['#fdf0e6', '#f7b58c', '#eb6834', '#b5431a', '#6e2408'])
    im = ax.imshow(M, cmap=cmap, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            col = 'white' if v > M.min() + 0.6 * (M.max() - M.min()) else fs.INK
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color=col)
    ax.set_xticks(range(len(fps)))
    ax.set_xticklabels([f'+{f:g}' for f in fps])
    ax.set_yticks(range(len(rms)))
    ax.set_yticklabels([f'x{r:.1f}' for r in rms])
    ax.set_xlabel('Additional false positives per frame')
    ax.set_ylabel('Recall multiplier on profile P1')
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label('Average waiting time (s/veh)', fontsize=8)
    cb.outline.set_visible(False)
    fs.save(fig, f'figs/fig14_sweep_{tag}.png')
    return M


def fig_attention(A, fname='figs/fig15_attention.png'):
    names = []
    for j in range(3):
        for d in ['W', 'N', 'E', 'S']:
            names.append(f'J{j + 1}-{d}')
    names += ['J1', 'J2', 'J3']
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list('vio', ['#ffffff', '#d9d4f2', '#9085e9', '#4a3aa7', '#231a66'])
    Am = np.ma.masked_where(A <= 0, A)
    im = ax.imshow(Am, cmap=cmap, vmin=0, vmax=max(0.3, float(A.max())))
    ax.set_xticks(range(15))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(15))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Attended node q')
    ax.set_ylabel('Receiving node p')
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k in (3.5, 7.5, 11.5):
        ax.axhline(k, color=fs.AXIS, lw=0.6)
        ax.axvline(k, color=fs.AXIS, lw=0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(r'Mean attention coefficient $\alpha_{pq}$ (layer 2)', fontsize=8)
    cb.outline.set_visible(False)
    fs.save(fig, fname)


def attention_matrix(scen='S4', seed=1):
    import simcore as sc
    import controllers as ct
    import train
    import json as _j
    cfg = _j.load(open('results/final_configs.json'))['GATC-S']
    model = train.load_model('gat_full')
    prof = sc.make_profiles()['P1']
    acc = []

    class Rec(ct.GATCoordinated):
        def plan_junction(self, sim, j, t):
            out = super().plan_junction(sim, j, t)
            acc.append(self.model.l2.last_alpha[0].mean(axis=0))
            return out
    c = Rec(prof, model, kappa=cfg['kappa'], rho0=cfg['rho0'], max_factor=cfg['max_factor'], c_max=cfg['c_max'],
            tactical=False, clear_cycle=False)
    sim = sc.Sim(scen, seed, c, perception=prof)
    sim.run()
    return np.mean(acc, axis=0)


def fig_dqn_learning(cfg, fname='figs/fig11_dqn_learning.png'):
    """Greedy DQN checkpoints evaluated on tuning seeds (mean of S1-S5) against the tuned reference controllers."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    lab = {'DQN-C': 'DQN-C, selected (camera frames)', 'DQN-O': 'DQN-O (exact state)',
           'DQN-C matched': 'Camera frames, DQN-O settings'}
    for name, pts in cfg['series'].items():
        col = fs.CTRL_COLOR['DQN-C'] if name.startswith('DQN-C') else fs.CTRL_COLOR[name]
        ls = '--' if name == 'DQN-C matched' else '-'
        mk = 's' if name == 'DQN-C matched' else 'o'
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=3.4, color=col, lw=1.7, ls=ls,
                label=lab[name], zorder=3)
        sel = cfg['selected'].get(name)
        if sel is not None:
            v = dict((int(a), b) for a, b in pts)[sel]
            ax.scatter([sel], [v], s=70, facecolors='none', edgecolors=fs.INK, linewidths=1.2, zorder=4)
    ls = {'GATC': '-', 'ACT': '--', 'MP-A': ':', 'FT-W': '-.'}
    for name, val in cfg['refs'].items():
        ax.axhline(val, color=fs.CTRL_COLOR[name], lw=1.3, ls=ls.get(name, '--'), label=f'{name} (tuned)', zorder=2)
    ax.set_yscale('log')
    ax.set_yticks([50, 100, 200, 400, 800])
    ax.set_yticklabels(['50', '100', '200', '400', '800'])
    ax.set_ylim(60, 1500)
    ax.set_xlabel('Training episode (open circles: frozen checkpoints)', labelpad=2)
    ax.set_ylabel('Waiting time, tuning seeds (s/veh)')
    ax.legend(fontsize=7.0, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    fs.save(fig, fname)


if __name__ == '__main__':
    import sys
    main = load_main()
    fig_training()
    fig_scatter()
    fig_bars(main)
    M = fig_heat_change(main)
    series = S.load('results/series.jsonl')
    if os.path.exists('results/dqn_series.jsonl'):
        series += S.load('results/dqn_series.jsonl')
    fig_series(series, 'S5', 'figs/fig11_series_s5.png')
    if os.path.exists('results/dqn_learning.json'):
        fig_dqn_learning(json.load(open('results/dqn_learning.json')))
    fig_cdf(main, 'S3')
    fig_ablation(json.load(open('results/ablation_fig.json')))
    sweep = S.load('results/sweep.jsonl')
    Ms = fig_sweep(sweep, 'GATC-S')
    np.save('results/sweep_gatcs.npy', Ms)
    Ml = fig_sweep(sweep, 'LPS')
    np.save('results/sweep_lps.npy', Ml)
    Mg = fig_sweep(sweep, 'GATC')
    np.save('results/sweep_gatc.npy', Mg)
    A = attention_matrix('S4', 1)
    np.save('results/attention_s4.npy', A)
    fig_attention(A)
    print('done')
