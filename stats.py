"""Statistics helpers: paired Wilcoxon tests, Holm correction, rank-biserial effect size, bootstrap CIs."""
import json
from collections import defaultdict

import numpy as np
from scipy import stats as st


def load(path):
    return [json.loads(l) for l in open(path)]


def tag_of(r):
    return r.get('tag', r['ctrl'])


def table(res, metric='delay'):
    d = defaultdict(dict)
    for r in res:
        d[(tag_of(r), r['scenario'])][r['seed']] = r[metric]
    out = {}
    for k, v in d.items():
        seeds = sorted(v)
        out[k] = np.array([v[s] for s in seeds])
    return out


def paired(res, a, b, scen, metric='delay'):
    da = {r['seed']: r[metric] for r in res if tag_of(r) == a and r['scenario'] == scen}
    db = {r['seed']: r[metric] for r in res if tag_of(r) == b and r['scenario'] == scen}
    seeds = sorted(set(da) & set(db))
    return np.array([da[s] for s in seeds]), np.array([db[s] for s in seeds])


def wilcoxon_rb(x, y):
    """two-sided Wilcoxon signed-rank on x - y; matched-pairs rank-biserial r (negative when x < y)."""
    d = x - y
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, 0.0
    ranks = st.rankdata(np.abs(nz))
    wp = ranks[nz > 0].sum()
    wm = ranks[nz < 0].sum()
    r = (wp - wm) / (wp + wm)
    p = st.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', method='auto').pvalue
    return float(p), float(r)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, idx in enumerate(order):
        val = min(1.0, (m - k) * p[idx])
        run = max(run, val)
        adj[idx] = run
    return adj


def boot_ci(x, n=10000, seed=0, stat=np.mean):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    bs = stat(x[idx], axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt_pm(x, nd=1):
    return f'{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}'


def fmt_p(p):
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'
