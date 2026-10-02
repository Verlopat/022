"""Train the node estimators (GAT, GAT-local, GCN, MLP) on the behaviour-policy data."""
import json
import os
import pickle
import sys
import time

import numpy as np

import controllers as ct
import nn
import simcore as sc

LAM_PSI = 2.0


def load_data(path='data/train_episodes.pkl', val_frac=0.2, seed=0):
    eps = pickle.load(open(path, 'rb'))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eps))
    nval = int(round(val_frac * len(eps)))
    val_ids = set(idx[:nval].tolist())
    tr = [e for i, e in enumerate(eps) if i not in val_ids]
    va = [e for i, e in enumerate(eps) if i in val_ids]

    def stack(es):
        Y2 = np.stack([np.concatenate([e['Q'] for e in es]), np.concatenate([e['A'] for e in es])], axis=-1)
        return (np.concatenate([e['X'] for e in es]), Y2, np.concatenate([e['psi'] for e in es]))
    return stack(tr), stack(va), (tr, va)


def make_model(kind, variant, seed=0, hidden=16, heads=4):
    net = sc.Network()
    mask = ct.build_mask(net, variant)
    return nn.NodeEstimator(kind, ct.NODE_F, mask, ct.LANE_IDX, ct.JUNC_IDX, hidden=hidden, heads=heads, seed=seed)


def loss_fn(model, X, Y, P):
    y, p = model.forward(X)
    ly = nn.huber(y, Y / ct.Y_SCALE, delta=0.5)
    lp = nn.mse(p, P)
    return ly + lp * LAM_PSI, ly, lp


def evaluate(model, X, Y, P, bs=512):
    ys, ps = [], []
    for i in range(0, len(X), bs):
        y, p = model.forward(X[i:i + bs])
        ys.append(y.data)
        ps.append(p.data)
    yh = np.concatenate(ys) * ct.Y_SCALE
    ph = np.concatenate(ps)
    return yh, ph


def metrics(yh2, Y2, ph, P, internal=(2, 4, 6, 8)):
    res = {}
    if yh2.ndim == 3:
        res['mae_q'] = float(np.abs(yh2[..., 0] - Y2[..., 0]).mean())
        res['mae_a'] = float(np.abs(yh2[..., 1] - Y2[..., 1]).mean())
        res['r2_q'] = float(1 - ((yh2[..., 0] - Y2[..., 0]) ** 2).sum() / ((Y2[..., 0] - Y2[..., 0].mean()) ** 2).sum())
        res['r2_a'] = float(1 - ((yh2[..., 1] - Y2[..., 1]) ** 2).sum() / ((Y2[..., 1] - Y2[..., 1].mean()) ** 2).sum())
        yh = yh2.sum(-1)
        Y = Y2.sum(-1)
    else:
        yh, Y = yh2, Y2
    res['mae_y'] = float(np.abs(yh - Y).mean())
    res['rmse_y'] = float(np.sqrt(((yh - Y) ** 2).mean()))
    res['r2_y'] = float(1 - ((yh - Y) ** 2).sum() / ((Y - Y.mean()) ** 2).sum())
    ii = list(internal)
    bb = [a for a in range(12) if a not in ii]
    res['mae_y_internal'] = float(np.abs(yh[:, ii] - Y[:, ii]).mean())
    res['mae_y_boundary'] = float(np.abs(yh[:, bb] - Y[:, bb]).mean())
    res['r2_y_internal'] = float(1 - ((yh[:, ii] - Y[:, ii]) ** 2).sum() / ((Y[:, ii] - Y[:, ii].mean()) ** 2).sum())
    res['mae_psi'] = float(np.abs(ph - P).mean())
    res['r2_psi'] = float(1 - ((ph - P) ** 2).sum() / ((P - P.mean()) ** 2).sum())
    return res


def train(kind, variant, epochs=100, bs=128, lr=3e-3, seed=0, patience=20, tag=None, verbose=True):
    (Xtr, Ytr, Ptr), (Xva, Yva, Pva), _ = load_data()
    model = make_model(kind, variant, seed=seed)
    opt = nn.Adam(model.params(), lr=lr, wd=1e-4)
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    hist = []
    best, best_state, bad = 1e9, None, 0
    t0 = time.time()
    for ep in range(epochs):
        lr_t = 3e-4 + 0.5 * (lr - 3e-4) * (1 + np.cos(np.pi * ep / epochs))
        perm = rng.permutation(n)
        tl, tn = 0.0, 0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            L, ly, lp = loss_fn(model, Xtr[b], Ytr[b], Ptr[b])
            opt.zero_grad()
            L.backward()
            opt.step(lr_t)
            tl += float(L.data) * len(b)
            tn += len(b)
        # validation loss (mean Huber over both lane outputs + weighted psi error)
        yh, ph = evaluate(model, Xva, Yva, Pva)
        r = yh / ct.Y_SCALE - Yva / ct.Y_SCALE
        a = np.abs(r)
        vy = np.where(a <= 0.5, 0.5 * r ** 2, 0.5 * (a - 0.25)).mean()
        vp = ((ph - Pva) ** 2).mean()
        vl = float(vy + LAM_PSI * vp)
        hist.append(dict(epoch=ep + 1, train_loss=tl / tn, val_loss=vl, lr=float(lr_t)))
        if vl < best - 1e-6:
            best, best_state, bad = vl, model.get_state(), 0
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f'{kind}-{variant} ep {ep + 1:3d} train {tl / tn:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if bad >= patience:
            if verbose:
                print('early stop at', ep + 1)
            break
    model.set_state(best_state)
    yh, ph = evaluate(model, Xva, Yva, Pva)
    m = metrics(yh, Yva, ph, Pva)
    m['train_time_s'] = time.time() - t0
    m['epochs_run'] = len(hist)
    m['best_val_loss'] = best
    m['n_params'] = int(sum(p.data.size for p in model.params()))
    tag = tag or f'{kind}_{variant}'
    os.makedirs('models', exist_ok=True)
    with open(f'models/{tag}.pkl', 'wb') as f:
        pickle.dump(dict(kind=kind, variant=variant, state=model.get_state()), f)
    with open(f'models/{tag}_hist.json', 'w') as f:
        json.dump(dict(hist=hist, metrics=m), f, indent=1)
    print(tag, json.dumps(m))
    return model, m


def load_model(tag):
    d = pickle.load(open(f'models/{tag}.pkl', 'rb'))
    m = make_model(d['kind'], d['variant'])
    m.set_state(d['state'])
    return m


def snapshot_baseline():
    """Calibrated camera-only estimator: y_hat = b * score (least squares on training)."""
    (Xtr, Ytr, _), (Xva, Yva, Pva), _ = load_data()
    s_tr = Xtr[:, :12, 0] * 15.0
    s_va = Xva[:, :12, 0] * 15.0
    Ytr = Ytr.sum(-1)
    Yva = Yva.sum(-1)
    b = float((s_tr * Ytr).sum() / (s_tr ** 2).sum())
    yh = b * s_va
    # junction psi from the camera: flow ratio proxy = sum of scores / H (clipped)
    ph = np.minimum(1.0, np.stack([s_va[:, 4 * j:4 * j + 4].sum(1) for j in range(3)], 1) / ct.H_TARGET)
    m = metrics(yh, Yva, ph, Pva)
    m['scale_b'] = b
    raw = metrics(s_va, Yva, ph, Pva)
    return m, raw


if __name__ == '__main__':
    which = sys.argv[1:] or ['gat:full']
    for w in which:
        kind, variant = w.split(':')
        train(kind, variant)
