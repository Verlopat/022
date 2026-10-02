"""
Minimal reverse-mode automatic differentiation on NumPy arrays, plus dense
multi-head Graph Attention (GAT), GCN and MLP node models used as the
next-cycle demand / junction-load estimators.
"""
import numpy as np


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    __slots__ = ('data', 'grad', 'parents', 'backward_fn', 'requires_grad')

    def __init__(self, data, parents=(), backward_fn=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.parents = parents
        self.backward_fn = backward_fn
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    @property
    def shape(self):
        return self.data.shape

    # ------------------------------------------------------------------
    def backward(self, grad=None):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v.parents:
                build(p)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data) if grad is None else grad
        for v in reversed(topo):
            if v.backward_fn is not None and v.grad is not None:
                v.backward_fn(v.grad)

    def _acc(self, g):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad = self.grad + g

    # ------------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def bw(g):
            self._acc(_unbroadcast(g, self.shape))
            other._acc(_unbroadcast(g, other.shape))
        out.backward_fn = bw
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, (self,))
        out.backward_fn = lambda g: self._acc(-g)
        return out

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Tensor) else Tensor(other)))

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def bw(g):
            self._acc(_unbroadcast(g * other.data, self.shape))
            other._acc(_unbroadcast(g * self.data, other.shape))
        out.backward_fn = bw
        return out

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other))

        def bw(g):
            if self.requires_grad:
                ga = np.matmul(g, np.swapaxes(other.data, -1, -2))
                self._acc(_unbroadcast(ga, self.shape))
            if other.requires_grad:
                a = self.data
                if a.ndim == 1:
                    a = a[None, :]
                gb = np.matmul(np.swapaxes(a, -1, -2), g)
                other._acc(_unbroadcast(gb, other.shape))
        out.backward_fn = bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).copy())
        out.backward_fn = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        old = self.shape
        out = Tensor(self.data.reshape(*shape), (self,))
        out.backward_fn = lambda g: self._acc(g.reshape(old))
        return out

    def transpose(self, *axes):
        inv = np.argsort(axes)
        out = Tensor(self.data.transpose(*axes), (self,))
        out.backward_fn = lambda g: self._acc(g.transpose(*inv))
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,))

        def bw(g):
            full = np.zeros_like(self.data)
            np.add.at(full, idx, g)
            self._acc(full)
        out.backward_fn = bw
        return out


# ----------------------------------------------------------------------------
# functional ops
# ----------------------------------------------------------------------------
def concat(ts, axis=-1):
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), tuple(ts))
    sizes = [t.shape[axis] for t in ts]
    splits = np.cumsum(sizes)[:-1]

    def bw(g):
        parts = np.split(g, splits, axis=axis)
        for t, p in zip(ts, parts):
            t._acc(p)
    out.backward_fn = bw
    return out


def leaky_relu(x, slope=0.2):
    d = x.data
    out = Tensor(np.where(d > 0, d, slope * d), (x,))
    out.backward_fn = lambda g: x._acc(g * np.where(d > 0, 1.0, slope))
    return out


def elu(x):
    d = x.data
    e = np.exp(np.minimum(d, 0.0)) - 1.0
    out = Tensor(np.where(d > 0, d, e), (x,))
    out.backward_fn = lambda g: x._acc(g * np.where(d > 0, 1.0, e + 1.0))
    return out


def relu(x):
    d = x.data
    out = Tensor(np.maximum(d, 0.0), (x,))
    out.backward_fn = lambda g: x._acc(g * (d > 0))
    return out


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x.data, -40, 40)))
    out = Tensor(s, (x,))
    out.backward_fn = lambda g: x._acc(g * s * (1.0 - s))
    return out


def softplus(x):
    d = x.data
    val = np.logaddexp(0.0, d)
    s = 1.0 / (1.0 + np.exp(-np.clip(d, -40, 40)))
    out = Tensor(val, (x,))
    out.backward_fn = lambda g: x._acc(g * s)
    return out


def masked_softmax(e, mask, axis=-1):
    """softmax over axis with a boolean mask (True = allowed)."""
    d = np.where(mask, e.data, -1e30)
    d = d - d.max(axis=axis, keepdims=True)
    ex = np.exp(d) * mask
    s = ex / (ex.sum(axis=axis, keepdims=True) + 1e-30)
    out = Tensor(s, (e,))

    def bw(g):
        gs = s * (g - (g * s).sum(axis=axis, keepdims=True))
        e._acc(gs)
    out.backward_fn = bw
    return out


def huber(pred, target, delta=0.5, weight=None):
    r = pred.data - target
    a = np.abs(r)
    quad = a <= delta
    val = np.where(quad, 0.5 * r ** 2, delta * (a - 0.5 * delta))
    if weight is None:
        weight = np.ones_like(val)
    n = weight.sum()
    out = Tensor((val * weight).sum() / n, (pred,))
    out.backward_fn = lambda g: pred._acc(g * np.where(quad, r, delta * np.sign(r)) * weight / n)
    return out


def mse(pred, target, weight=None):
    r = pred.data - target
    if weight is None:
        weight = np.ones_like(r)
    n = weight.sum()
    out = Tensor((weight * r ** 2).sum() / n, (pred,))
    out.backward_fn = lambda g: pred._acc(g * 2.0 * r * weight / n)
    return out


# ----------------------------------------------------------------------------
# parameters and optimiser
# ----------------------------------------------------------------------------
class Param(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


def glorot(rng, fan_in, fan_out, shape=None):
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return Param(rng.uniform(-lim, lim, size=shape if shape is not None else (fan_in, fan_out)))


class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8, wd=1e-4):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = wd
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= lr * mh / (np.sqrt(vh) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ----------------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------------
class GATLayer:
    """Dense multi-head graph attention layer (Velickovic et al. formulation).

    e_pq = LeakyReLU(a_src . W x_p + a_dst . W x_q), alpha = masked softmax over q,
    h_p = sum_q alpha_pq W x_q ; heads concatenated (or averaged).
    """

    def __init__(self, rng, fin, fout, heads, concat_heads=True):
        self.K, self.F = heads, fout
        self.W = glorot(rng, fin, heads * fout)
        self.a_src = glorot(rng, fout, 1, shape=(1, heads, 1, fout))
        self.a_dst = glorot(rng, fout, 1, shape=(1, heads, 1, fout))
        self.b = Param(np.zeros(heads * fout if concat_heads else fout))
        self.concat_heads = concat_heads
        self.last_alpha = None

    def params(self):
        return [self.W, self.a_src, self.a_dst, self.b]

    def __call__(self, x, mask):
        B, N, _ = x.shape
        z = (x @ self.W).reshape(B, N, self.K, self.F).transpose(0, 2, 1, 3)  # B,K,N,F
        es = (z * self.a_src).sum(axis=-1)  # B,K,N
        ed = (z * self.a_dst).sum(axis=-1)
        e = es.reshape(B, self.K, N, 1) + ed.reshape(B, self.K, 1, N)
        e = leaky_relu(e, 0.2)
        alpha = masked_softmax(e, mask[None, None, :, :], axis=-1)
        self.last_alpha = alpha.data
        h = alpha @ z  # B,K,N,F
        if self.concat_heads:
            h = h.transpose(0, 2, 1, 3).reshape(B, N, self.K * self.F)
        else:
            h = h.mean(axis=1)
        return h + self.b


class GCNLayer:
    def __init__(self, rng, fin, fout):
        self.W = glorot(rng, fin, fout)
        self.b = Param(np.zeros(fout))

    def params(self):
        return [self.W, self.b]

    def __call__(self, x, A_hat):
        return Tensor(A_hat) @ (x @ self.W) + self.b


class Linear:
    def __init__(self, rng, fin, fout):
        self.W = glorot(rng, fin, fout)
        self.b = Param(np.zeros(fout))

    def params(self):
        return [self.W, self.b]

    def __call__(self, x):
        return x @ self.W + self.b


class NodeEstimator:
    """Two message-passing layers + skip connection + two output heads.

    kind: 'gat' | 'gcn' | 'mlp'
    Outputs: lane-node demand (normalised, softplus) and junction impact score (sigmoid).
    """

    def __init__(self, kind, fin, mask, lane_idx, junc_idx, hidden=16, heads=4, seed=0):
        rng = np.random.default_rng(seed)
        self.kind = kind
        self.mask = mask.astype(bool)
        A = mask.astype(float)
        d = A.sum(axis=1)
        self.A_hat = A / np.sqrt(np.outer(d, d))
        self.lane_idx = np.array(lane_idx)
        self.junc_idx = np.array(junc_idx)
        H = hidden * heads
        if kind == 'gat':
            self.l1 = GATLayer(rng, fin, hidden, heads, concat_heads=True)
            self.l2 = GATLayer(rng, H, hidden, heads, concat_heads=True)
        elif kind == 'gcn':
            self.l1 = GCNLayer(rng, fin, H)
            self.l2 = GCNLayer(rng, H, H)
        else:
            self.l1 = Linear(rng, fin, H)
            self.l2 = Linear(rng, H, H)
        self.head_y = Linear(rng, H + fin, 2)
        self.head_psi = Linear(rng, H + fin, 1)

    def params(self):
        return self.l1.params() + self.l2.params() + self.head_y.params() + self.head_psi.params()

    def forward(self, X):
        x = X if isinstance(X, Tensor) else Tensor(X)
        if self.kind == 'gat':
            h = elu(self.l1(x, self.mask))
            h = elu(self.l2(h, self.mask))
        elif self.kind == 'gcn':
            h = elu(self.l1(x, self.A_hat))
            h = elu(self.l2(h, self.A_hat))
        else:
            h = elu(self.l1(x))
            h = elu(self.l2(h))
        hs = concat([h, x], axis=-1)
        y = softplus(self.head_y(hs[:, self.lane_idx, :]))
        psi = sigmoid(self.head_psi(hs[:, self.junc_idx, :]))[..., 0]
        return y, psi

    def get_state(self):
        return [p.data.copy() for p in self.params()]

    def set_state(self, st):
        for p, s in zip(self.params(), st):
            p.data = s.copy()

    def attention(self):
        """mean attention of layer 2 over heads, from the last forward pass"""
        if self.kind != 'gat':
            return None
        return self.l1.last_alpha, self.l2.last_alpha


def gradcheck():
    rng = np.random.default_rng(1)
    N, F = 6, 5
    mask = np.eye(N, dtype=bool)
    mask[0, 1] = mask[1, 0] = mask[2, 3] = mask[3, 2] = mask[4, 5] = mask[1, 4] = True
    for kind in ('gat', 'gcn', 'mlp'):
        m = NodeEstimator(kind, F, mask, [0, 1, 2, 3], [4, 5], hidden=3, heads=2, seed=3)
        X = rng.normal(size=(2, N, F))
        yt = rng.random((2, 4, 2))
        pt = rng.random((2, 2))

        def loss_fn():
            y, p = m.forward(X)
            return huber(y, yt) + mse(p, pt)
        L = loss_fn()
        for p in m.params():
            p.grad = None
        L.backward()
        maxrel = 0.0
        for p in m.params():
            g = p.grad
            flat = p.data.reshape(-1)
            for k in rng.choice(flat.size, size=min(5, flat.size), replace=False):
                old = flat[k]
                flat[k] = old + 1e-6
                lp = loss_fn().data
                flat[k] = old - 1e-6
                lm = loss_fn().data
                flat[k] = old
                num = (lp - lm) / 2e-6
                an = g.reshape(-1)[k]
                rel = abs(num - an) / max(1e-8, abs(num) + abs(an))
                maxrel = max(maxrel, rel)
        print(kind, 'max relative gradient error', maxrel)


if __name__ == '__main__':
    gradcheck()
