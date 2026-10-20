"""Fast vectorised forward model and the generator's STRUCTURAL acceptance rules for queue-network-diagnosis.

The rules are evaluated on noise-free readings of the drawn configuration and never call the reference solver: every
error must change the readings by at least REMOVE_MIN (sum of squared normalised differences), every replacement of an
error by any other single error must leave at least ALT_MIN, and every slowdown factor and demand factor must have a
Fisher standard deviation of at most FISHER_SD_MAX and DEMAND_SD_MAX. Every rule cost is minimised over the continuous
parameters of the alternative (its slowdown factors and the six demand factors), so a margin is never overstated by
holding them at the true values."""
import math

import numpy as np

HUBER = 3.0
F_LO, F_HI = 0.49, 0.81
U_LO, U_HI = 0.79, 1.26
G_LO, G_HI = 0.84, 1.16


class Model:
    def __init__(self, spec, z):
        self.spec = spec
        st = spec["stations"]
        self.n = len(st)
        self.mu0 = np.array([s["service_rate"] for s in st], float)
        self.c0 = np.array([s["servers"] for s in st], int)
        self.K0 = np.array([s["capacity"] for s in st], int)
        self.ext = np.array([s["external_rate"] for s in st], float)
        self.links = spec["links"]
        self.lfrom = np.array([l["from"] for l in self.links], int)
        self.lto = np.array([l["to"] for l in self.links], int)
        self.lp = np.array([l["prob"] for l in self.links], float)
        self.scale = np.array(spec["arrival_scale"], float)
        self.T = len(self.scale)
        m = spec["meters"]
        self.sig = np.array([x["sigma"] for x in m], float)
        self.z = np.asarray(z, float)
        self.ty = [x["type"] for x in m]
        self.mk = {t: np.array([i for i, x in enumerate(m) if x["type"] == t], int) for t in ("occupancy", "offered", "lost", "dispatch")}
        self.ms = {t: np.array([m[i]["station"] for i in self.mk[t]], int) for t in ("occupancy", "offered", "lost")}
        self.ml = np.array([m[i]["link"] for i in self.mk["dispatch"]], int)
        self.mask = np.ones_like(self.z, bool)
        self.warm = None

    def predict(self, cfg):
        n, T = self.n, self.T
        mu = self.mu0 * float(cfg.get("service", {}).get(0, 1.0)); c = self.c0.copy(); K = self.K0.copy()
        for s, d in cfg["servers_down"].items():
            c[s] -= d
        for s, f in cfg["slowdown"].items():
            mu[s] *= f
        for s, k in cfg["buffer_cut"].items():
            K[s] -= k
        to = self.lto.copy()
        for l, d in cfg["misroutes"].items():
            to[l] = d
        P = np.zeros((n, n))
        np.add.at(P, (self.lfrom, to), self.lp)
        Kmax = int(K.max())
        k = np.arange(1, Kmax + 1)
        invrate = np.where(k[None, :] <= K[:, None], 1.0 / (np.minimum(k[None, :], c[:, None]) * mu[:, None]), 0.0)   # zero beyond the capacity
        u = np.array([float(cfg.get("demand", {}).get(t, 1.0)) for t in range(T)])
        ext = (self.scale * u)[:, None] * self.ext[None, :]
        idx = np.arange(n)
        nvec = np.arange(Kmax + 1)
        lam = self.warm if self.warm is not None and self.warm.shape == ext.shape else ext.copy()
        ones = np.ones((T, n, 1))
        for it in range(4000):
            p = np.concatenate([ones, np.cumprod(lam[:, :, None] * invrate[None], axis=2)], axis=2)
            p /= p.sum(axis=2, keepdims=True)
            theta = lam * (1 - p[:, idx, K])
            new = ext + theta @ P
            done = np.max(np.abs(new - lam)) <= 1e-11 * (1 + new.max())
            lam = new
            if done:
                break
        self.warm = lam
        p = np.concatenate([ones, np.cumprod(lam[:, :, None] * invrate[None], axis=2)], axis=2)
        p /= p.sum(axis=2, keepdims=True)
        L = p @ nvec
        B = p[:, idx, K]
        theta = lam * (1 - B)
        out = np.zeros((T, len(self.sig)))
        out[:, self.mk["occupancy"]] = L[:, self.ms["occupancy"]]
        out[:, self.mk["offered"]] = lam[:, self.ms["offered"]]
        out[:, self.mk["lost"]] = (lam * B)[:, self.ms["lost"]]
        out[:, self.mk["dispatch"]] = theta[:, self.lfrom[self.ml]] * self.lp[self.ml]
        return out

    def resid(self, cfg):
        return (self.z - self.predict(cfg)) / self.sig

    def cost(self, cfg, robust=True):
        r = self.resid(cfg)[self.mask]
        if not robust:
            return float(r @ r)
        a = np.abs(r)
        return float(np.sum(np.where(a <= HUBER, a * a, 2 * HUBER * a - HUBER * HUBER)))


def empty():
    return {"servers_down": {}, "slowdown": {}, "buffer_cut": {}, "misroutes": {}, "demand": {}, "service": {}}


def with_(cfg, kind, key, val):
    new = {k: dict(v) for k, v in cfg.items()}
    new[kind][key] = val
    return new


def without(cfg, kind, key):
    new = {k: dict(v) for k, v in cfg.items()}
    del new[kind][key]
    return new


def used_stations(model, cfg):
    s = set(cfg["servers_down"]) | set(cfg["slowdown"]) | set(cfg["buffer_cut"])
    s |= {int(model.lfrom[l]) for l in cfg["misroutes"]}
    return s


def fit_slow_one(model, cfg, s, robust=True):
    """golden-section on the slowdown factor of station s (others fixed), followed by a coarse bracket guard"""
    grid = np.linspace(F_LO, F_HI, 7)
    vals = [model.cost(with_(cfg, "slowdown", s, f), robust) for f in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = model.cost(with_(cfg, "slowdown", s, x1), robust), model.cost(with_(cfg, "slowdown", s, x2), robust)
    for _ in range(13):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a); f1 = model.cost(with_(cfg, "slowdown", s, x1), robust)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a); f2 = model.cost(with_(cfg, "slowdown", s, x2), robust)
    f = (a + b) / 2
    return f, model.cost(with_(cfg, "slowdown", s, f), robust)


def continuous_keys(cfg):
    return ([("slowdown", k) for k in sorted(cfg["slowdown"])] + [("demand", t) for t in sorted(cfg.get("demand", {}))]
            + [("service", 0) for _ in cfg.get("service", {})])


def refit_continuous(model, cfg, robust=True, sweeps=3):
    """joint fit of every slowdown factor and every demand factor by damped Gauss-Newton with finite-difference Jacobians
    (IRLS weights for the Huber loss); bounds F_LO..F_HI and U_LO..U_HI"""
    cfg = {k: dict(v) for k, v in cfg.items()}
    keys = continuous_keys(cfg)
    if not keys:
        return cfg, model.cost(cfg, robust)
    bounds = {"slowdown": (F_LO, F_HI), "demand": (U_LO, U_HI), "service": (G_LO, G_HI)}
    lo = np.array([bounds[k[0]][0] for k in keys])
    hi = np.array([bounds[k[0]][1] for k in keys])
    x = np.array([cfg[k][i] for k, i in keys], float)

    def put(xv):
        c = {k: dict(v) for k, v in cfg.items()}
        for (k, i), v in zip(keys, xv):
            c[k][i] = float(v)
        return c

    def rho(r):
        if not robust:
            return float(r @ r)
        a = np.abs(r)
        return float(np.sum(np.where(a <= HUBER, a * a, 2 * HUBER * a - HUBER * HUBER)))

    r0 = model.resid(put(x))[model.mask]
    c0 = rho(r0)
    damp, h = 1e-3, 2e-3
    for _ in range(3 + sweeps):
        cols = []
        for i in range(len(keys)):
            xh = x.copy(); sgn = -1.0 if x[i] + h > hi[i] else 1.0
            xh[i] += sgn * h
            cols.append((model.resid(put(xh))[model.mask] - r0) / (sgn * h))
        J = np.array(cols).T
        w = np.ones_like(r0) if not robust else np.minimum(1.0, HUBER / np.maximum(np.abs(r0), 1e-12))
        A = J.T @ (J * w[:, None]); g = J.T @ (w * r0)
        moved = False
        for _try in range(4):
            step = np.linalg.solve(A + damp * np.diag(np.diag(A) + 1e-9), -g)
            xn = np.clip(x + step, lo, hi)
            rn = model.resid(put(xn))[model.mask]; cn = rho(rn)
            if cn < c0:
                moved = np.max(np.abs(xn - x)) > 2e-4
                x, r0, c0 = xn, rn, cn
                damp = max(damp / 3, 1e-6)
                break
            damp *= 8
        else:
            break
        if not moved:
            break
    return put(x), c0


def refit_slowdowns(model, cfg, robust=True, sweeps=3):
    return refit_continuous(model, cfg, robust, sweeps)


def candidates(model, cfg):
    used = used_stations(model, cfg)
    for s in range(model.n):
        if s in used:
            continue
        c, K = model.c0[s], model.K0[s]
        for d in range(1, min(2, c - 1) + 1):
            yield ("servers_down", s, d)
        for k in range(1, K - c):
            yield ("buffer_cut", s, k)
        yield ("slowdown", s, None)
        taken = {int(x) for k, d in cfg["misroutes"].items() for x in (model.lfrom[k], model.lto[k], d)}
        if s in taken:
            continue
        for l in np.nonzero(model.lfrom == s)[0]:
            if int(model.lto[l]) in taken:
                continue
            outs = set(model.lto[model.lfrom == s].tolist())
            for d in range(model.n):
                if d != s and d not in outs and d not in taken:
                    yield ("misroutes", int(l), d)


def _evaluate(model, cfg, cand, robust=True):
    kind, key, val = cand
    if kind == "slowdown":
        f, c = fit_slow_one(model, cfg, key, robust)
        new = with_(cfg, kind, key, f)
    else:
        new = with_(cfg, kind, key, val)
        c = model.cost(new, robust)
    new, c = refit_continuous(model, new, robust, sweeps=1)
    return new, c



REMOVE_MIN = 200.0
ALT_MIN = 100.0
FISHER_SD_MAX = 0.008
DEMAND_SD_MAX = 0.01
SERVICE_SD_MAX = 0.002


def _int_cfg(cfg):
    return {"servers_down": dict(cfg["servers_down"]), "slowdown": dict(cfg["slowdown"]),
            "buffer_cut": dict(cfg["buffer_cut"]), "misroutes": dict(cfg["misroutes"]), "demand": dict(cfg.get("demand", {})),
            "service": dict(cfg.get("service", {}))}


def _jacobian(m, cfg, extra=None):
    """columns: d resid / d parameter for every continuous parameter of cfg (slowdown and demand factors), plus one column for
    the slowdown factor of station `extra` when given; central differences"""
    keys = continuous_keys(cfg) + ([("slowdown", extra)] if extra is not None else [])
    cols = []
    for k, i in keys:
        h = 1e-4
        up = {kk: dict(v) for kk, v in cfg.items()}; dn = {kk: dict(v) for kk, v in cfg.items()}
        up[k][i] = cfg[k][i] + h; dn[k][i] = cfg[k][i] - h
        cols.append(((m.predict(up) - m.predict(dn)) / (2 * h) / m.sig).ravel())
    return np.array(cols).T if cols else np.zeros((m.z.size, 0))


def _projected(r, J):
    """least-squares cost after the best linear adjustment of the continuous parameters (the Jacobian J taken at the base)"""
    if J.shape[1] == 0:
        return float(r @ r)
    q, _ = np.linalg.qr(J)
    t = q.T @ r
    return float(r @ r - t @ t)


def accept(spec, cfg, y):
    """returns (ok, reason). y = noise-free readings of the truth. Every margin is minimised over the continuous parameters
    (slowdown, demand and service factors): cheaply by a linear projection with the base Jacobian, and by the full non-linear refit
    whenever the projected margin is below 2 * ALT_MIN."""
    m = Model(spec, y)
    tr = _int_cfg(cfg)
    for kind, key in list(errors_of(tr)):
        base = without(tr, kind, key)                    # the other errors at their true values, continuous parameters refitted
        _, c_rm = refit_continuous(m, base, robust=False, sweeps=2)
        if c_rm < REMOVE_MIN:
            return False, (kind, key, f"weak {c_rm:.0f}")
        Jb = _jacobian(m, base)
        # every single replacement of this error by another error anywhere (stations not used by the other errors)
        for cand in candidates(m, base):
            k2, key2, val2 = cand
            if (k2, key2) == (kind, key) and (kind == "slowdown" or val2 == tr[kind][key]):
                continue
            if k2 == "slowdown":
                grid = (0.5, 0.575, 0.65, 0.725, 0.8)
                costs = [m.cost(with_(base, k2, key2, f), False) for f in grid]
                if min(costs) > 10 * ALT_MIN:
                    continue
                f0 = grid[int(np.argmin(costs))]
                alt = with_(base, k2, key2, f0)
                r = m.resid(alt).ravel()
                h = 1e-4
                col = ((m.predict(with_(base, k2, key2, f0 + h)) - m.predict(with_(base, k2, key2, f0 - h))) / (2 * h) / m.sig).ravel()
                c = _projected(r, np.column_stack([Jb, col]))
            else:
                alt = with_(base, k2, key2, val2)
                c = m.cost(alt, False)
                if c > 10 * ALT_MIN:
                    continue
                c = _projected(m.resid(alt).ravel(), Jb)
            if c < 2 * ALT_MIN:                           # confirm with the full non-linear refit
                _, c = refit_continuous(m, alt, robust=False, sweeps=2)
            if c < ALT_MIN:
                return False, (kind, key, f"alt {cand} {c:.0f}")
    # compound rule: exchanging the values of two errors of the same kind (two misroute destinations, two cuts, two outages,
    # two slowdown factors) must also leave at least ALT_MIN
    for kind in ("servers_down", "slowdown", "buffer_cut", "misroutes"):
        keys = sorted(tr[kind])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if tr[kind][a] == tr[kind][b]:
                    continue
                alt = {k: dict(v) for k, v in tr.items()}
                alt[kind][a], alt[kind][b] = tr[kind][b], tr[kind][a]
                if kind == "servers_down" and any(alt[kind][x] > min(2, m.c0[x] - 1) for x in (a, b)):
                    continue
                if kind == "buffer_cut" and any(alt[kind][x] > m.K0[x] - m.c0[x] - 1 for x in (a, b)):
                    continue
                if kind == "misroutes":
                    touched = [int(v) for x, d in alt[kind].items() for v in (m.lfrom[x], m.lto[x], d)]
                    if len(touched) != len(set(touched)) or any(alt[kind][x] in set(m.lto[m.lfrom == m.lfrom[x]].tolist()) | {int(m.lfrom[x])} for x in (a, b)):
                        continue
                c = m.cost(alt, False)
                if c < 10 * ALT_MIN:
                    _, c = refit_continuous(m, alt, robust=False, sweeps=2)
                if c < ALT_MIN:
                    return False, (kind, b, f"swap with {a}: {c:.0f}")
    # Fisher precision of every slowdown factor and every demand factor, jointly (the inverse of the full Fisher matrix of
    # the continuous parameters, so a factor that trades off against another is counted as imprecise)
    keys = continuous_keys(tr)
    J = _jacobian(m, tr)
    cov = np.linalg.inv(J.T @ J)
    for (k, i), var in zip(keys, np.diag(cov)):
        sd = math.sqrt(float(var))
        if k == "slowdown" and sd > FISHER_SD_MAX:
            return False, ("slowdown", i, f"imprecise sd {sd:.4f}")
        if k == "demand" and sd > DEMAND_SD_MAX:
            return False, ("demand", i, f"imprecise demand sd {sd:.4f}")
        if k == "service" and sd > SERVICE_SD_MAX:
            return False, ("service", i, f"imprecise service sd {sd:.4f}")
    return True, None



def errors_of(cfg):
    for kind in ("servers_down", "slowdown", "buffer_cut", "misroutes"):
        for key in list(cfg[kind]):
            yield kind, key
