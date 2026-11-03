#!/usr/bin/env python3
"""Reference solver for queue-network-diagnosis.

Fits the M/M/c/K reduced-load network model to the monitor readings of every period and searches the hidden station errors
(servers down, slowdowns, buffer cuts) and misrouted dispatch channels by greedy forward selection with a robust (Huber) loss,
drop-and-retune refinement, and a final estimator-free gross-reading test (largest normalised residual)."""
import argparse
import itertools
import json
import math
import multiprocessing
import time
from pathlib import Path

import numpy as np

HUBER = 3.0
ADD_GAIN = 30.0
GROSS_FLAG = 6.0
BEAM_WIDTH = 12
BEAM_EXPAND = 6
PER_KIND = 5
MAX_DEPTH = 8
ERROR_PENALTY = 40.0
GROSS_PENALTY = 25.0
FIT_LIMIT = 2.5
RESTARTS = 4
GLOBAL_SECONDS = 620.0
NETWORK_SECONDS = 240.0
MAX_GROSS = 2
MAX_ERRORS = 5
F_LO, F_HI = 0.49, 0.81
U_LO, U_HI = 0.79, 1.26       # demand factor of a period
G_LO, G_HI = 0.84, 1.16       # service factor of the network
CONTINUOUS = {"slowdown": (F_LO, F_HI), "demand": (U_LO, U_HI), "service": (G_LO, G_HI)}
ERROR_KINDS = ("servers_down", "slowdown", "buffer_cut", "misroutes")


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
        self.tabu = set()

    def predict(self, cfg):
        return self.predict_batch([cfg], tol=1e-11)[0]

    def predict_batch(self, cfgs, tol=1e-9):
        """readings of many configurations at once (one fixed-point iteration over the whole batch); shape (B, T, meters)"""
        n, T, B = self.n, self.T, len(cfgs)
        mu = np.tile(self.mu0, (B, 1)); c = np.tile(self.c0, (B, 1)); K = np.tile(self.K0, (B, 1))
        P = np.zeros((B, n, n))
        U = np.ones((B, T))
        for b, cfg in enumerate(cfgs):
            mu[b] *= float(cfg.get("service", {}).get(0, 1.0))
            for t, u in cfg.get("demand", {}).items():
                U[b, t] = u
            for s, d in cfg["servers_down"].items():
                c[b, s] -= d
            for s, f in cfg["slowdown"].items():
                mu[b, s] *= f
            for s, k in cfg["buffer_cut"].items():
                K[b, s] -= k
            to = self.lto.copy()
            for l, d in cfg["misroutes"].items():
                to[l] = d
            np.add.at(P[b], (self.lfrom, to), self.lp)
        Kmax = int(K.max())
        k = np.arange(1, Kmax + 1)
        invrate = np.where(k[None, None, :] <= K[:, :, None], 1.0 / (np.minimum(k[None, None, :], c[:, :, None]) * mu[:, :, None]), 0.0)   # (B, n, Kmax)
        ext = (U * self.scale[None, :])[:, :, None] * self.ext[None, None, :]                  # (B, T, n)
        nvec = np.arange(Kmax + 1)
        idx = np.arange(n)
        lam = np.broadcast_to(self.warm, (B, T, n)).copy() if self.warm is not None and self.warm.shape == (T, n) else ext.copy()
        ones = np.ones((B, T, n, 1))
        Kb = np.broadcast_to(K[:, None, :], (B, T, n))
        def dist(lam):
            p = np.concatenate([ones, np.cumprod(lam[:, :, :, None] * invrate[:, None, :, :], axis=3)], axis=3)
            p /= p.sum(axis=3, keepdims=True)
            return p
        eye = np.eye(n)
        def state(lam):
            p = dist(lam)
            blk = np.take_along_axis(p, Kb[..., None], axis=3)[..., 0]
            Lm = p @ nvec
            theta = lam * (1 - blk)
            return blk, Lm, theta, ext + np.einsum("btn,bnm->btm", theta, P) - lam             # last: -(lam - F(lam))
        blk, Lm, theta, neg = state(lam)
        for it in range(200):
            scale = tol * (1 + lam.max())
            if np.max(np.abs(neg)) <= scale:
                break
            dth = (1 - blk) - blk * (Kb - Lm)                                    # d theta / d lambda of every station
            Jm = eye[None, None] - np.einsum("bnm,btn->btmn", P, dth)             # I - P^T diag(theta')
            try:
                step = np.linalg.solve(Jm, neg[..., None])[..., 0]                # Newton step: J step = -R, R = lam - F(lam) = -neg
            except np.linalg.LinAlgError:
                step = neg
            old = np.max(np.abs(neg))
            t = 1.0
            for _ in range(6):
                lam_try = np.maximum(lam + t * step, 0.0)
                blk2, Lm2, theta2, neg2 = state(lam_try)
                if np.max(np.abs(neg2)) < old or t < 0.05:
                    break
                t *= 0.5
            lam, blk, Lm, theta, neg = lam_try, blk2, Lm2, theta2, neg2
        if B == 1:
            self.warm = lam[0]
        p = dist(lam)
        L = p @ nvec
        Bk = np.take_along_axis(p, Kb[..., None], axis=3)[..., 0]
        theta = lam * (1 - Bk)
        out = np.zeros((B, T, len(self.sig)))
        out[:, :, self.mk["occupancy"]] = L[:, :, self.ms["occupancy"]]
        out[:, :, self.mk["offered"]] = lam[:, :, self.ms["offered"]]
        out[:, :, self.mk["lost"]] = (lam * Bk)[:, :, self.ms["lost"]]
        out[:, :, self.mk["dispatch"]] = theta[:, :, self.lfrom[self.ml]] * self.lp[self.ml]
        return out

    def cost_batch(self, cfgs, robust=True, chunk=96):
        res = []
        for i in range(0, len(cfgs), chunk):
            out = self.predict_batch(cfgs[i:i + chunk])
            r = ((self.z[None] - out) / self.sig[None]) * self.mask[None]
            a = np.abs(r)
            if robust:
                res.append(np.sum(np.where(a <= HUBER, a * a, 2 * HUBER * a - HUBER * HUBER), axis=(1, 2)))
            else:
                res.append(np.sum(a * a, axis=(1, 2)))
        return np.concatenate(res) if res else np.zeros(0)

    def resid(self, cfg):
        return (self.z - self.predict(cfg)) / self.sig

    def cost(self, cfg, robust=True):
        r = self.resid(cfg)[self.mask]
        if not robust:
            return float(r @ r)
        a = np.abs(r)
        return float(np.sum(np.where(a <= HUBER, a * a, 2 * HUBER * a - HUBER * HUBER)))


def empty():
    return {"servers_down": {}, "slowdown": {}, "buffer_cut": {}, "misroutes": {}, "demand": {}, "service": {0: 1.0}}


def continuous_keys(cfg):
    return ([("slowdown", k) for k in sorted(cfg["slowdown"])] + [("demand", t) for t in sorted(cfg.get("demand", {}))]
            + [("service", 0) for _ in cfg.get("service", {})])


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


def _cost_of(model, out, robust):
    r = ((model.z[None] - out) / model.sig[None])[:, model.mask]
    if not robust:
        return np.sum(r * r, axis=1)
    a = np.abs(r)
    return np.sum(np.where(a <= HUBER, a * a, 2 * HUBER * a - HUBER * HUBER), axis=1)


def fit_slow_one(model, cfg, s, robust=True):
    """slowdown factor of station s (others fixed): a 9-point grid over the allowed range, then a 9-point grid around the best point (two batched evaluations)"""
    lo, hi = F_LO, F_HI
    for _ in range(2):
        grid = np.linspace(lo, hi, 9)
        costs = _cost_of(model, model.predict_batch([with_(cfg, "slowdown", s, float(f)) for f in grid]), robust)
        i = int(np.argmin(costs))
        best, step = float(grid[i]), float(grid[1] - grid[0])
        lo, hi = max(F_LO, best - step), min(F_HI, best + step)
    return best, float(costs[i])


def _rho_cost(r, robust):
    if not robust:
        return float(r @ r)
    a = np.abs(r)
    return float(np.sum(np.where(a <= HUBER, a * a, 2 * HUBER * a - HUBER * HUBER)))


def refit_slowdowns(model, cfg, robust=True, sweeps=3):
    """joint refit of every continuous parameter (slowdown factors, the demand factor of every period, the service factor) by
    damped Gauss-Newton with IRLS weights for the Huber loss; the name is historical"""
    cfg = {k: dict(v) for k, v in cfg.items()}
    keys = continuous_keys(cfg)
    if not keys:
        return cfg, model.cost(cfg, robust)
    lo = np.array([CONTINUOUS[k][0] for k, _ in keys]); hi = np.array([CONTINUOUS[k][1] for k, _ in keys])
    x = np.array([cfg[k][i] for k, i in keys], float)
    def put(xv):
        c = {k: dict(v) for k, v in cfg.items()}
        for (k, i), v in zip(keys, xv):
            c[k][i] = float(v)
        return c
    r0 = model.resid(put(x))[model.mask]
    c0 = _rho_cost(r0, robust)
    lam = 1e-3
    h = 2e-3
    for _ in range(2 + sweeps):
        xhs, sgns = [], []
        for i in range(len(keys)):
            xh = x.copy()
            sgn = -1.0 if x[i] + h > hi[i] else 1.0
            xh[i] += sgn * h
            xhs.append(put(xh)); sgns.append(sgn)
        out = model.predict_batch(xhs, tol=1e-11)
        R1 = ((model.z[None] - out) / model.sig[None])[:, model.mask]
        J = ((R1 - r0[None]) / np.array(sgns)[:, None] / h).T
        w = np.ones_like(r0) if not robust else np.minimum(1.0, HUBER / np.maximum(np.abs(r0), 1e-12))
        A = J.T @ (J * w[:, None])
        gvec = J.T @ (w * r0)
        moved = False
        for _try in range(3):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -gvec)
            xn = np.clip(x + step, lo, hi)
            rn = model.resid(put(xn))[model.mask]
            cn = _rho_cost(rn, robust)
            if cn < c0:
                moved = np.max(np.abs(xn - x)) > 3e-4
                x, r0, c0 = xn, rn, cn
                lam = max(lam / 3, 1e-6)
                break
            lam *= 8
        else:
            break
        if not moved:
            break
    return put(x), c0


def candidates(model, cfg):
    for cand in _candidates(model, cfg):
        if not model.tabu or (cand[0], cand[1]) not in model.tabu:
            yield cand


def _candidates(model, cfg):
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


def evaluate(model, cfg, cand, robust=True):
    kind, key, val = cand
    if kind == "slowdown":
        f, c = fit_slow_one(model, cfg, key, robust)
        new = with_(cfg, kind, key, f)
    else:
        new = with_(cfg, kind, key, val)
        c = model.cost(new, robust)
    new, c = refit_slowdowns(model, new, robust, sweeps=1)
    return new, c


def screen(model, cfg, base):
    """cheap screen: discrete candidates directly, slowdowns on a 5-point grid; all configurations are evaluated as one batch"""
    cands, cfgs, owner = [], [], []
    for cand in candidates(model, cfg):
        kind, key, val = cand
        i = len(cands)
        cands.append(cand)
        if kind == "slowdown":
            for f in (0.5, 0.575, 0.65, 0.725, 0.8):
                cfgs.append(with_(cfg, kind, key, f)); owner.append(i)
        else:
            cfgs.append(with_(cfg, kind, key, val)); owner.append(i)
    costs = model.cost_batch(cfgs)
    best = np.full(len(cands), np.inf)
    np.minimum.at(best, np.array(owner, int), costs)
    scored = [(float(best[i]), cands[i]) for i in range(len(cands))]
    scored.sort(key=lambda x: x[0])
    return scored


def greedy(model, cfg, cost):
    while True:
        scored = screen(model, cfg, cost)
        best = None
        for c0, cand in scored[:8]:
            new, c = evaluate(model, cfg, cand)
            if best is None or c < best[1]:
                best = (new, c)
        if best is None or cost - best[1] < ADD_GAIN:
            return cfg, cost
        cfg, cost = best


def key_of(cfg):
    return tuple(sorted((k, int(a), (round(v, 2) if k == "slowdown" else v)) for k in ERROR_KINDS for a, v in cfg[k].items()))


def nerr(cfg):
    return sum(len(cfg[k]) for k in ERROR_KINDS)


def beam(model, width=BEAM_WIDTH, expand=BEAM_EXPAND, max_depth=MAX_DEPTH, per_kind=PER_KIND):
    """beam search over sets of errors; returns the configuration minimising robust cost + ERROR_PENALTY * errors"""
    cfg0 = empty()
    cfg0["demand"] = {t: 1.0 for t in range(model.T)}
    cfg0, c00 = refit_slowdowns(model, cfg0, sweeps=3)
    states = [(c00, cfg0)]
    best = (states[0][0], cfg0)
    for depth in range(max_depth):
        children = {}
        for cost, cfg in states:
            ranked_c = screen(model, cfg, cost)
            picked, used_kind = list(ranked_c[:expand]), {}
            for item in ranked_c[expand:]:
                kd = item[1][0]
                if used_kind.get(kd, 0) < per_kind:
                    used_kind[kd] = used_kind.get(kd, 0) + 1
                    picked.append(item)
            for c0, cand in picked:
                new, c = evaluate(model, cfg, cand)
                k = key_of(new)
                if k not in children or c < children[k][0]:
                    children[k] = (c, new)
        if not children:
            break
        ranked = sorted(children.values(), key=lambda x: x[0])[:width]
        if ranked[0][0] > min(s[0] for s in states) - ADD_GAIN:
            break
        states = ranked
        if ranked[0][0] + ERROR_PENALTY * nerr(ranked[0][1]) < best[0] + ERROR_PENALTY * nerr(best[1]):
            best = ranked[0]
    return best[1], best[0]


def errors_of(cfg):
    for kind in ("servers_down", "slowdown", "buffer_cut", "misroutes"):
        for key in list(cfg[kind]):
            yield kind, key


def misroutes_disjoint(model, cfg):
    """the stated structure: no station is the from, to or actual destination of two misrouted links, and a destination is neither the
    link's own from nor the to of a link from the same station"""
    seen = []
    for k, d in cfg["misroutes"].items():
        if d == model.lfrom[k] or d in set(model.lto[model.lfrom == model.lfrom[k]].tolist()):
            return False
        seen += [int(model.lfrom[k]), int(model.lto[k]), int(d)]
    return len(seen) == len(set(seen))


def swaps(model, cfg, cost):
    """exchange the values of two errors of the same kind (two misroute destinations, two cuts, two outages)"""
    for kind in ("servers_down", "buffer_cut", "misroutes"):
        keys = sorted(cfg[kind])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if cfg[kind][a] == cfg[kind][b]:
                    continue
                new = {k: dict(v) for k, v in cfg.items()}
                new[kind][a], new[kind][b] = cfg[kind][b], cfg[kind][a]
                if kind == "servers_down" and any(new[kind][x] > min(2, model.c0[x] - 1) for x in (a, b)):
                    continue
                if kind == "buffer_cut" and any(new[kind][x] > model.K0[x] - model.c0[x] - 1 for x in (a, b)):
                    continue
                if kind == "misroutes" and not misroutes_disjoint(model, new):
                    continue
                new, c = refit_slowdowns(model, new, sweeps=1)
                if c < cost - 0.1:
                    return new, c, True
    return cfg, cost, False


def prune(model, cfg, cost):
    """drop every error whose removal costs less than ADD_GAIN, one at a time (weakest first)"""
    while True:
        best = None
        for kind, key in list(errors_of(cfg)):
            base = without(cfg, kind, key)
            base, bc = refit_slowdowns(model, base, sweeps=1)
            if bc - cost < ADD_GAIN and (best is None or bc < best[1]):
                best = (base, bc)
        if best is None:
            return cfg, cost
        cfg, cost = best


def permutation_variants(model, cfg, max_perm=5):
    """configurations in which the values of the errors of one kind are permuted among their keys (all permutations of up to
    max_perm errors, all pair exchanges otherwise), restricted to valid ones"""
    out = []
    for kind in ("servers_down", "buffer_cut", "misroutes"):
        keys = sorted(cfg[kind])
        if len(keys) < 2:
            continue
        vals = [cfg[kind][k] for k in keys]
        if len(keys) <= max_perm:
            perms = list(itertools.permutations(range(len(keys))))
        else:
            perms = []
            for x in range(len(keys)):
                for y in range(x + 1, len(keys)):
                    q = list(range(len(keys))); q[x], q[y] = q[y], q[x]
                    perms.append(tuple(q))
        for p in perms:
            new = {k: dict(v) for k, v in cfg.items()}
            for i, k in enumerate(keys):
                new[kind][k] = vals[p[i]]
            if new[kind] == cfg[kind]:
                continue
            if kind == "servers_down" and any(new[kind][x] > min(2, model.c0[x] - 1) for x in keys):
                continue
            if kind == "buffer_cut" and any(new[kind][x] > model.K0[x] - model.c0[x] - 1 for x in keys):
                continue
            if kind == "misroutes":
                if not misroutes_disjoint(model, new):
                    continue
                if any(new[kind][x] in set(model.lto[model.lfrom == model.lfrom[x]].tolist()) | {int(model.lfrom[x])} for x in keys):
                    continue
            out.append(new)
    return out


def kick_repair(model, cfg, cost, top=4):
    """a permuted assignment of values may only pay off once the compensating errors around it are redrawn: take the best few
    permutations by plain cost, prune and greedily re-add errors, and accept a lower penalised objective"""
    variants = permutation_variants(model, cfg)
    if not variants:
        return cfg, cost, False
    raw = model.cost_batch(variants)
    order = np.argsort(raw)[:top]
    target = cost + ERROR_PENALTY * nerr(cfg)
    for i in order:
        new, c = refit_slowdowns(model, variants[i], sweeps=1)
        new, c = prune(model, new, c)
        new, c = greedy(model, new, c)
        new, c = prune(model, new, c)
        if c + ERROR_PENALTY * nerr(new) < target - 1e-6:
            return new, c, True
    return cfg, cost, False


def joint_misroute_repair(model, cfg, cost, budget=3000, top=3, lookahead=60):
    """re-choose all misroutes jointly: a pool of promising (link, destination) options (the best single misroutes of the
    configuration without misroutes, the best replacement of each current misroute, and the current ones) is combined k at a
    time, the combinations are scored in batches and the most promising few are repaired (refit, prune, greedy)"""
    k = len(cfg["misroutes"])
    if k < 2:
        return cfg, cost, False
    size = k
    while math.comb(size + 1, k) <= budget:
        size += 1
    L = len(model.lfrom)
    def options(base):
        busy = used_stations(model, base)
        opts, cfgs = [], []
        for l2 in range(L):
            s2 = int(model.lfrom[l2])
            if s2 in busy:
                continue
            outs = set(model.lto[model.lfrom == s2].tolist())
            for d2 in range(model.n):
                if d2 == s2 or d2 in outs:
                    continue
                opts.append((l2, d2)); cfgs.append(with_(base, "misroutes", l2, d2))
        costs = model.cost_batch(cfgs)
        return [opts[i] for i in np.argsort(costs)]
    free = {kk: dict(v) for kk, v in cfg.items()}
    free["misroutes"] = {}
    pool = [(l, d) for l, d in cfg["misroutes"].items()]
    for opt in options(free):
        if len(pool) >= size - 2 * 0 and len(pool) >= size:
            break
        if opt not in pool:
            pool.append(opt)
    for l in sorted(cfg["misroutes"]):
        for opt in options(without(cfg, "misroutes", l))[:3]:
            if opt not in pool:
                pool.append(opt)
    variants = []
    for combo in itertools.combinations(pool, k):
        links = [c[0] for c in combo]
        if len(set(links)) < k:
            continue
        new = {kk: dict(v) for kk, v in cfg.items()}
        new["misroutes"] = {l2: d2 for l2, d2 in combo}
        if new["misroutes"] == cfg["misroutes"] or not misroutes_disjoint(model, new):
            continue
        busy_all = set(new["servers_down"]) | set(new["slowdown"]) | set(new["buffer_cut"])
        if any(int(model.lfrom[l2]) in busy_all for l2 in new["misroutes"]):
            continue
        variants.append(new)
    if not variants:
        return cfg, cost, False
    raw = model.cost_batch(variants)
    short = np.argsort(raw)[:lookahead]
    ahead = [(screen(model, variants[i], float(raw[i]))[0][0], int(i)) for i in short]     # cost after the best single further error
    ahead.sort()
    target = cost + ERROR_PENALTY * nerr(cfg)
    for _, i in ahead[:top]:
        new, c = refit_slowdowns(model, variants[i], sweeps=1)
        new, c = prune(model, new, c)
        new, c = greedy(model, new, c)
        new, c = prune(model, new, c)
        if c + ERROR_PENALTY * nerr(new) < target - 0.1:
            return new, c, True
    return cfg, cost, False


def refine(model, cfg, cost, deep=False):
    improved = True
    rounds = 0
    while improved and rounds < 24:
        improved = False
        rounds += 1
        cfg, cost, improved = swaps(model, cfg, cost)
        if improved:
            continue
        for kind, key in list(errors_of(cfg)):
            if key not in cfg[kind]:
                continue
            base = without(cfg, kind, key)
            base, bc = refit_slowdowns(model, base, sweeps=1)
            if bc - cost < ADD_GAIN:          # the error does not earn its place
                cfg, cost = greedy(model, base, bc)
                improved = True
                break
            scored = screen(model, base, bc)
            station = int(model.lfrom[key]) if kind == "misroutes" else key
            local = [cand for cand in candidates(model, base) if (int(model.lfrom[cand[1]]) if cand[0] == "misroutes" else cand[1]) == station
                     or (kind == "misroutes" and cand[0] == "misroutes" and cand[2] == cfg["misroutes"][key])]      # same station, or same destination reached over another link
            results = []
            for cand in local + [x[1] for x in scored[:10]]:
                if cand[0] == kind and cand[1] == key and (kind == "slowdown" or cand[2] == cfg[kind][key]):
                    continue                      # the dropped error itself
                results.append(evaluate(model, base, cand))
            results.sort(key=lambda x: x[1])
            for rank, (new, c) in enumerate(results):
                results[rank] = (new, c, False)
            for rank, (new, c, forced) in enumerate(results):
                if c < cost + 3 * ADD_GAIN or rank < 3:      # pruning can only help a candidate that is already close (or one of the three best)
                    new, c = prune(model, new, c)
                if c + ERROR_PENALTY * nerr(new) < cost + ERROR_PENALTY * nerr(cfg) - 0.1:
                    new, c = greedy(model, new, c)
                    if c + ERROR_PENALTY * nerr(new) < cost + ERROR_PENALTY * nerr(cfg) - 0.1:
                        cfg, cost = new, c
                        improved = True
                        break
            if improved:
                break
        if not improved:
            cfg, cost, improved = kick_repair(model, cfg, cost)
        if not improved and deep:
            cfg, cost, improved = joint_misroute_repair(model, cfg, cost)
    return cfg, cost


def gross_test(model, cfg):
    """estimator-free gross test: plain least squares fit, remove the single largest normalised residual while it exceeds the flag, refit"""
    model.mask = np.ones_like(model.z, bool)
    bad = []
    while True:
        cfg, _ = refit_slowdowns(model, cfg, robust=False, sweeps=2)
        r = np.abs(model.resid(cfg))
        r[~model.mask] = 0
        t, j = np.unravel_index(int(np.argmax(r)), r.shape)
        if r[t, j] <= GROSS_FLAG:
            return cfg, sorted(bad)
        model.mask[t, j] = False
        bad.append((int(t), int(j)))


def violations(cfg, bad):
    """how far a diagnosis is outside the stated bounds: at most MAX_ERRORS errors and at most MAX_GROSS gross readings"""
    return max(0, nerr(cfg) - MAX_ERRORS) + max(0, len(bad) - MAX_GROSS)


def fit_statistic(model, cfg, bad):
    """normalised chi-square excess of the plain least-squares fit with the flagged readings left out: about 0 +- 1 for the true configuration"""
    model.mask = np.ones_like(model.z, bool)
    for t, j in bad:
        model.mask[t, j] = False
    r = model.resid(cfg)[model.mask]
    df = r.size - len(continuous_keys(cfg))
    return (float(r @ r) - df) / math.sqrt(2 * df)


def final_objective(model, cfg, bad):
    """penalised plain-least-squares objective of a finished diagnosis (flagged readings left out)"""
    model.mask = np.ones_like(model.z, bool)
    for t, j in bad:
        model.mask[t, j] = False
    return model.cost(cfg, False) + ERROR_PENALTY * nerr(cfg) + GROSS_PENALTY * len(bad)


def search(model, deep=False, **beam_args):
    model.mask = np.ones_like(model.z, bool)
    cfg, cost = beam(model, **beam_args)
    cfg, cost = refine(model, cfg, cost, deep)
    return gross_test(model, cfg)


def deep_refine(model, cfg, rounds=3, margin=3500.0):
    """pair moves: change one error to any alternative at its station that fits almost as well, then let the greedy search add errors"""
    model.mask = np.ones_like(model.z, bool)
    cfg, cost = refit_slowdowns(model, cfg, sweeps=2)
    for _ in range(rounds):
        target = cost + ERROR_PENALTY * nerr(cfg)
        found = None
        for kind, key in list(errors_of(cfg)):
            base = without(cfg, kind, key)          # the other factors stay at their fitted values while the options are screened
            station = int(model.lfrom[key]) if kind == "misroutes" else key
            local = [cand for cand in candidates(model, base) if (int(model.lfrom[cand[1]]) if cand[0] == "misroutes" else cand[1]) == station]
            local = [cand for cand in local if not (cand[0] == kind and cand[1] == key and (kind == "slowdown" or cand[2] == cfg[kind][key]))]
            options = []
            if kind == "misroutes":      # a destination that holds another error: the misroute takes that station and the error goes
                outs = set(model.lto[model.lfrom == station].tolist())
                for d in sorted(set(base["servers_down"]) | set(base["slowdown"]) | set(base["buffer_cut"])):
                    if d == station or d in outs:
                        continue
                    base_d = {kk: dict(v) for kk, v in base.items()}
                    for kd in ("servers_down", "slowdown", "buffer_cut"):
                        base_d[kd].pop(d, None)
                    for l in np.nonzero(model.lfrom == station)[0]:
                        if int(l) != key and misroutes_disjoint(model, with_(base_d, "misroutes", int(l), d)):
                            options.append((base_d, ("misroutes", int(l), d)))
                        elif int(l) == key and misroutes_disjoint(model, with_(base_d, "misroutes", int(l), d)) and d != cfg["misroutes"][key]:
                            options.append((base_d, ("misroutes", int(l), d)))
            for base_d, cand in options:
                base_d, _ = refit_slowdowns(model, base_d, sweeps=1)
                new, c = evaluate(model, base_d, cand)
                new, c = prune(model, new, c)
                new, c = greedy(model, new, c)
                new, c = prune(model, new, c)
                obj = c + ERROR_PENALTY * nerr(new)
                if obj < target - 0.1 and (found is None or obj < found[0]):
                    found = (obj, new, c)
            if not local:
                continue
            trial = [(cand, [with_(base, cand[0], cand[1], f) for f in (0.5, 0.575, 0.65, 0.725, 0.8)] if cand[0] == "slowdown" else [with_(base, *cand)]) for cand in local]
            flat = [c for _, cs in trial for c in cs]
            costs = model.cost_batch(flat)
            pos, scored = 0, []
            for cand, cs in trial:
                scored.append((float(np.min(costs[pos:pos + len(cs)])), cand)); pos += len(cs)
            scored.sort(key=lambda x: x[0])
            for c0, cand in [x for x in scored if x[0] < cost + margin][:4]:
                new, c = evaluate(model, base, cand)
                new, c = prune(model, new, c)
                new, c = greedy(model, new, c)
                new, c = prune(model, new, c)
                obj = c + ERROR_PENALTY * nerr(new)
                if obj < target - 0.1 and (found is None or obj < found[0]):
                    found = (obj, new, c)
        if found is None:
            break
        cfg, cost = found[1], found[2]
        cfg, cost = refine(model, cfg, cost)
    return cfg


def restart_search(model, seed, drop_prob=0.3, deep=False):
    """beam search from scratch with a random subset of (error kind, station or link) options forbidden, then a free refinement"""
    rng = np.random.default_rng(seed)
    pairs = [(k, s) for k in ("servers_down", "buffer_cut", "slowdown") for s in range(model.n)] + [("misroutes", l) for l in range(len(model.lfrom))]
    model.tabu = {p for p in pairs if rng.random() < drop_prob}
    model.mask = np.ones_like(model.z, bool)
    cfg, cost = beam(model)
    model.tabu = set()
    cfg, cost = refine(model, cfg, cost, deep)
    return gross_test(model, cfg)


def polish(model, cfg, bad):
    """final clean-up: drop an error and look for the gross readings again (an error may only be fitting an outlier); keep the
    change when the penalised objective falls"""
    cfg, bad = gross_test(model, cfg)
    obj = final_objective(model, cfg, bad)
    while True:
        best = None
        for kind, key in list(errors_of(cfg)):
            cand, bad2 = gross_test(model, without(cfg, kind, key))
            o = final_objective(model, cand, bad2)
            if o < obj - 0.1 and (best is None or o < best[0]):
                best = (o, cand, bad2)
        if best is None:
            return cfg, bad
        obj, cfg, bad = best


def diagnose(spec, z, deadline=None):
    model = Model(spec, z)
    def out_of_time():
        return deadline is not None and time.monotonic() > deadline
    def good(c, b):
        return violations(c, b) == 0 and fit_statistic(model, c, b) <= FIT_LIMIT
    cfg, bad = search(model)
    cfg, bad = polish(model, cfg, bad)
    best = (final_objective(model, cfg, bad), cfg, bad)
    def consider(c, b):
        nonlocal best
        o = final_objective(model, c, b)
        if o < best[0]:
            best = (o, c, b)
    if not good(cfg, bad) and not out_of_time():       # the diagnosis breaks a stated bound or fits the readings badly: stronger moves, then restarts
        cfg4 = deep_refine(model, best[1])
        cfg4, bad4 = gross_test(model, cfg4)
        consider(cfg4, bad4)
    if not good(best[1], best[2]) and not out_of_time():
        model.mask = np.ones_like(model.z, bool)
        cfg1, cost1 = refit_slowdowns(model, best[1], sweeps=2)
        cfg1, cost1 = refine(model, cfg1, cost1, deep=True)
        cfg1, bad1 = gross_test(model, cfg1)
        consider(cfg1, bad1)
    seed = 0
    while not good(best[1], best[2]) and seed < RESTARTS and not out_of_time():
        cfg2, bad2 = restart_search(model, seed)
        consider(cfg2, bad2)
        seed += 1
    if not good(best[1], best[2]) and not out_of_time():
        cfg3, bad3 = search(model, deep=True, width=3 * BEAM_WIDTH, expand=BEAM_EXPAND + 4, per_kind=PER_KIND + 4)
        consider(cfg3, bad3)
    _, cfg, bad = best
    cfg, bad = polish(model, cfg, bad)
    return {
        "servers_down": {str(s): int(d) for s, d in sorted(cfg["servers_down"].items())},
        "slowdowns": {str(s): round(min(max(float(f), 0.5), 0.8), 4) for s, f in sorted(cfg["slowdown"].items())},
        "buffer_cuts": {str(s): int(k) for s, k in sorted(cfg["buffer_cut"].items())},
        "misroutes": {str(l): int(d) for l, d in sorted(cfg["misroutes"].items())},
        "demand_factors": [round(min(max(float(cfg["demand"].get(t, 1.0)), 0.8), 1.25), 4) for t in range(model.T)],
        "service_factor": round(min(max(float(cfg["service"].get(0, 1.0)), 0.85), 1.15), 4),
        "bad_readings": [list(p) for p in bad],
    }


def _solve(spec, z, start):
    """one network with a wall-clock allowance: the whole run stops searching further at GLOBAL_SECONDS, one network at NETWORK_SECONDS"""
    now = time.monotonic()
    return diagnose(spec, z, deadline=min(start + GLOBAL_SECONDS, now + NETWORK_SECONDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    root = Path(a.input_dir)
    ids = json.loads((root / "campaign.json").read_text())["networks"]
    jobs = []
    for nid in ids:
        spec = json.loads((root / f"network_{nid}.json").read_text())
        with np.load(root / f"readings_{nid}.npz") as data:
            jobs.append((spec, np.array(data["z"], float)))
    # the networks are independent: two worker processes share them (the stated limit is 2 CPUs)
    start = time.monotonic()
    jobs = [(spec, z, start) for spec, z in jobs]
    with multiprocessing.get_context("fork").Pool(2) as pool:
        answers = pool.starmap(_solve, jobs, chunksize=1)
    result = dict(zip(ids, answers))
    (Path(a.output_dir) / "diagnosis.json").write_text(json.dumps({"networks": result}))


if __name__ == "__main__":
    main()
