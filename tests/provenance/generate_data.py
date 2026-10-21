#!/usr/bin/env python3
"""Deterministic data generator for queue-network-diagnosis (development campaign D and private campaigns H1..H3).

Ground truth is INJECTED here: the hidden station errors, misrouted dispatch links, the actual demand factor of every period,
the actual service factor and the gross readings are drawn, the readings are simulated with the forward model of qn_physics.py, and the drawn values
are written as the truth. The
reference solver is never called. Every draw is checked by the structural acceptance rules of qn_rules.py on noise-free
readings; an error that breaks a rule is redrawn. Usage: QND_ROOT=<task folder> python3 generate_data.py all
"""
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qn_physics import SIGMA, predict  # noqa: E402
from qn_rules import accept  # noqa: E402

ROOT = Path(os.environ.get("QND_ROOT", Path(__file__).resolve().parents[2]))
CAMPAIGNS = {"D": (["N1", "N2"], 5004), "H1": (["A1", "A2", "A3", "A4", "A5", "A6"], 6001),
             "H2": (["B1", "B2", "B3", "B4", "B5", "B6"], 7001), "H3": (["C1", "C2", "C3", "C4", "C5", "C6"], 8001)}
# networks drawn again with a different seed salt (none in this version); every network keeps its original seed
REDRAWN = {}
MAX_GROSS = 2
N_RANGE = (20, 28)             # stations per network
E_RANGE = (2, 5)               # errors per network, not counting gross readings
DEMAND_RANGE = (0.8, 1.25)     # actual demand factor of a period (the forecast error of the arrival volume)
SERVICE_RANGE = (0.85, 1.15)   # actual service factor of the network (the productivity of the day relative to the stored rates)
FRACS = (0.6, 0.2, 0.2, 0.3)     # share of stations with an occupancy, offered or lost meter, and of links with a dispatch meter
GROSS_MIN_SIGMA, GROSS_MAX_SIGMA = 12.0, 30.0
NON_GROSS_MAX_SIGMA = 4.5
MAX_SERVERS_DOWN = 2
MIN_STATION_FLOW = 0.5          # jobs per minute offered to every station at arrival scale 1, before losses
OCC_FRAC = 0.8


def make_spec(rng, nid):
    n = int(rng.integers(N_RANGE[0], N_RANGE[1] + 1))
    links = []
    for i in range(n - 1):
        k = 1 if (i >= n - 3 or rng.random() < 0.45) else 2
        cand = list(range(i + 1, min(n, i + 7)))
        succ = sorted(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())
        tot = rng.uniform(0.6, 0.92) if i < n - 1 else 0
        w = rng.dirichlet(np.ones(len(succ))) * tot
        for j, p in zip(succ, w):
            links.append([i, int(j), float(p)])
        if i >= 3 and rng.random() < 0.18:
            j = int(rng.integers(max(0, i - 6), i))
            links.append([i, j, float(rng.uniform(0.04, 0.08))])
    links.sort(key=lambda x: (x[0], x[1]))
    # remove duplicates
    seen = set(); L2 = []
    for a, b, p in links:
        if (a, b) in seen:
            continue
        seen.add((a, b)); L2.append([a, b, round(p, 4)])
    # ensure probabilities from a station sum <= 0.95
    links = [{"id": k, "from": a, "to": b, "prob": p} for k, (a, b, p) in enumerate(L2)]
    has_in = {l["to"] for l in links if l["to"] != l["from"]}
    sources = {0} | set(rng.choice(np.arange(1, n), size=int(rng.integers(1, 4)), replace=False).tolist())
    for j in range(1, n):
        if j not in has_in:
            sources.add(j)
    ext = np.zeros(n)
    for s in sources:
        ext[s] = rng.uniform(0.5, 3.0) if s else rng.uniform(2.0, 5.0)
    P = np.zeros((n, n))
    for l in links:
        P[l["from"], l["to"]] += l["prob"]
    lam = np.linalg.solve(np.eye(n) - P.T, ext)
    if lam.min() < MIN_STATION_FLOW:                      # every station carries a real flow, so its rates are not degenerate
        return None
    stations = []
    for s in range(n):
        c = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.3, 0.25, 0.2]))
        rho = rng.uniform(0.55, 0.9)
        mu = lam[s] / (c * rho)
        K = c + int(rng.integers(3, 11))
        stations.append({"id": s, "servers": c, "service_rate": round(float(mu), 4), "capacity": K, "external_rate": round(float(ext[s]), 4)})
    scales = np.sort(rng.uniform(0.35, 1.45, size=6))
    scales[0] = rng.uniform(0.3, 0.4); scales[-1] = rng.uniform(1.4, 1.5)
    scales = np.sort(np.round(scales, 3))
    spec = {"id": nid, "stations": stations, "links": links, "arrival_scale": [float(x) for x in scales]}
    meters = []
    for ty, frac, key, N in (("occupancy", FRACS[0], "station", n), ("offered", FRACS[1], "station", n), ("lost", FRACS[2], "station", n), ("dispatch", FRACS[3], "link", len(links))):
        sel = sorted(rng.choice(N, size=max(2, int(round(frac * N))), replace=False).tolist())
        for x in sel:
            meters.append({"type": ty, key: int(x), "sigma": SIGMA[ty]})
    for i, m in enumerate(meters):
        m["index"] = i
    spec["meters"] = [dict(index=m["index"], **{k: v for k, v in m.items() if k != "index"}) for m in meters]
    return spec




def draw_one(rng, spec, cfg, used):
    st = spec["stations"]; n = len(st)
    for _ in range(500):
        s = int(rng.integers(0, n))
        if s in used:
            continue
        kind = str(rng.choice(["servers_down", "slowdown", "buffer_cut", "misroutes"]))
        c, K = st[s]["servers"], st[s]["capacity"]
        if kind == "servers_down":
            if c < 2: continue
            cfg[kind][s] = int(rng.integers(1, min(MAX_SERVERS_DOWN, c - 1) + 1))
        elif kind == "slowdown":
            cfg[kind][s] = round(float(rng.uniform(0.5, 0.8)), 3)
        elif kind == "buffer_cut":
            if K - c < 2: continue
            cfg[kind][s] = int(rng.integers(1, K - c))          # the reduced capacity is at least servers + 1
        else:
            out = [l for l in spec["links"] if l["from"] == s]
            if not out: continue
            l = out[int(rng.integers(0, len(out)))]
            # the from, to and actual destination of a misrouted link touch no station of another misrouted link
            taken = {x for k, d in cfg["misroutes"].items() for x in (spec["links"][k]["from"], spec["links"][k]["to"], d)}
            if s in taken or l["to"] in taken: continue
            dests = [j for j in range(n) if j != s and not any(o["to"] == j for o in out) and j not in taken]
            if not dests: continue
            cfg[kind][l["id"]] = int(rng.choice(dests))
        used.add(s)
        return
    raise RuntimeError("no error slot")


def station_of(spec, kind, key):
    return spec["links"][key]["from"] if kind == "misroutes" else key


def draw_demand(rng, spec):
    return {t: round(float(rng.uniform(*DEMAND_RANGE)), 3) for t in range(len(spec["arrival_scale"]))}


def draw_accepted(rng, spec, max_redraws=40):
    """draw the demand factors and E_RANGE errors, redraw any error that breaks a structural rule; None if the network cannot
    carry them or if one of its demand factors would be imprecise"""
    E = int(rng.integers(E_RANGE[0], E_RANGE[1] + 1))
    cfg = {"servers_down": {}, "slowdown": {}, "buffer_cut": {}, "misroutes": {}, "demand": draw_demand(rng, spec),
           "service": {0: round(float(rng.uniform(*SERVICE_RANGE)), 3)}}
    used = set()
    for _ in range(E):
        draw_one(rng, spec, cfg, used)
    for _ in range(max_redraws):
        ok, bad = accept(spec, cfg, predict(spec, cfg))
        if ok:
            return cfg
        kind, key, _ = bad
        if kind in ("demand", "service"):
            return None                                  # the precision of a demand factor is a property of the meters, not of the draw
        used.discard(station_of(spec, kind, key))
        del cfg[kind][key]
        draw_one(rng, spec, cfg, used)
    return None


def make_network(nid, seed):
    """one accepted network: spec, configuration, readings z and gross set"""
    rng = np.random.default_rng(seed)
    while True:
        spec = make_spec(rng, nid)
        if spec is None:
            continue
        cfg = draw_accepted(rng, spec)
        if cfg is None:
            continue
        y = predict(spec, cfg)
        sig = np.array([m["sigma"] for m in spec["meters"]])
        noise = rng.normal(size=y.shape) * sig
        if np.max(np.abs(noise) / sig) > NON_GROSS_MAX_SIGMA:      # ordinary readings stay inside the stated noise band
            continue
        z = y + noise
        T, M = z.shape
        G = int(rng.integers(0, MAX_GROSS + 1))
        cells = sorted({(int(t), int(j)) for t, j in zip(rng.integers(0, T, size=G), rng.integers(0, M, size=G))})
        for t, j in cells:
            z[t, j] = y[t, j] + float(rng.choice([-1.0, 1.0])) * rng.uniform(GROSS_MIN_SIGMA, GROSS_MAX_SIGMA) * sig[j]
        return spec, cfg, z, cells


def truth_entry(cfg, cells):
    return {"servers_down": {str(k): int(v) for k, v in sorted(cfg["servers_down"].items())},
            "slowdowns": {str(k): float(v) for k, v in sorted(cfg["slowdown"].items())},
            "buffer_cuts": {str(k): int(v) for k, v in sorted(cfg["buffer_cut"].items())},
            "misroutes": {str(k): int(v) for k, v in sorted(cfg["misroutes"].items())},
            "demand_factors": [float(cfg["demand"][t]) for t in sorted(cfg["demand"])],
            "service_factor": float(cfg["service"][0]),
            "bad_readings": [[t, j] for t, j in cells]}


def write_campaign(name, out_dir):
    ids, base = CAMPAIGNS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = {}
    files = {"campaign.json": json.dumps({"networks": ids}, indent=1).encode()}
    for k, nid in enumerate(ids):
        spec, cfg, z, cells = make_network(nid, base + 7919 * k + 1000003 * REDRAWN.get(nid, 0))
        files[f"network_{nid}.json"] = json.dumps(spec, indent=1).encode()
        buf = io.BytesIO()
        np.savez(buf, z=np.round(z, 6))
        files[f"readings_{nid}.npz"] = buf.getvalue()
        truth[nid] = truth_entry(cfg, cells)
        print(name, nid, truth[nid], flush=True)
    return files, {"networks": truth}


def archive(files, path):
    """deterministic tar.gz: sorted members, owner 0, mode 0644, mtime 0, gzip header mtime 0"""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = len(files[name]), 0, 0o644, 0, 0, "", ""
            tar.addfile(info, io.BytesIO(files[name]))
    with open(path, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
            gz.write(raw.getvalue())


def main(which):
    names = list(CAMPAIGNS) if which == "all" else [which]
    for name in names:
        if name == "D":
            files, truth = write_campaign("D", ROOT / "environment" / "data")
            for fname, data in files.items():
                (ROOT / "environment" / "data" / fname).write_bytes(data)
            (ROOT / "environment" / "data" / "development_key.json").write_text(json.dumps(truth, indent=1) + "\n")
            (ROOT / "tests" / "provenance" / "development_truth.json").write_text(json.dumps(truth, indent=1) + "\n")
            continue
        case = ROOT / "tests" / "cases" / name
        if (case / "truth.json").exists() and (case / f"{name}_input.tar.gz").exists():
            print(name, "exists, skipped (resumable)")
            continue
        files, truth = write_campaign(name, case)
        archive(files, case / f"{name}_input.tar.gz")
        (case / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    sums = {}
    for name in ("H1", "H2", "H3"):
        a = ROOT / "tests" / "cases" / name / f"{name}_input.tar.gz"
        if a.exists():
            sums[name] = hashlib.sha256(a.read_bytes()).hexdigest()
    (ROOT / "tests" / "cases" / "archive_checksums.json").write_text(json.dumps(sums, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
