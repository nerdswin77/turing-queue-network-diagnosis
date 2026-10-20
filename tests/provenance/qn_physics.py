"""Forward model of a queueing network of M/M/c/K stations with probabilistic routing (reduced-load fixed point).

The outside arrival rate of station s in period t is external_rate[s] * arrival_scale[t] * demand[t], where demand[t] is the
period's actual demand factor (nominal 1); every service rate is service_rate[s] * service[0], the network's actual service
factor (nominal 1), times the station's slowdown factor if it is slowed. Both factors are unknown to the solver and part of
the configuration."""
import numpy as np

KINDS = ("servers_down", "slowdown", "buffer_cut")
TYPES = ("occupancy", "offered", "lost", "dispatch")
SIGMA = {"occupancy": 0.05, "offered": 0.04, "lost": 0.01, "dispatch": 0.03}


def station_stats(lam, mu, c, K):
    """lam, mu, c, K arrays (n,). returns L (mean number in station) and B (blocking probability)."""
    n = lam.shape[0]
    Kmax = int(K.max())
    k = np.arange(1, Kmax + 1)[None, :]
    rate = np.minimum(k, c[:, None]) * mu[:, None]
    with np.errstate(divide="ignore"):
        step = np.log(lam[:, None]) - np.log(rate)
    step = np.where(k <= K[:, None], step, -np.inf)
    logp = np.concatenate([np.zeros((n, 1)), np.cumsum(step, axis=1)], axis=1)
    m = logp.max(axis=1, keepdims=True)
    p = np.exp(logp - m)
    p /= p.sum(axis=1, keepdims=True)
    L = p @ np.arange(Kmax + 1)
    B = p[np.arange(n), K]
    return L, B


def solve_network(ext, P, mu, c, K, tol=1e-13, maxit=5000):
    """ext (n,) external arrival rate, P (n,n) routing prob from i to j. returns offered lam, throughput theta, L, B."""
    lam = ext.copy()
    for _ in range(maxit):
        L, B = station_stats(lam, mu, c, K)
        theta = lam * (1.0 - B)
        new = ext + P.T @ theta
        if np.max(np.abs(new - lam)) <= tol * (1.0 + np.max(new)):
            lam = new
            break
        lam = new
    L, B = station_stats(lam, mu, c, K)
    return lam, lam * (1.0 - B), L, B


def build(spec, config):
    """spec: network dict; config: dict with servers_down{st:d}, slowdown{st:f}, buffer_cut{st:k}, misroutes{link:to}."""
    st = spec["stations"]
    n = len(st)
    mu = np.array([s["service_rate"] for s in st], float)
    c = np.array([s["servers"] for s in st], int)
    K = np.array([s["capacity"] for s in st], int)
    ext = np.array([s["external_rate"] for s in st], float)
    mu *= float(config.get("service", {}).get(0, 1.0))
    for s, d in config.get("servers_down", {}).items():
        c[int(s)] -= int(d)
    for s, f in config.get("slowdown", {}).items():
        mu[int(s)] *= float(f)
    for s, k in config.get("buffer_cut", {}).items():
        K[int(s)] -= int(k)
    P = np.zeros((n, n))
    mis = {int(a): int(b) for a, b in config.get("misroutes", {}).items()}
    for l in spec["links"]:
        to = mis.get(l["id"], l["to"])
        P[l["from"], to] += l["prob"]
    return ext, P, mu, c, K


def predict(spec, config):
    """returns array (periods, meters) of noise-free readings."""
    ext, P, mu, c, K = build(spec, config)
    meters = spec["meters"]
    links = spec["links"]
    demand = config.get("demand", {})
    out = np.zeros((len(spec["arrival_scale"]), len(meters)))
    for t, a in enumerate(spec["arrival_scale"]):
        lam, theta, L, B = solve_network(ext * a * float(demand.get(t, 1.0)), P, mu, c, K)
        for j, m in enumerate(meters):
            ty = m["type"]
            if ty == "occupancy":
                out[t, j] = L[m["station"]]
            elif ty == "offered":
                out[t, j] = lam[m["station"]]
            elif ty == "lost":
                out[t, j] = lam[m["station"]] * B[m["station"]]
            else:
                l = links[m["link"]]
                out[t, j] = theta[l["from"]] * l["prob"]
    return out
