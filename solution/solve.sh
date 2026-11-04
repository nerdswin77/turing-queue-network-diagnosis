#!/bin/bash
set -euo pipefail

# The deliverable is a reusable program, so the oracle installs the reference solver at the required path ...
mkdir -p /root/results/development
cp /solution/solver.py /root/results/solver.py
chmod +x /root/results/solver.py

# ... and exercises it on the development campaign. The solver reads every campaign file (campaign.json, each network_<ID>.json and
# readings_<ID>.npz), fits the reduced-load M/M/c/K network model under hypothesised configurations, with the demand factor of
# every period and the service factor estimated jointly, and writes the diagnosis.
python3 -B /root/results/solver.py --input-dir /root/data --output-dir /root/results/development

# Agentic check: the computed diagnosis is compared with the development answer key shipped with the data.
python3 - <<'PY'
import json
produced = json.load(open('/root/results/development/diagnosis.json'))['networks']
key = json.load(open('/root/data/development_key.json'))['networks']
assert sorted(produced) == sorted(key), 'solver output does not cover the campaign networks'
for nid, e in produced.items():
    k = key[nid]
    for field in ('servers_down', 'buffer_cuts', 'misroutes', 'bad_readings'):
        assert e[field] == k[field], (nid, field, e[field], k[field])
    assert set(e['slowdowns']) == set(k['slowdowns']), (nid, 'slowdowns')
    assert all(abs(e['slowdowns'][s] - f) <= 0.03 for s, f in k['slowdowns'].items()), (nid, 'slowdown factors')
    assert all(abs(u - v) <= 0.05 for u, v in zip(e['demand_factors'], k['demand_factors'])), (nid, 'demand factors')
    assert abs(e['service_factor'] - k['service_factor']) <= 0.01, (nid, 'service factor')
print('development campaign matches the answer key:', {nid: {f: e[f] for f in ('servers_down', 'slowdowns', 'buffer_cuts', 'misroutes')} for nid, e in produced.items()})
PY
