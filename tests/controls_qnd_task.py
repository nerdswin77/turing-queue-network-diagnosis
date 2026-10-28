"""Negative-control definitions for solver-deliverable tasks.

Each control is {label, code, expect_fail, patch?}. `code` is the full text written to /root/results/solver.py (None = no
solver at all). expect_fail=True means the verifier must reject it; False means it is a legitimate solution the verifier must
ACCEPT (guards against over-strict checks - several review failures were verifiers rejecting legitimate output layouts).

The CLI used here (--input-dir / --output-dir, output file parameters.json) and the mutations in the "output" section are
specific to each task (this module: queue-network-diagnosis): when you start a new task, edit OUTPUT_FILE, the CLI preamble and the output mutations; keep the
generic security / resource / hygiene controls as they are.
"""
OUTPUT_FILE = "diagnosis.json"

PREAMBLE = """
import argparse as _ap, json as _j, os as _os
_p = _ap.ArgumentParser(); _p.add_argument('--input-dir'); _p.add_argument('--output-dir'); _a = _p.parse_args()
_f = _os.path.join(_a.output_dir, '%s')
""" % OUTPUT_FILE


def controls(good, truth_path):
    def mutate(body):          # run the real solver first (appended code runs after it), then rewrite its output
        return good + PREAMBLE + "_d = _j.load(open(_f))\n" + body + "_j.dump(_d, open(_f, 'w'))\n"

    def raw(body): return good + PREAMBLE + "_t = open(_f).read()\n" + body
    real = good.replace('"%s"' % OUTPUT_FILE, '"real_output.json"')
    L = []
    add = lambda label, code, fail=True, **kw: L.append(dict(label=label, code=code, expect_fail=fail, **kw))
    # --- hygiene / security / resources: generic ------------------------------------------------------------------------
    add("NOP: no solver submitted", None)
    add("reads the ground truth", f"print(open({truth_path!r}).read())\n")
    add("tries to become root", "import os; os.setuid(0)\n")
    add("writes the wrong output filename", good.replace('"%s"' % OUTPUT_FILE, '"result.json"'))
    add("writes next to its inputs", PREAMBLE.replace("_f =", "_g =") + "open(_os.path.join(_a.input_dir, 'x.txt'), 'w').write('x')\n")
    add("imports an unavailable package", "import not_a_real_package\n")
    _site = "import glob, sys\nfor _d in glob.glob('/tmp/qnd-pytest-site') + glob.glob('/usr/local/lib/python3*/site-packages') + glob.glob('/usr/local/lib/python3*/dist-packages'):\n    sys.path.append(_d)\n"
    add("imports pytest, which the verifier image holds (only the standard library, numpy and scipy are allowed)", _site + "import pytest\n" + good)
    add("imports pytest inside a multiprocessing worker (the restriction must reach worker processes)", _site + "import multiprocessing as _mp\ndef _w(_x):\n    import pytest\n    return 1\nif __name__ == '__main__':\n    with _mp.get_context('spawn').Pool(1) as _pool:\n        _pool.map(_w, [0])\n" + good)
    add("reaches pytest through importlib.import_module", _site + "import importlib\nimportlib.import_module('pytest')\n" + good)
    add("PASS: imports the standard library, numpy and many scipy modules", "import json, math, heapq, itertools, multiprocessing, concurrent.futures, numpy, scipy.optimize, scipy.linalg, scipy.sparse, scipy.special, scipy.stats\n" + good, fail=False)
    add("hangs (timeout patched to 4 s)", "import time; time.sleep(60)\n", patch=("SOLVER_TIMEOUT_SECONDS = 900", "SOLVER_TIMEOUT_SECONDS = 4"))
    add("allocates 6 GiB (stated limit 4 GB)", "x = bytearray(6 * 1024 ** 3)\n")
    add("starts 100 threads (stated limit 64 processes and threads)", "import threading, time\n_ts = [threading.Thread(target=time.sleep, args=(3,)) for _ in range(100)]\nfor _t in _ts:\n    _t.start()\n" + good)
    add("writes a 20 MB file (stated limit 16 MB per file)", PREAMBLE.replace("_f =", "_g =") + "open(_os.path.join(_a.output_dir, 'big.bin'), 'wb').write(b'0' * (20 * 1024 * 1024))\n" + good)
    add("symlinks the output to /etc/passwd", PREAMBLE.replace("_f =", "_g =") + "_os.symlink('/etc/passwd', _os.path.join(_a.output_dir, '%s'))\n" % OUTPUT_FILE)
    add("symlinks the output to the ground truth", PREAMBLE.replace("_f =", "_g =") + "_os.symlink(%r, _os.path.join(_a.output_dir, '%s'))\n" % (truth_path, OUTPUT_FILE))
    # --- content mutations: queue-network diagnosis schema ---------------------------------------------------------------------
    pick = lambda key: "_n = next(v for v in _d['networks'].values() if v['%s'])\n_k = next(iter(_n['%s']))\n" % (key, key)
    add("a server outage missing from the report", mutate(pick("servers_down") + "del _n['servers_down'][_k]\n"))
    add("a server outage count wrong (1 <-> 2 where allowed, else reported on a 1-server station)", mutate(pick("servers_down") + "_n['servers_down'][_k] = 3 - _n['servers_down'][_k]\n"))
    add("a slowdown factor off by 0.05", mutate(pick("slowdowns") + "_n['slowdowns'][_k] = _n['slowdowns'][_k] + (0.05 if _n['slowdowns'][_k] < 0.75 else -0.05)\n"))
    add("PASS: a slowdown factor off by 0.02 (inside the 0.03 tolerance)", mutate(pick("slowdowns") + "_n['slowdowns'][_k] = round(_n['slowdowns'][_k] + (0.02 if _n['slowdowns'][_k] < 0.75 else -0.02), 6)\n"), fail=False)
    add("a slowdown missing", mutate(pick("slowdowns") + "del _n['slowdowns'][_k]\n"))
    add("a slowdown factor outside 0.5..0.8 (0.45)", mutate(pick("slowdowns") + "_n['slowdowns'][_k] = 0.45\n"))
    add("a slowdown factor just below the range (0.49)", mutate(pick("slowdowns") + "_n['slowdowns'][_k] = 0.49\n"))
    add("a slowdown factor just above the range (0.8014)", mutate(pick("slowdowns") + "_n['slowdowns'][_k] = 0.8014\n"))
    add("a buffer cut off by one", mutate(pick("buffer_cuts") + "_n['buffer_cuts'][_k] = _n['buffer_cuts'][_k] + (1 if _n['buffer_cuts'][_k] == 1 else -1)\n"))
    add("a buffer cut missing", mutate(pick("buffer_cuts") + "del _n['buffer_cuts'][_k]\n"))
    add("a misroute reported with the wrong destination", mutate(pick("misroutes") + "_n['misroutes'][_k] = _n['misroutes'][_k] + 1 if _n['misroutes'][_k] == 0 else _n['misroutes'][_k] - 1\n"))
    add("a misroute missing", mutate(pick("misroutes") + "del _n['misroutes'][_k]\n"))
    add("a gross-reading pair missing", mutate("_n = next(v for v in _d['networks'].values() if v['bad_readings']); _n['bad_readings'] = _n['bad_readings'][1:]\n"))
    add("a wrong extra gross-reading pair", mutate("_n = next(iter(_d['networks'].values())); _n['bad_readings'] = sorted({tuple(p) for p in _n['bad_readings']} | {(5, 0)}); _n['bad_readings'] = [list(p) for p in _n['bad_readings']]\n"))
    add("bad_readings not strictly increasing", mutate("_n = next(iter(_d['networks'].values())); _n['bad_readings'] = [[3, 1], [0, 2]]\n"))
    add("bad_readings pair outside the readings array", mutate("_n = next(iter(_d['networks'].values())); _n['bad_readings'] = sorted(_n['bad_readings'] + [[6, 0]])\n"))
    add("a station key with a leading zero", mutate(pick("buffer_cuts") + "_n['buffer_cuts']['0' + _k] = _n['buffer_cuts'].pop(_k)\n"))
    add("an extra error on a station that already has one", mutate(pick("buffer_cuts") + "_n['slowdowns'][_k] = 0.7\n"))
    add("a demand factor off by 0.08", mutate("_n = next(iter(_d['networks'].values()))\n_n['demand_factors'][2] = round(_n['demand_factors'][2] + (0.08 if _n['demand_factors'][2] < 1.0 else -0.08), 4)\n"))
    add("PASS: a demand factor off by 0.03 (inside the 0.05 tolerance)", mutate("_n = next(iter(_d['networks'].values()))\n_n['demand_factors'][2] = round(_n['demand_factors'][2] + (0.03 if _n['demand_factors'][2] < 1.0 else -0.03), 4)\n"), fail=False)
    add("a demand factor outside 0.8..1.25 (0.79)", mutate("_n = next(iter(_d['networks'].values()))\n_n['demand_factors'][0] = 0.79\n"))
    add("demand_factors with one entry too few", mutate("_n = next(iter(_d['networks'].values()))\n_n['demand_factors'] = _n['demand_factors'][:-1]\n"))
    add("a service factor off by 0.02", mutate("_n = next(iter(_d['networks'].values()))\n_n['service_factor'] = round(_n['service_factor'] + (0.02 if _n['service_factor'] < 1.0 else -0.02), 4)\n"))
    add("PASS: a service factor off by 0.005 (inside the 0.01 tolerance)", mutate("_n = next(iter(_d['networks'].values()))\n_n['service_factor'] = round(_n['service_factor'] + (0.005 if _n['service_factor'] < 1.0 else -0.005), 4)\n"), fail=False)
    add("a service factor outside 0.85..1.15 (1.16)", mutate("_n = next(iter(_d['networks'].values()))\n_n['service_factor'] = 1.16\n"))
    add("extra key in a network entry (schema)", mutate("next(iter(_d['networks'].values()))['extra'] = 1\n"))
    add("a network missing from the report", mutate("_d['networks'].pop(next(iter(_d['networks'])))\n"))
    add("a buffer cut reported as a float", mutate(pick("buffer_cuts") + "_n['buffer_cuts'][_k] = float(_n['buffer_cuts'][_k])\n"))
    add("a slowdown reported as NaN", raw("_dd = _j.loads(_t); _n = next(v for v in _dd['networks'].values() if v['slowdowns']); _n['slowdowns'][next(iter(_n['slowdowns']))] = float('nan'); open(_f, 'w').write(_j.dumps(_dd))\n"))
    # --- must PASS: legitimate organisations of the output (verify the verifier is outcome-only) --------------------------------
    add("PASS: output is a symlink to a file inside OUTPUT_DIR", real + PREAMBLE.replace("_f =", "_g =") + "_os.symlink(_os.path.join(_a.output_dir, 'real_output.json'), _os.path.join(_a.output_dir, '%s'))\n" % OUTPUT_FILE, fail=False)
    add("PASS: output is a hard link inside OUTPUT_DIR", real + PREAMBLE.replace("_f =", "_g =") + "_os.link(_os.path.join(_a.output_dir, 'real_output.json'), _os.path.join(_a.output_dir, '%s'))\n" % OUTPUT_FILE, fail=False)
    add("PASS: uses tempfile (lands in OUTPUT_DIR)", "import tempfile\nf = tempfile.NamedTemporaryFile(delete=False); f.write(b'x'); f.close()\n" + good, fail=False)
    add("PASS: forks a child that exits, then solves", "import os\nif os.fork() == 0:\n    os._exit(0)\n" + good, fail=False)
    add("PASS: sees at most 2 CPUs and still works", "import os\nassert len(os.sched_getaffinity(0)) <= 2\n" + good, fail=False)
    add("PASS: uses multiprocessing.Pool(2), then solves", "import multiprocessing as mp\nwith mp.Pool(2) as _p:\n    _p.map(abs, [-1, -2])\n" + good, fail=False)
    add("PASS: uses multiprocessing.Manager and a Queue, then solves", "import multiprocessing as mp\n_m = mp.Manager(); _q = _m.Queue(); _q.put(1); _q.get()\n" + good, fail=False)
    add("PASS: leaves a semaphore-named file in /dev/shm; it must be gone for the next campaign", "import os\n_p = '/dev/shm/sem.mp-leftover'\nassert not os.path.exists(_p), 'state carried between campaigns'\nopen(_p, 'w').write('x')\n" + good, fail=False)
    add("creates a directory in /dev/shm", "import os\nos.mkdir('/dev/shm/stash')\n" + good)
    add("writes /dev/shm/sem.mp-A followed by a newline (a semaphore name plus a newline is not a multiprocessing name)", "open('/dev/shm/sem.mp-A\\n', 'w').write('x')\n" + good)
    add("writes /dev/shm/psm_A followed by a newline (a segment name plus a newline is not a multiprocessing name)", "open('/dev/shm/psm_A\\n', 'w').write('x')\n" + good)
    add("creates a prohibited file in /dev/shm and removes it again before it exits (the check must be in flight, not only after the exit)", "import os\nopen('/dev/shm/forbidden_state', 'w').write('x')\nos.remove('/dev/shm/forbidden_state')\n" + good)
    add("creates a prohibited directory in /dev/shm and removes it again before it exits", "import os, shutil\nos.mkdir('/dev/shm/stash2')\nshutil.rmtree('/dev/shm/stash2')\n" + good)
    add("PASS: creates and removes a semaphore-named file in /dev/shm (the names multiprocessing uses)", "import os\nopen('/dev/shm/sem.mp-tmpABC', 'w').write('x')\nos.remove('/dev/shm/sem.mp-tmpABC')\n" + good, fail=False)
    add("PASS: creates and removes a FIFO named /dev/shm/psm_fifo (an entry that is not a directory, under an allowed name; nothing in /dev/shm survives the invocation)", "import os\nos.mkfifo('/dev/shm/psm_fifo')\nos.remove('/dev/shm/psm_fifo')\n" + good, fail=False)
    add("creates and removes a FIFO named /dev/shm/forbidden_fifo (the name is not allowed, whatever the type)", "import os\nos.mkfifo('/dev/shm/forbidden_fifo')\nos.remove('/dev/shm/forbidden_fifo')\n" + good)
    add("creates and removes a DIRECTORY named /dev/shm/psm_dir (a directory is never allowed: the names inside it would not be watched)", "import os\nos.mkdir('/dev/shm/psm_dir')\nos.rmdir('/dev/shm/psm_dir')\n" + good)
    add("leaves a directory named /dev/shm/sem.mp-dir (a directory is never allowed)", "import os\nos.mkdir('/dev/shm/sem.mp-dir')\n" + good)
    add("creates and removes 40,000 allowed-name entries in /dev/shm (more than the 1000 entries stated), which could overflow the inotify queue", "import os\nfor _i in range(40000):\n    _p = '/dev/shm/psm_%d' % _i\n    open(_p, 'w').close()\n    os.remove(_p)\n" + good)
    add("PASS: creates and removes 500 allowed-name entries in /dev/shm (inside the 1000 stated)", "import os\nfor _i in range(500):\n    _p = '/dev/shm/psm_%d' % _i\n    open(_p, 'w').close()\n    os.remove(_p)\n" + good, fail=False)
    add("PASS: creates and removes the temporary name of a semaphore (sem. and six letters or digits) in /dev/shm", "import os\nopen('/dev/shm/sem.aB3dE9', 'w').write('x')\nos.remove('/dev/shm/sem.aB3dE9')\n" + good, fail=False)
    add("creates and removes /dev/shm/sem.abcdefg (seven characters: not the temporary name of a semaphore)", "import os\nopen('/dev/shm/sem.abcdefg', 'w').write('x')\nos.remove('/dev/shm/sem.abcdefg')\n" + good)
    add("creates and removes /dev/shm/sem.abcde (five characters: not the temporary name of a semaphore)", "import os\nopen('/dev/shm/sem.abcde', 'w').write('x')\nos.remove('/dev/shm/sem.abcde')\n" + good)
    add("creates and removes /dev/shm/sem.abcdef followed by a newline (not a name that the C library gives)", "import os\nopen('/dev/shm/sem.abcdef\\n', 'w').write('x')\nos.remove('/dev/shm/sem.abcdef\\n')\n" + good)
    add("a repeated top-level key (the report is {'<top>': {}, '<top>': <real report>}; json.loads keeps the real one)", raw("_top = next(iter(_j.loads(_t))); assert _t.startswith('{\"%s\":' % _top); open(_f, 'w').write('{\"%s\":{},' % _top + _t[1:])\n"))
    add("a repeated key inside an entry (the first value is dropped by json.loads)", raw("_e = next(iter(next(iter(_j.loads(_t).values())).values())); _k = '\"%s\":' % next(iter(_e)); _i = _t.index(_k); open(_f, 'w').write(_t[:_i] + _k + '0,' + _t[_i:])\n"))
    return L

