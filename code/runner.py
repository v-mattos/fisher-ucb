"""Atomic checkpoints identified by experiment configuration."""
from pathlib import Path
import csv
import json
import os
import platform
import tempfile
import time
import numpy as np

SCHEMA = 1


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), machine=platform.machine(),
                dependencies={k: __import__('importlib.metadata',fromlist=['version']).version(k)
                              for k in ['scipy','sympy','matplotlib']},
                threads={k: os.environ.get(k) for k in
                         ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})


def atomic_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def atomic_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name); f.write(data)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv(path, rows):
    from io import StringIO
    if not rows: raise ValueError('Cannot write an empty result table')
    out = StringIO(); writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_bytes(path, out.getvalue().encode())


def checkpoint_directory(root, spec):
    parent = Path(root)/spec['experiment']
    matches = [p.parent for p in sorted(parent.glob('*/manifest.json'))
               if json.loads(p.read_text())['configuration'] == spec]
    if len(matches) > 1:
        raise ValueError('Multiple checkpoint folders have the same configuration')
    if matches:
        return matches[0]
    index = 1
    while (parent/f'config_{index:03d}').exists():
        index += 1
    directory = parent/f'config_{index:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory/'manifest.json', dict(schema=SCHEMA, configuration=spec))
    return directory


def save_record(path, record):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
        tmp = Path(f.name)
    try:
        np.savez_compressed(tmp, version=SCHEMA, **record)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_record(path):
    with np.load(path, allow_pickle=False) as z:
        if int(z['version']) != SCHEMA: raise ValueError('Checkpoint schema mismatch')
        record = {k: z[k].copy() for k in z.files if k != 'version'}
    if any(not np.isfinite(v).all() for v in record.values()):
        raise ValueError(f'Non-finite checkpoint: {path}')
    if record['elapsed'] < 0 or record['final'] < 0: raise ValueError('Invalid checkpoint')
    if 'curve' in record:
        if np.any(np.diff(record['curve']) < -1e-8): raise ValueError('Non-monotone regret')
        if not np.isclose(record['final'], record['curve'][-1]): raise ValueError('Regret mismatch')
    return record


def compact_result(result, T, horizons=(), keep_curve=True):
    if isinstance(result, dict):
        curve = np.asarray(result['curve'])
        forced = result.get('forced_pulls',0); cost = result.get('forced_regret',0.)
        states = result.get('states', {})
    elif isinstance(result, tuple):
        curve, forced, cost = result; states = {}
    else:
        curve = np.asarray(result); forced, cost, states = 0, 0., {}
    if curve.shape != (T,) or not np.isfinite(curve).all(): raise ValueError('Invalid trajectory')
    record = dict(final=np.array(float(curve[-1])), forced_pulls=np.array(forced),
                  forced_regret=np.array(cost))
    if keep_curve:
        grid = np.unique(np.r_[np.linspace(1,T,min(T,4096)).astype(int), horizons,T]).astype(int)
        record.update(rounds=grid, curve=curve[grid-1])
    if horizons:
        record.update(horizons=np.array(horizons), regret_at_horizons=curve[np.array(horizons)-1])
        if states:
            record.update(counts=np.array([states[h][0] for h in horizons]),
                          means=np.array([states[h][1] for h in horizons]))
    return record


def run_job(root, spec, seeds, execute, resume=True, keep_curve=True, horizons=()):
    stored_spec = dict(spec, storage=dict(keep_curve=bool(keep_curve),
                                        snapshot_horizons=list(horizons),
                                        display_points=4096))
    folder = checkpoint_directory(root, stored_spec)
    records = []
    for seed in range(seeds):
        path = folder/f'seed_{seed}.npz'
        if resume and path.exists():
            record = load_record(path)
        else:
            start = time.perf_counter(); result = execute(seed)
            elapsed = time.perf_counter()-start
            record = compact_result(result, spec['T'], horizons, keep_curve)
            record['elapsed'] = np.array(elapsed)
            save_record(path, record)
        records.append(record)
        if seed == 0 or (seed+1)%10 == 0 or seed+1 == seeds:
            print(f"{spec['experiment']}: {spec['policy']} {seed+1}/{seeds}", flush=True)
    return records


def summarize(records):
    x = np.array([r['final'] for r in records], float)
    n = len(x)
    return dict(seeds=n, mean=float(x.mean()),
                ci95=float(1.96*x.std(ddof=1)/np.sqrt(n)) if n>1 else 0.,
                median=float(np.median(x)), p95=float(np.quantile(x,.95)),
                p99=float(np.quantile(x,.99)), max=float(x.max()),
                forced_pulls=float(np.mean([r['forced_pulls'] for r in records])),
                forced_regret=float(np.mean([r['forced_regret'] for r in records])),
                time_mean=float(np.mean([r['elapsed'] for r in records])))
