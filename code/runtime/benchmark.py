"""Prepare or run the isolated seven-method full-simulation timing benchmark.

No arguments means dry run. Production measurements require --run explicitly.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from math import ceil, log, sqrt
from pathlib import Path
import platform
import random
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from runner import atomic_json, atomic_bytes, environment
from experiment_definitions import family, BETA_CERT, GAMMA
from certified_benchmark import local_constants
from certification import certificate
from fisher_ucb import run_fisher_ucb
from certified_benchmark import _Arm
import numpy as np
from runtime import solvers as SOLVERS

METHODS = ('KL bisection', 'KL scalar Newton', 'KL vectorized Newton',
           'Fisher practical', 'Fisher matched', 'Fisher certified', 'UCB-V')
bisection, newton_scalar, newton_vector, Q_TOL = (
    SOLVERS.bisection, SOLVERS.newton_scalar, SOLVERS.newton_vector, SOLVERS.Q_TOL)


def csv_write(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def simulate(fam, T, method, Q, seed=0):
    """Shared loop; timer includes RNG, sampling, updates and regret computation."""
    rng = np.random.default_rng(seed)
    means = np.asarray(fam['means'])
    K = len(means)
    counts = np.zeros(K, int)
    empirical = np.zeros(K)
    squares = np.zeros(K)
    regret = np.zeros(T)
    gaps = means.max() - means
    forced = 0
    forced_cost = 0.
    certified = method == 'Fisher certified'
    beta = BETA_CERT if certified else sqrt(2) if method == 'Fisher matched' else 1.25
    for t in range(T):
        if certified and t < K:
            arm, is_forced = t, True
        else:
            quota = ceil(Q * log(t + 1)) if certified else 20
            under = np.flatnonzero(counts < quota)
            is_forced = bool(under.size)
            if is_forced:
                arm = int(under[0])
            else:
                L = log(t + 1)
                if method == 'KL bisection':
                    index = np.array([bisection(float(x), int(n), L) for x, n in zip(empirical, counts)])
                elif method == 'KL scalar Newton':
                    index = np.array([newton_scalar(float(x), int(n), L) for x, n in zip(empirical, counts)])
                elif method == 'KL vectorized Newton':
                    index = newton_vector(empirical, counts, L)
                elif method == 'UCB-V':
                    variance = np.maximum(squares / counts - empirical ** 2, 0.)
                    index = empirical + np.sqrt(2 * variance * L / counts) + 3 * L / counts
                else:
                    information = fam['fisher'](np.clip(empirical, *fam['K']))
                    index = empirical + beta * np.sqrt(L / (counts * information))
                arm = int(np.argmax(index))
        reward = fam['sample'](rng, means[arm])
        counts[arm] += 1
        empirical[arm] += (reward - empirical[arm]) / counts[arm]
        if method == 'UCB-V':
            squares[arm] += reward * reward
        regret[t] = gaps[arm]
        if is_forced:
            forced += 1
            forced_cost += gaps[arm]
    return dict(curve=np.cumsum(regret), forced_pulls=forced,
                forced_regret=forced_cost, counts=counts.tolist())


def hardware(cpu):
    model = 'unknown'
    info = Path('/proc/cpuinfo')
    if info.exists():
        model = next((line.split(':', 1)[1].strip() for line in info.read_text().splitlines()
                      if line.startswith('model name')), 'unknown')
    governor = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor')
    return dict(cpu_model=model, logical_cpus=os.cpu_count(), selected_cpu=cpu,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                governor=governor.read_text().strip() if governor.exists() else None,
                platform=platform.platform())


def preflight():
    """Numerical oracle and short deterministic policy checks; no production timing."""
    from scipy.optimize import brentq
    from scipy.special import xlogy, xlog1py
    divergence = SOLVERS.divergence
    def reference(p, n, L):
        if p == 1 or L == 0:
            return p
        if p == 0:
            return -np.expm1(-L/n)
        def f(q):
            if q == 1:
                return np.inf
            return float(xlogy(p, p/q) + xlog1py(1-p, (q-p)/(1-q)) - L/n)
        return brentq(f, p, 1., xtol=5e-15, rtol=1e-14)
    ps = np.unique(np.r_[0., 1e-12, 1e-6, .001, .01,
                               np.linspace(.02, .98, 25), .999, 1-1e-12, 1.])
    worst = dict.fromkeys(METHODS[:3], 0.)
    cases = 0
    for n in [1, 20, 1000, 10000000]:
        for L in [0., log(2), log(50000), 30.]:
            ref = np.array([reference(float(p), n, L) for p in ps])
            outputs = [np.array([bisection(float(p), n, L) for p in ps]),
                       np.array([newton_scalar(float(p), n, L) for p in ps]),
                       newton_vector(ps, np.full(len(ps), n), L)]
            for name, got in zip(METHODS[:3], outputs):
                error = float(np.max(np.abs(got - ref)))
                worst[name] = max(worst[name], error)
                if error > Q_TOL + 1e-13 or np.any(got < ps) or np.any(got > 1):
                    raise AssertionError((name, n, L, error))
                for p, q in zip(ps, got):
                    if q < 1 and divergence(float(p), float(q)) > L/n + 1e-10:
                        raise AssertionError('Returned index is infeasible.')
            cases += len(ps)
    # Independent scalar policy checks, including quota and index rounds.
    from verification.timing_reference import simulation as reference_simulation
    parity = []
    for K in [4, 20]:
        fam = family('bernoulli', [.5] + list(np.linspace(.2, .45, K-1)))
        _, Q = local_constants(fam, GAMMA, BETA_CERT)
        assert certificate(fam, BETA_CERT, GAMMA, Q, fam['K'])['certified']
        for method in METHODS[3:]:
            T = 3000 if K == 4 else 500
            got = simulate(fam, T, method, Q)
            reference = reference_simulation(fam, T, 0, method, Q)
            assert np.array_equal(got['curve'], reference['curve']), (K, method)
            assert got['forced_pulls'] == reference['forced_pulls']
            assert got['forced_regret'] == reference['forced_regret']
            parity.append(dict(K=K, method=method, T=T))
        if K == 4:
            got = simulate(fam, 3000, 'Fisher certified', Q)
            ref = run_fisher_ucb([_Arm(fam, m) for m in fam['means']], 3000,
                    fam['fisher'], fam['K'], beta=BETA_CERT, seed=0,
                    mode='certified', Q=Q, return_forced=True)
            assert np.array_equal(got['curve'], ref[0])
            assert got['forced_pulls'] == ref[1] and np.isclose(got['forced_regret'], ref[2])
    return dict(passed=True, numerical_states=cases, maximum_index_error=worst,
                absolute_index_tolerance=Q_TOL, short_policy_parity=parity,
                production_benchmark_executed=False)


def summarize(output, manifest, schedule):
    records = []
    for item in schedule:
        path = output / 'checkpoints' / f"K{item['K']}_r{item['repeat']}_m{item['method_index']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record['trial'] != item:
            raise ValueError('Checkpoint schedule mismatch.')
        if (not np.isfinite(record['elapsed_seconds']) or record['elapsed_seconds'] <= 0
                or not np.isfinite(record['regret']) or record['regret'] < 0
                or len(record['counts']) != item['K']
                or sum(record['counts']) != manifest['settings']['T']
                or any(n < 0 for n in record['counts'])
                or not 0 <= record['forced_pulls'] <= manifest['settings']['T']
                or not 0 <= record['forced_regret'] <= record['regret']+1e-7):
            raise ValueError('Invalid checkpoint contents.')
        records.append(record)
    if not records:
        return
    csv_write(output / 'per_trial.csv', [dict(**r['trial'], session=r['session'],
        elapsed_seconds=r['elapsed_seconds'], regret=r['regret'],
        forced_pulls=r['forced_pulls'], forced_regret=r['forced_regret'],
        counts=json.dumps(r['counts'])) for r in records])
    rows = []
    for K in manifest['settings']['arm_counts']:
        for method in METHODS:
            selected = [r for r in records if r['trial']['K'] == K and r['trial']['method'] == method]
            if not selected:
                continue
            seconds = np.array([r['elapsed_seconds'] for r in selected])
            rows.append(dict(K=K, method=method, repeats=len(selected),
                seconds_median=float(np.median(seconds)), seconds_mean=float(seconds.mean()),
                seconds_std=float(seconds.std(ddof=1)) if len(seconds)>1 else 0.,
                seconds_min=float(seconds.min()), seconds_max=float(seconds.max()),
                regret=selected[0]['regret'], forced_pulls=selected[0]['forced_pulls'],
                forced_fraction=selected[0]['forced_pulls']/manifest['settings']['T']))
            assert all(r['regret'] == selected[0]['regret'] and r['counts'] == selected[0]['counts']
                       and r['forced_pulls'] == selected[0]['forced_pulls'] for r in selected)
    csv_write(output / 'runtime.csv', rows)
    sessions = sorted(set(r['session'] for r in records))
    complete = len(records) == len(schedule)
    atomic_json(output / 'completion.json', dict(complete=complete,
        completed_trials=len(records), expected_trials=len(schedule),
        measurement_sessions=sessions, single_session=len(sessions)==1,
        ready_for_single_session_comparison=complete and len(sessions)==1,
        note='Timing variability is across repeated seed-0 runs, not across reward seeds.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Explicitly permit another measurement session.')
    parser.add_argument('--output', type=Path, default=BASE.parent/'build/runtime')
    parser.add_argument('--T', type=int, default=50000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--arm-counts', type=int, nargs='+', default=[4,20,50,100])
    parser.add_argument('--order-seed', type=int, default=20261018)
    parser.add_argument('--cpu', type=int, help='Default: first CPU in current affinity; pin the process to it.')
    args = parser.parse_args()
    if (args.repeats < 1 or len(set(args.arm_counts)) != len(args.arm_counts)
            or min(args.arm_counts) < 2 or args.T < 20*max(args.arm_counts)):
        parser.error('Need unique K>=2, T>=20*max(K), repeats>=1.')
    schedule = []
    rng = random.Random(args.order_seed)
    # Each repetition is a randomized block containing all methods and K values.
    for repeat in range(args.repeats):
        block = [dict(K=K, repeat=repeat, method=method, method_index=i)
                 for K in args.arm_counts for i, method in enumerate(METHODS)]
        rng.shuffle(block)
        schedule.extend(block)
    settings = dict(T=args.T, repeats=args.repeats, arm_counts=args.arm_counts,
        order_seed=args.order_seed, reward_seed=0, warmup=20, beta_practical=1.25,
        beta_matched=sqrt(2), beta_certified=BETA_CERT, gamma=GAMMA,
        projection=[.01,.99], means='[0.5] + linspace(0.2,0.45,K-1)',
        absolute_index_tolerance=Q_TOL, newton_initialization='Fisher plus 2c/3, Pinsker safeguard',
        fixed_state_index_benchmark=False, timer_scope='Full simulation including sampling; excludes calibration, preflight, checkpoints, plotting, untimed warmup')
    if not args.run and not args.verify_only:
        print(json.dumps(dict(mode='dry-run', production_started=False,
            settings=settings, planned_trials=len(schedule), output=str(args.output.resolve())), indent=2))
        return
    if args.verify_only:
        result = preflight()
        atomic_json(args.output/'preflight.json', result)
        print(json.dumps(result, indent=2))
        return
    if not hasattr(os, 'sched_setaffinity'):
        parser.error('Single-CPU affinity requires Linux sched_setaffinity.')
    allowed = os.sched_getaffinity(0)
    cpu = args.cpu if args.cpu is not None else min(allowed)
    if cpu not in allowed:
        parser.error('Selected CPU is outside the allowed affinity.')
    os.sched_setaffinity(0, {cpu})
    output = args.output.resolve()
    # Do not overwrite publication data or another stage's output.
    for protected in [BASE.parent/'data', BASE]:
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            parser.error(f'Output overlaps protected evidence: {protected}')
    configurations = []
    for K in args.arm_counts:
        fam = family('bernoulli', [.5]+list(np.linspace(.2,.45,K-1)))
        _, Q = local_constants(fam, GAMMA, BETA_CERT)
        configurations.append(dict(K=K, means=fam['means'], quota=Q,
            certificate=certificate(fam, BETA_CERT, GAMMA, Q, fam['K'])))
    identity = dict(settings=settings, schedule=schedule, configurations=configurations, environment=environment(),
                    hardware=hardware(cpu))
    manifest_path = output/'run_manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(k) != v for k, v in identity.items()):
            raise ValueError('Existing benchmark identity differs; choose a new output directory.')
        identity = existing
        if all((output/'checkpoints'/f"K{i['K']}_r{i['repeat']}_m{i['method_index']}.json").exists()
               for i in schedule):
            summarize(output, identity, schedule)
            print('Reusing completed timing session.', flush=True)
            return
        if not args.resume:
            raise ValueError('Existing benchmark: use a new output directory or explicit --resume.')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output without a matching manifest.')
    atomic_json(manifest_path, identity)
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    atomic_json(output/'sessions'/f'{session}.json', dict(started_utc=session,hardware=hardware(cpu)))
    for position, item in enumerate(schedule, 1):
        path = output/'checkpoints'/f"K{item['K']}_r{item['repeat']}_m{item['method_index']}.json"
        if path.exists():
            continue
        fam = family('bernoulli', [.5]+list(np.linspace(.2,.45,item['K']-1)))
        _, Q = local_constants(fam, GAMMA, BETA_CERT)
        assert certificate(fam, BETA_CERT, GAMMA, Q, fam['K'])['certified']
        # Same short untimed warmup before every trial; fresh RNG for measurement.
        simulate(fam, 20*item['K']+5, item['method'], Q)
        start = time.perf_counter()
        got = simulate(fam, args.T, item['method'], Q)
        elapsed = time.perf_counter()-start
        record = dict(trial=item, session=session, elapsed_seconds=elapsed,
            regret=float(got['curve'][-1]), counts=got['counts'],
            forced_pulls=got['forced_pulls'], forced_regret=got['forced_regret'], Q=Q)
        atomic_json(path, record)
        print(f"{position}/{len(schedule)} K={item['K']} {item['method']} r={item['repeat']} {elapsed:.3f}s", flush=True)
        summarize(output, identity, schedule)
    summarize(output, identity, schedule)


if __name__ == '__main__':
    main()
