"""Read and independently validate publication evidence without running policies."""
from collections import defaultdict
import csv
import json
import re
from pathlib import Path
import numpy as np


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def stats(values):
    x = np.asarray(values, dtype=float)
    require(len(x) > 0 and np.isfinite(x).all(), 'Empty or nonfinite observations')
    return dict(seeds=len(x), mean=float(x.mean()),
                ci95=float(1.96*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.,
                median=float(np.median(x)), p95=float(np.quantile(x,.95)),
                p99=float(np.quantile(x,.99)), max=float(x.max()))


def policy_label(config):
    return config['policy']+(f" beta={config['beta']:.12g}" if config.get('beta') else '')+(
        f" Q={config['Q']:.12g}" if config.get('Q') else '')


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.statistics = self.root/'statistics'
        self.results = self.statistics/'output/results'

    def rows(self, group):
        return read_csv(self.results/f'{group}_per_seed.csv')

    def metadata(self, group):
        return read_json(self.results/f'{group}_metadata.json')

    def configuration(self, group, policy):
        matches = [c for c in self.metadata(group)['configurations']
                   if (policy_label(c)==policy if ' beta=' in policy else c['policy']==policy)]
        require(len(matches)==1, f'Expected one configuration: {group}/{policy}')
        return matches[0]

    def observations(self, group, policy):
        rows = [r for r in self.rows(group)
                if (r['policy']==policy if ' beta=' in policy else r['policy'].split(' beta=')[0]==policy)]
        return sorted(rows, key=lambda r:int(r['seed']))

    def summary(self, group, policy):
        return stats([float(r['regret']) for r in self.observations(group,policy)])

    def records(self, group, policy):
        meta = self.metadata(group)
        config = self.configuration(group,policy)
        candidates = []
        for path in (self.statistics/'checkpoints'/group).glob('*/manifest.json'):
            identity = read_json(path)
            if all(
                    identity['configuration'].get(k)==v for k,v in config.items()):
                candidates.append(path.parent)
        require(len(candidates)==1, f'Ambiguous/missing checkpoint identity: {group}/{policy}')
        folder = candidates[0]
        expected = {f'seed_{i}.npz' for i in range(meta['seeds'])}
        require({p.name for p in folder.glob('seed_*.npz')}==expected,
                f'Incomplete/extra checkpoint seeds: {group}/{policy}')
        records=[]
        for i in range(meta['seeds']):
            with np.load(folder/f'seed_{i}.npz', allow_pickle=False) as z:
                records.append({k:z[k].copy() for k in z.files})
        return records

    def validate(self):
        groups=policies=seeds=checkpoints=0
        for path in sorted(self.results.glob('*_per_seed.csv')):
            group=path.name.removesuffix('_per_seed.csv')
            meta=self.metadata(group)
            rows=read_csv(path)
            summaries=read_csv(self.results/f'{group}.csv')
            require(len({r['policy'] for r in summaries})==len(summaries), f'Duplicate summaries: {group}')
            configurations=meta['configurations']
            require(len({policy_label(c) for c in configurations})==len(configurations), f'Duplicate configurations: {group}')
            expected={policy_label(c) for c in configurations}
            require({r['policy'] for r in rows}==expected, f'Missing/extra policies: {group}')
            require({r['policy'] for r in rows}=={r['policy'] for r in summaries}, f'Summary policy mismatch: {group}')
            for c in configurations:
                policy=policy_label(c); observed=self.observations(group,policy)
                require([int(r['seed']) for r in observed]==list(range(meta['seeds'])), f'Seed coverage: {group}/{policy}')
                require(len({r['policy'] for r in observed})==1, f'Multiple policy labels: {group}/{policy}')
                label=observed[0]['policy']; saved=next(r for r in summaries if r['policy']==label)
                computed=stats([float(r['regret']) for r in observed])
                for key,value in computed.items():
                    if key in saved:
                        require(np.isclose(float(saved[key]),value,rtol=1e-12,atol=1e-8), f'Summary mismatch: {group}/{label}/{key}')
                records=self.records(group,policy)
                for r,z in zip(observed,records):
                    for field,key in [('regret','final'),('forced_pulls','forced_pulls'),('forced_regret','forced_regret')]:
                        require(np.isclose(float(r[field]),float(z[key]),rtol=0,atol=1e-8), f'CSV/checkpoint mismatch: {group}/{label}/{r["seed"]}/{field}')
                    for key,value in z.items():
                        require(np.isfinite(value).all(), f'Nonfinite checkpoint field: {group}/{key}')
                    if 'curve' in z:
                        require(len(z['curve'])==len(z['rounds']) and np.all(np.diff(z['rounds'])>0), f'Invalid grid: {group}')
                        require(np.all(np.diff(z['curve'])>=-1e-8) and int(z['rounds'][-1])==c['T'], f'Invalid curve: {group}')
                        require(np.isclose(float(z['curve'][-1]),float(z['final']),atol=1e-8,rtol=0), f'Curve endpoint: {group}')
                    checkpoints+=1
                policies+=1;seeds+=len(observed)
            groups+=1
        require(groups>=16, 'Incomplete publication evidence: required statistical groups missing')
        required=['main_bernoulli','main_exponential','main_poisson','poisson_replication',
                  'binomial_favourable','binomial_adverse','boundary_nearzero_paper',
                  'boundary_nearzero_wide','boundary_ctr_paper','boundary_ctr_wide',
                  'poisson_curves','poisson_boundary_projected','poisson_boundary_minimal_clip',
                  'horizon','many_arms_bernoulli','many_arms_exponential']
        for group in required:
            require((self.results/f'{group}_per_seed.csv').is_file(), f'Missing required group: {group}')
        extension=self.validate_extension()
        timing=self.validate_timing()
        return dict(passed=True, statistical_groups=groups, policy_groups=policies,
                    per_seed_records=seeds, checkpoints=checkpoints,
                    poisson_certified=extension, runtime=timing)

    def validate_extension(self):
        root=self.root/'poisson_certified';results=root/'results'
        rows=read_csv(results/'poisson_certified_replication_per_seed.csv')
        meta=read_json(results/'poisson_certified_replication_metadata.json')
        require([int(r['seed']) for r in rows]==list(range(meta['seeds'])), 'Certified Poisson coverage')
        require(len(set(r['policy'] for r in rows))==1, 'Certified Poisson policy coverage')
        calculated=stats([float(r['regret']) for r in rows])
        summary=read_csv(results/'poisson_certified_replication.csv')
        require(len(summary)==1, 'Certified Poisson summary rows')
        for k,v in calculated.items():
            require(np.isclose(float(summary[0][k]),v,rtol=1e-12,atol=1e-8), f'Certified Poisson summary: {k}')
        # The first 20 seeds overlap the main Poisson experiment.
        old=self.observations('main_poisson','Fisher certified')
        config=meta['configurations'][0]
        checkpoint_root=(self.root/'poisson_certified_checkpoints'
                         if (self.root/'poisson_certified_checkpoints').exists() else root/'checkpoints')
        candidates=[]
        for p in checkpoint_root.glob('poisson_certified_replication/*/manifest.json'):
            identity=read_json(p)
            if all(
                    identity['configuration'].get(k)==v for k,v in config.items()):
                candidates.append(p.parent)
        require(len(candidates)==1,'Certified Poisson checkpoint identity')
        folder=candidates[0]
        require({p.name for p in folder.glob('seed_*.npz')}=={f'seed_{i}.npz' for i in range(meta['seeds'])},
                'Certified Poisson checkpoint coverage')
        for row in rows:
            with np.load(folder/f"seed_{row['seed']}.npz",allow_pickle=False) as record:
                for field,key in [('regret','final'),('forced_pulls','forced_pulls'),('forced_regret','forced_regret')]:
                    require(np.isclose(float(row[field]),float(record[key]),rtol=0,atol=1e-8),'Certified Poisson checkpoint/CSV mismatch')
        if config['T']==self.configuration('main_poisson','Fisher certified')['T']:
            for r in old[:len(rows)]:
                for key in ['regret','forced_pulls','forced_regret']:
                    require(np.isclose(float(rows[int(r['seed'])][key]),float(r[key]),rtol=0,atol=1e-8), f'Certified Poisson overlapping seed mismatch: {key}')
        return calculated

    def validate_timing(self):
        root=self.root/'runtime';trials=read_csv(root/'per_trial.csv');summary=read_csv(root/'runtime.csv')
        manifest=read_json(root/'run_manifest.json');settings=manifest['settings']
        grouped=defaultdict(list)
        require(len({(r['K'],r['method'],r['repeat']) for r in trials})==len(trials), 'Duplicate timing trials')
        for row in trials:grouped[(int(row['K']),row['method'])].append(row)
        methods={r['method'] for r in summary}
        require(len(methods)==7, 'Expected seven timing methods')
        require(set(grouped)=={(int(k),m) for k in settings['arm_counts'] for m in methods}, 'Missing timing methods/arms')
        require(len(summary)==len(grouped), 'Timing summary group count')
        for row in summary:
            rs=grouped[(int(row['K']),row['method'])]
            require(sorted(int(r['repeat']) for r in rs)==list(range(settings['repeats'])), 'Timing repetition coverage')
            require(np.isclose(np.median([float(r['elapsed_seconds']) for r in rs]),float(row['seconds_median']),rtol=1e-12,atol=1e-12), 'Timing median mismatch')
            for r in rs:
                require(sum(json.loads(r['counts']))==settings['T'], 'Timing pull accounting')
        require(len({r['session'] for r in trials})==1, 'Timing evidence must belong to one session')
        require(settings['absolute_index_tolerance']==1e-6, 'Wrong selected timing accuracy')
        return dict(trials=len(trials), groups=len(grouped), sessions=1,
                    absolute_index_tolerance=1e-6)

    def validate_profile(self, profile):
        """Check the complete approved experiment grid, not just self-reported metadata."""
        actual={p.name.removesuffix('_per_seed.csv') for p in self.results.glob('*_per_seed.csv')}
        require(actual==set(profile['groups']), 'Missing/extra statistical experiments against publication profile')
        for group,expected in profile['groups'].items():
            meta=self.metadata(group)
            require(meta['seeds']==expected['seeds'], f'Wrong publication seed count: {group}')
            def sort(cs):
                normalized=[]
                for config in cs:
                    config=dict(config)
                    config['rng']=re.sub(r'^v\d+\s+','',config.get('rng',''))
                    normalized.append(config)
                return sorted(normalized,key=lambda c:json.dumps(c,sort_keys=True))
            require(sort(meta['configurations'])==sort(expected['configurations']),f'Publication configuration mismatch: {group}')
        extension=read_json(self.root/'poisson_certified/results/poisson_certified_replication_metadata.json')
        require(extension['seeds']==profile['poisson_certified']['seeds'] and sort(extension['configurations'])==sort(profile['poisson_certified']['configurations']),
                'Certified Poisson publication profile mismatch')
        settings=read_json(self.root/'runtime/run_manifest.json')['settings']
        require(settings==profile['runtime_settings'], 'Selected runtime publication protocol mismatch')
        return dict(passed=True,groups=len(actual),scope='Publication configurations and seed counts')
