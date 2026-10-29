"""Rerun saved configurations with the current policies and compare actual trajectories."""
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'code'))
from experiment_definitions import family, simulate
from evidence import Evidence, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=1)
    args=parser.parse_args()
    if args.seeds<1:parser.error('Seeds must be positive')
    e=Evidence(args.reference);checks=[]
    groups=['main_bernoulli','main_exponential','main_poisson',
            'binomial_favourable','binomial_adverse','boundary_nearzero_paper',
            'boundary_nearzero_wide','boundary_ctr_paper','boundary_ctr_wide']
    for group in groups:
        for c in e.metadata(group)['configurations']:
            policy=c['policy'];records=e.records(group,policy)
            fam=family(c['family'],c['means'],c['projection'])
            for seed in range(min(args.seeds,len(records))):
                result=simulate(fam,policy,c['T'],seed,c['beta'] or 1.25,c['Q'])
                curve=result['curve'] if isinstance(result,dict) else result[0] if isinstance(result,tuple) else result
                old=records[seed];difference=float(abs(curve[-1]-float(old['final'])))
                require(difference<=1e-8,f'Replayed regret differs: {group}/{policy}/{seed}: {difference}')
                if 'rounds' in old:
                    require(np.allclose(curve[old['rounds'].astype(int)-1],old['curve'],rtol=0,atol=1e-8),f'Replayed curve differs: {group}/{policy}/{seed}')
                if isinstance(result,tuple):
                    require(int(result[1])==int(old['forced_pulls']) and np.isclose(result[2],float(old['forced_regret']),rtol=0,atol=1e-8), f'Replayed forced accounting: {group}/{policy}')
                row=dict(group=group,policy=policy,seed=seed,T=c['T'],absolute_final_error=difference)
                checks.append(row);print(json.dumps(row),flush=True)
    data=dict(passed=True,
              scope='Selected seeds at actual saved horizons; not a rerun of all seeds or long-horizon experiments',checks=checks)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
