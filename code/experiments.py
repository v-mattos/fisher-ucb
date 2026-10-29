"""Execute statistical experiment stages with resumable checkpoints."""
import argparse
import os
from pathlib import Path
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
from runner import atomic_json,environment

STAGES=['constants','poisson_projection','binomial','main_benchmark',
        'poisson_certified','boundary','many_arms','horizon']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--only',choices=STAGES)
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--boundary-seeds',type=int,default=200)
    parser.add_argument('--binomial-seeds',type=int,default=100)
    parser.add_argument('--poisson-seeds',type=int,default=1000)
    parser.add_argument('--poisson-curve-seeds',type=int,default=20)
    parser.add_argument('--T',type=int,default=50000)
    parser.add_argument('--poisson-boundary-T',type=int,default=20000)
    parser.add_argument('--horizons',type=lambda s:[int(x) for x in s.split(',')],
                        default=[10000,50000,100000,200000,1000000,10000000])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint-root',dest='checkpoints',type=Path)
    parser.add_argument('--no-resume',action='store_true')
    args=parser.parse_args()
    if min(args.seeds,args.boundary_seeds,args.binomial_seeds,args.poisson_curve_seeds,
           args.T,args.poisson_boundary_T,args.poisson_seeds)<1:
        parser.error('Seed counts and horizons must be positive')
    if not args.horizons or min(args.horizons)<2:
        parser.error('Snapshot horizons must be at least 2')
    args.horizons=sorted(set(args.horizons))
    if args.quick:
        args.T=300;args.poisson_boundary_T=200
        args.seeds=args.boundary_seeds=args.binomial_seeds=args.poisson_curve_seeds=2
        args.poisson_seeds=2;args.horizons=[100,300]
    args.output=args.output.resolve()
    args.checkpoints=(args.checkpoints or args.output.parent/'checkpoints').resolve()
    metadata=dict(environment=environment(),
                  settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  publication_run=not args.quick,
                  reference='data/reference/publication_profile.json',
                  curve_storage='4096 display points; regret computed at every round')
    manifest=args.output/'run_manifest.json'
    args.output.mkdir(parents=True,exist_ok=True)
    import experiment_definitions
    atomic_json(manifest,metadata)
    for stage in [args.only] if args.only else STAGES:
        print('Running:',stage,flush=True)
        getattr(experiment_definitions,stage)(args)
    print('Statistical stages completed successfully.')


if __name__=='__main__':main()
