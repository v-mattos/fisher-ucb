#!/usr/bin/env python3
"""One entry point for all publication experiments, four tables and six figures.

Default: execute the complete experiment profile in a new output tree.
--from-data: validate and rebuild the article from the published reference data.
--smoke: run every stage with reduced horizons and seeds, including both extensions.
--check: rebuild from reference data and replay selected actual-horizon seeds.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/'code'
REFERENCE=ROOT/'data/reference'


def ensure_reference(refresh=False):
    if REFERENCE.is_dir() and not refresh:return
    archive=ROOT/'data/reference.zip'
    if not archive.is_file():raise FileNotFoundError('Reference data missing: expected data/reference.zip')
    with tempfile.TemporaryDirectory(prefix='reference-extract-',dir=ROOT/'data') as temporary:
        stage=Path(temporary)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if not (stage/member.filename).resolve().is_relative_to(stage.resolve()):
                    raise ValueError('Unsafe reference archive member')
            handle.extractall(stage)
        incoming=stage/'reference'
        if not incoming.is_dir():raise ValueError('Reference archive must contain reference/')
        if REFERENCE.exists():shutil.rmtree(REFERENCE)
        incoming.rename(REFERENCE)


def protect(destination, source):
    destination=destination.resolve();source=source.resolve()
    if destination==source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Output must not overlap reference data')
    if destination==ROOT or not destination.is_relative_to(ROOT/'build'):
        # External destinations are allowed, but never repository source trees.
        if destination.is_relative_to(ROOT):
            raise ValueError('Use build/ or an external output directory')


def command(args, env):
    print('+', ' '.join(str(x) for x in args),flush=True)
    subprocess.run([str(x) for x in args],cwd=ROOT,env=env,check=True)


def build(data,output,smoke=False):
    from evidence import Evidence
    from reporting import tables,figures
    evidence=Evidence(data)
    validation=evidence.validate()
    values=tables(evidence,output/'tables');figures(evidence,output/'figures')
    if not smoke:
        profile=json.loads((REFERENCE/'publication_profile.json').read_text())
        validation['publication_profile']=evidence.validate_profile(profile)
        expected=json.loads((REFERENCE/'article_expectations.json').read_text())['values']
        fields=0
        for key,record in expected.items():
            for field,value in record.items():
                if round(values[key][field],1)!=value:
                    raise ValueError(f'Article regression: {key}/{field}: {values[key][field]} vs {value}')
                fields+=1
        validation['article_table_regression']=dict(passed=True,rounded_numeric_fields=fields)
    output.mkdir(parents=True,exist_ok=True)
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--from-data',action='store_true')
    modes.add_argument('--smoke',action='store_true')
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--verify-only',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-root',type=Path,default=REFERENCE)
    parser.add_argument('--refresh-data',action='store_true',
                        help='Replace data/reference/ with a fresh extraction of data/reference.zip')
    parser.add_argument('--replay-seeds',type=int,default=1)
    parser.add_argument('--cpu',type=int)
    args=parser.parse_args()
    if args.replay_seeds<1:parser.error('Replay seeds must be positive')
    output=(args.output or ROOT/'build'/('verification' if args.verify_only else 'smoke' if args.smoke else 'paper' if args.from_data or args.check else 'full')).resolve()
    protect(output,args.data_root)
    output.mkdir(parents=True,exist_ok=True)
    (output/'pipeline_manifest.json').unlink(missing_ok=True)
    if args.refresh_data or not args.verify_only:ensure_reference(refresh=args.refresh_data)
    env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg',
        'MPLCONFIGDIR':str(output/'matplotlib'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    # Also control analysis imports in this process, not only child simulations.
    os.environ.update({k:env[k] for k in ['MPLBACKEND','MPLCONFIGDIR','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    python=[sys.executable,'-B']
    if args.verify_only or args.check:
        command([*python,ENGINE/'verification/verify.py','--output',output/'verification/numerical.json'],env)
        command([*python,ENGINE/'runtime/benchmark.py','--verify-only','--output',output/'verification/runtime'],env)
        if args.verify_only:return
    if args.from_data or args.check:
        if args.check:
            command([*python,ENGINE/'replay.py','--reference',args.data_root,'--output',output/'replay.json','--seeds',args.replay_seeds],env)
        validation=build(args.data_root,output)
        mode='reference-replay' if args.check else 'reference-analysis'
    else:
        data=output/'data'
        command([*python,ENGINE/'experiments.py','--output',data/'statistics/output',
                 '--checkpoint-root',data/'statistics/checkpoints',*(['--quick'] if args.smoke else [])],env)
        runtime=[*python,ENGINE/'runtime/benchmark.py','--run','--output',data/'runtime']
        if args.smoke:runtime+=['--T','300','--repeats','2','--arm-counts','4']
        if args.cpu is not None:runtime+=['--cpu',args.cpu]
        command(runtime,env)
        validation=build(data,output,smoke=args.smoke)
        mode='smoke' if args.smoke else 'full-execution'
    manifest=dict(mode=mode,data_root=str(args.data_root if args.from_data or args.check else output/'data'),
                  output=str(output),validation=validation,
                  statistical_simulations_run=not(args.from_data),
                  timing_note='Published timing session when reading supplied data; new timings depend on hardware and session.')
    (output/'pipeline_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(passed=True,mode=mode,output=str(output),tables=4,figures=6),indent=2))


if __name__=='__main__':main()
