#!/usr/bin/env python3
"""Run the public CLI using explicit manuscript dataset conventions."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--local',type=int,nargs='?',const=3)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    datasets=json.loads((ROOT/'experiments/manuscript/configs/datasets.json').read_text())['datasets']
    choices={d['dataset']:d for d in datasets}
    if args.dataset not in choices:
        p.error('Choose dataset: '+', '.join(choices))
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory must be new or empty')
    args.output.mkdir(parents=True,exist_ok=True)
    out=args.output.resolve()
    dataset=choices[args.dataset]
    (out/'dataset.json').write_text(json.dumps(dataset,indent=2)+'\n')
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    commands=[]
    index=[sys.executable,'-m','okkmer','index','--distances',str(ROOT/dataset['distances']),
           '--input-kind',dataset['input_kind'],'--output',str(out/'index')]
    if (ROOT/dataset['distances']).is_dir():
        # These explicit user-confirmed aliases apply only to this experiment.
        source=ROOT/dataset['distances']/'jaccard_k6.tsv'
        with source.open() as handle:
            columns=next(line for line in handle if line.startswith('#Sources\t')).strip().split('\t')[1:]
        mapping=['source_id\tgenome_id']
        for value in columns:
            accession=re.match(r'GC[AF]_\d+\.\d+',Path(value).name).group()
            mapping.append(value+'\t'+dataset['aliases'].get(accession,accession))
        path=out/'genome_map.tsv'
        path.write_text('\n'.join(mapping)+'\n')
        index+=['--genome-map',str(path)]
    search=[sys.executable,'-m','okkmer','search','--tree',str(ROOT/dataset['tree']),
            '--index',str(out/'index'),'--threads',str(args.threads),'--output',str(out/'search')]
    if args.local is not None:
        search+=['--local',str(args.local)]
    if args.dry_run:
        search+=['--dry-run']
    if dataset['exclusions']:
        path=out/'excluded_genomes.txt'
        path.write_text('\n'.join(dataset['exclusions'])+'\n')
        search+=['--exclude-genomes',str(path)]
    commands=[index,search]
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    for command in commands:
        subprocess.run(command,env=env,check=True)


if __name__=='__main__': main()
