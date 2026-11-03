#!/usr/bin/env python3
"""Compare legacy reference scores with the new exact-tie objective."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from okkmer.scoring import Scorer
from okkmer.io import digest, write_json

ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error('Refusing to overwrite validation report')
    results=[]
    for dataset,folder,prefix in [('Chlamydia','chlamydiales','chlamydia'),('Cyano','cyanobacteriales','cyano')]:
        cached=ROOT/f'data/{folder}/grid_search/{prefix}_merged_df.csv'
        ks=list(range(6,41))
        columns=[f'k{k}_distance' for k in ks]
        df=pd.read_csv(cached,usecols=columns+['tree_distance'])
        scorer=Scorer(1-df[columns].to_numpy(),df.tree_distance.to_numpy())
        baseline=next((ROOT/'data/stability_3k/results').glob(f'*_{dataset}_mask00_sim00.json'))
        three=json.loads(baseline.read_text())
        two=json.loads((ROOT/f'data/{folder}/grid_search/{prefix}_slurm_summary_2k.json').read_text())
        source={'dataset':dataset,'cached_csv_sha256':digest(cached),'scores':[]}
        singles=[scorer.score([i],[1]) for i in range(len(ks))]
        i=max(range(len(ks)),key=lambda j:singles[j] if singles[j] is not None else -np.inf)
        for kind,lengths,weights,reference in [
            ('single',[ks[i]],[1],-three['initial_single_k_spearman']),
            ('two',[two['k1'],two['k2']],[two['w_k1'],1-two['w_k1']],-two['min_spearman_corr']),
            ('three',three['raw_ks'],three['raw_weights'],-three['correlation'])]:
            rho=scorer.score([ks.index(k) for k in lengths],[round(w*20) for w in weights])
            row=dict(size=kind,ks=lengths,weights=weights,legacy_rho=reference,new_rho=rho,difference=rho-reference)
            source['scores'].append(row)
            print(dataset,kind,f'legacy={reference:.12f}',f'new={rho:.12f}',f'difference={rho-reference:.3g}')
        results.append(source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,results)


if __name__=='__main__': main()
