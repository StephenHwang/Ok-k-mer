"""Progressive global/local search with one result per distinct ensemble."""
import csv
import json
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
from math import comb
from pathlib import Path
import time

import numpy as np
from ete3 import Tree

from . import __version__
from .index import load, validate_bounds
from .io import digest, new_directory, utc_now, write_json, write_table
from .scoring import Scorer, compositions, weight_denominator

FIELDS = ['ensemble_size', 'ks', 'weights', 'rho', 'gap_to_stage_best']


def candidates(ks, size, previous=None, delta=None):
    if size == 1 or delta is None:
        yield from combinations(ks, size)
        return
    neighborhoods = [[k for k in ks if abs(k-old) <= delta] for old in previous]
    unique = set()
    for varied in product(*neighborhoods):
        if len(set(varied)) != len(varied):
            continue
        for additional in ks:
            key = tuple(sorted((*varied, additional)))
            if len(set(key)) == size:
                unique.add(key)
    yield from sorted(unique)


def estimate(ks, maximum, denominator):
    return {size: comb(len(ks),size)*comb(denominator-1,size-1)
            for size in range(1,maximum+1)}


def optimize(distances, target, ks, maximum, step, delta=None, threads=1, tolerance=1e-12, grid_directory=None, stop_on_no_improvement=True):
    scorer = Scorer(distances, target)
    lookup = {k:i for i,k in enumerate(ks)}
    denominator = weight_denominator(step)
    stages, previous, winner, reason = [], None, None, 'max_kmers_reached'
    for size in range(1,maximum+1):
        coefficients = list(compositions(denominator, size))

        def evaluate(lengths):
            indices = [lookup[k] for k in lengths]
            best = None
            scores = [] if grid_directory is not None else None
            for weights in coefficients:
                rho = scorer.score(indices, weights)
                if scores is not None:
                    scores.append((list(lengths), [w/denominator for w in weights], rho))
                if rho is not None and (best is None or rho > best['rho']):
                    best = {'ensemble_size': size, 'ks': list(lengths),
                            'weights': [w/denominator for w in weights], 'rho': rho}
            return best, scores

        tasks = candidates(ks, size, previous, delta)
        rows = []
        grid_path = Path(grid_directory)/f'weights_{size}.tsv' if grid_directory is not None else None
        with (grid_path.open('w', newline='') if grid_path else nullcontext()) as handle:
            writer = csv.writer(handle,delimiter='\t') if handle else None
            if writer:
                writer.writerow(['ks','weights','rho'])
            with (ThreadPoolExecutor(max_workers=threads) if threads > 1 else nullcontext()) as pool:
                evaluated = pool.map(evaluate,tasks) if pool else map(evaluate,tasks)
                for row,scores in evaluated:
                    if writer:
                        for lengths,weights,rho in scores:
                            writer.writerow([json.dumps(lengths),json.dumps(weights),rho])
                    if row is not None:
                        rows.append(row)
        rows.sort(key=lambda row:(-row['rho'],row['ks'],row['weights']))
        if not rows:
            if winner is None:
                raise ValueError('No defined correlations; genome distances may all be constant')
            reason = 'no_defined_candidates'
            break
        best = rows[0]
        for row in rows:
            row['gap_to_stage_best'] = best['rho']-row['rho']
        stages.append({'size':size,'rows':rows,'best':best})
        improves = winner is None or best['rho'] > winner['rho']+tolerance
        if not improves and stop_on_no_improvement:
            reason = 'no_improvement'
            break
        if improves:
            winner = best
        previous = best['ks']
    return stages, winner, reason


def load_reference(path):
    tree = Tree(str(Path(path).resolve()), format=1)
    names = tree.get_leaf_names()
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError('Reference tree requires unique, nonempty leaf IDs')
    if any(not np.isfinite(n.dist) or n.dist < 0 for n in tree.traverse()):
        raise ValueError('Reference branch lengths must be finite and nonnegative')
    return tree


def save_search(out, stages, winner, reason):
    for stage in stages:
        write_table(out/f"combinations_{stage['size']}.tsv",stage['rows'],FIELDS)
    if stages:
        write_table(out/'single_k.tsv',stages[0]['rows'],FIELDS)
    write_table(out/'recommendations.tsv',[dict(stage['best'],selected=stage['best'] is winner)
                                          for stage in stages],FIELDS+['selected'])
    return {'recommendation':winner,'stop_reason':reason,
            'stages':[{'ensemble_size':s['size'],'n_combinations':len(s['rows']),'best':s['best']} for s in stages]}


def run(args):
    started, start = utc_now(), time.monotonic()
    validate_bounds(args.k_low,args.k_high)
    denominator = weight_denominator(args.w_step)
    ks = list(range(args.k_low,args.k_high+1))
    if args.max_kmers < 1 or args.max_kmers > min(len(ks),denominator):
        raise ValueError('max-kmers must be >=1 and <= both the number of k values and 1/w-step')
    if args.threads < 1 or args.local is not None and args.local < 0:
        raise ValueError('threads must be positive; local delta must be nonnegative')
    if not np.isfinite(args.improvement_tol) or args.improvement_tol < 0:
        raise ValueError('improvement-tol must be finite and nonnegative')
    manifest, distances = load(args.index)
    missing = set(ks)-set(manifest['ks'])
    if missing:
        raise ValueError(f'Index lacks requested k values: {sorted(missing)}')
    names = manifest['genomes']
    tree = load_reference(args.tree)
    leaves = set(tree.get_leaf_names())
    exclusions = set()
    if args.exclude_genomes:
        exclusions = set(Path(args.exclude_genomes).read_text().splitlines())
        exclusions.discard('')
        if not exclusions <= leaves | set(names):
            raise ValueError('Exclusion IDs are absent from both the tree and index')
    if leaves-exclusions != set(names)-exclusions:
        raise ValueError(f'Tree/index IDs differ. Tree only: {sorted(leaves-set(names)-exclusions)}; '
                         f'index only: {sorted(set(names)-leaves-exclusions)}. '
                         'Provide an explicit --exclude-genomes file for intentional exclusions.')
    indices = np.triu_indices(len(names),1)
    a,b = np.asarray(names)[indices[0]],np.asarray(names)[indices[1]]
    keep = np.asarray([x not in exclusions and y not in exclusions for x,y in zip(a,b)])
    a,b = a[keep],b[keep]
    if len(set(a)|set(b)) < 3:
        raise ValueError('Need at least three retained genomes')
    nodes = {leaf.name:leaf for leaf in tree}
    target = np.asarray([nodes[x].get_distance(nodes[y]) for x,y in zip(a,b)])
    distances = distances[keep][:,[manifest['ks'].index(k) for k in ks]]
    estimates = estimate(ks,args.max_kmers,denominator)
    print('Global candidate-weight upper bounds:', ', '.join(f'{m} lengths: {n:,}' for m,n in estimates.items()),flush=True)
    if args.dry_run:
        return
    out = new_directory(args.output)
    stages,winner,reason = optimize(distances,target,ks,args.max_kmers,args.w_step,args.local,args.threads,args.improvement_tol,out if args.save_grid else None, not args.no_early_stop)
    summary = save_search(out,stages,winner,reason)
    node_rows = []
    if args.annotate_nodes:
        for i,node in enumerate(n for n in tree.traverse('preorder') if not n.is_leaf()):
            node_id = f'node_{i:06d}'
            retained = sorted(set(node.get_leaf_names())-exclusions)
            row = {'node_id':node_id,'original_name':node.name,'genomes':retained,'n_genomes':len(retained)}
            if node is tree:
                row.update(status='ok',**summary)
            elif len(retained) < 3:
                row.update(status='too_few_genomes',recommendation=None)
            else:
                mask = np.isin(a,retained)&np.isin(b,retained)
                if np.unique(target[mask]).size < 2:
                    row.update(status='constant_tree_distances',recommendation=None)
                else:
                    try:
                        ns,nw,nr = optimize(distances[mask],target[mask],ks,args.max_kmers,args.w_step,args.local,args.threads,args.improvement_tol,stop_on_no_improvement=not args.no_early_stop)
                    except ValueError as exc:
                        if not str(exc).startswith('No defined correlations'):
                            raise
                        row.update(status='constant_genome_distances',recommendation=None)
                    else:
                        node_out=out/'nodes'/node_id
                        node_out.mkdir(parents=True)
                        row.update(status='ok',**save_search(node_out,ns,nw,nr))
            node_rows.append(row)
        write_json(out/'nodes.json',node_rows)
        write_table(out/'nodes.tsv',[{'node_id':r['node_id'],'original_name':r['original_name'],'n_genomes':r['n_genomes'],
                                     'status':r['status'],**(r['recommendation'] or {})} for r in node_rows],
                    ['node_id','original_name','n_genomes','status',*FIELDS])
    # Bundle a copy for plotting; the user's source tree is read only.
    (out/'reference.nwk').write_text(Path(args.tree).read_text())
    import scipy
    run = {'schema_version':1,'okkmer_version':__version__,'status':'complete','started_utc':started,
           'finished_utc':utc_now(),'elapsed_seconds':time.monotonic()-start,
           'numpy_version':np.__version__,'scipy_version':scipy.__version__,
           'source_sha256':{name:digest(Path(__file__).with_name(name)) for name in ('search.py','scoring.py','index.py','io.py')},
           'tree_path':str(Path(args.tree).resolve()),'tree_sha256':digest(args.tree),
           'index_path':str(Path(args.index).resolve()),'index_manifest_sha256':digest(Path(args.index)/'manifest.json'),
           'mode':'global' if args.local is None else 'local','delta_k':args.local,'ks':ks,'max_kmers':args.max_kmers,
           'early_stopping':not args.no_early_stop,'w_step':args.w_step,'save_grid':args.save_grid,'threads':args.threads,'improvement_tol':args.improvement_tol,
           'excluded_genomes':sorted(exclusions),'n_genomes':len(set(a)|set(b)), 'n_pairs':len(target),
           'candidate_weight_upper_bounds':estimates,'rank_ties':'average','score':'spearman_distance',**summary}
    write_json(out/'run.json',run)
    print(f"Recommended {winner['ks']} with weights {winner['weights']}; rho={winner['rho']:.10f} ({reason})")
