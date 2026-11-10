import contextlib
import gzip
import io
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from scipy.stats import rankdata, spearmanr

from okkmer.cli import main
from okkmer.index import load
from okkmer.io import digest, read_table
from okkmer.scoring import Scorer, compositions, weight_denominator
from okkmer.search import candidates, optimize


class ScoringTests(unittest.TestCase):
    def test_ties_match_scipy(self):
        distances = np.array([[0,1],[0,.5],[.5,.5],[1,0],[1,0.]])
        target = np.array([1,2,2,3,4.])
        scorer = Scorer(distances,target)
        composite = rankdata(distances[:,0])*2 + rankdata(distances[:,1])
        self.assertAlmostEqual(scorer.score([0,1],[2,1]),spearmanr(composite,target).statistic,places=14)
        # Equal mixtures have ties; they must not be broken by row order.
        expected = spearmanr(rankdata(distances[:,0])+rankdata(distances[:,1]),target).statistic
        self.assertAlmostEqual(scorer.score([0,1],[1,1]),expected,places=14)

    def test_integer_weight_grid(self):
        self.assertEqual(weight_denominator(.05),20)
        for size in range(1,5):
            weights=list(compositions(20,size))
            self.assertEqual(len(weights),math.comb(19,size-1))
            self.assertTrue(all(sum(w)==20 and min(w)>0 for w in weights))
        for step in [0,-1,.3,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): weight_denominator(step)

    def test_local_varies_every_previous_length(self):
        values=list(candidates(list(range(6,26)),3,[10,20],1))
        self.assertIn((9,19,25),values)
        self.assertNotIn((6,7,8),values)
        self.assertEqual(len(values),len(set(values)))
        self.assertEqual(list(candidates([6,7,8,9],3)),list(itertools.combinations([6,7,8,9],3)))

    def test_global_matches_independent_exhaustive_reference(self):
        rng=np.random.default_rng(42)
        distances=rng.random((12,4))
        target=rng.random(12)
        ks=[6,7,8,9]
        stages,winner,_=optimize(distances,target,ks,3,.25,threads=2)
        for stage in stages:
            size=stage['size']
            self.assertEqual(len(stage['rows']),math.comb(4,size))
            for row in stage['rows']:
                expected=max(spearmanr(sum(w*rankdata(distances[:,ks.index(k)]) for k,w in zip(row['ks'],weights)),target).statistic
                             for weights in compositions(4,size))
                self.assertAlmostEqual(row['rho'],expected,places=14)
        self.assertGreaterEqual(winner['rho'],stages[0]['best']['rho'])

    def test_early_stop_prefers_smaller_ensemble(self):
        d=np.arange(8.)
        stages,winner,reason=optimize(np.column_stack([d,d,d]),d,[6,7,8],3,.25)
        self.assertEqual(winner['ensemble_size'],1)
        self.assertEqual(len(stages),2)
        self.assertEqual(reason,'no_improvement')


    def test_no_early_stop_evaluates_triples_and_keeps_smaller_winner(self):
        d=np.arange(8.)
        stages,winner,reason=optimize(np.column_stack([d,d,d]),d,[6,7,8],3,.25,
                                      stop_on_no_improvement=False)
        self.assertEqual([s['size'] for s in stages],[1,2,3])
        self.assertEqual(winner['ensemble_size'],1)
        self.assertEqual(reason,'max_kmers_reached')
        self.assertEqual(stages[2]['rows'][0]['ks'],[6,7,8])

    def test_no_early_stop_keeps_improvement_below_default_tolerance(self):
        target=np.arange(20000,dtype=float)
        first,second=target.copy(),target.copy()
        first[:2]=first[1::-1]
        second[-2:]=second[:-3:-1]
        distances=np.column_stack([first,second])
        single=max(float(spearmanr(col,target).statistic) for col in distances.T)
        expected=float(spearmanr(rankdata(first)+rankdata(second),target).statistic)
        self.assertGreater(expected,single)
        self.assertLess(expected-single,1e-12)
        stages,winner,reason=optimize(distances,target,[6,7],2,.5,
                                      stop_on_no_improvement=False)
        self.assertEqual(len(stages),2)
        self.assertEqual(reason,'max_kmers_reached')
        self.assertEqual(winner['ensemble_size'],2)
        self.assertAlmostEqual(winner['rho'],expected,places=14)


    def test_four_length_stage_matches_scipy(self):
        rng=np.random.default_rng(51)
        distances=rng.random((15,4))
        target=rng.random(15)
        stages,_,_=optimize(distances,target,[6,7,8,9],4,.25,
                            stop_on_no_improvement=False)
        best=stages[-1]['best']
        self.assertEqual(best['ks'],[6,7,8,9])
        self.assertEqual(best['weights'],[.25,.25,.25,.25])
        composite=sum(rankdata(distances[:,i],method='average') for i in range(4))
        self.assertAlmostEqual(best['rho'],spearmanr(composite,target).statistic,places=14)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.tree=self.root/'tree.nwk'
        self.tree.write_text('((a:1,b:2):1,(c:3,d:4):2,e:6);')
        self.table=self.root/'distances.tsv'
        names=list('abcde')
        rng=np.random.default_rng(14)
        lines=['genome_1\tgenome_2\tk6\tk7\tk8']
        for a,b in itertools.combinations(names,2):
            lines.append('\t'.join([a,b,*map(str,rng.random(3))]))
        self.table.write_text('\n'.join(lines)+'\n')

    def tearDown(self): self.temp.cleanup()

    def invoke(self,*args):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return main(list(map(str,args)))

    def index(self,source=None,**kwargs):
        out=self.root/kwargs.get('name','index')
        code=self.invoke('index','--distances',source or self.table,'--k-low',6,'--k-high',8,'--output',out,
                         '--input-kind',kwargs.get('kind','distance'))
        return code,out

    def test_index_search_annotation_and_plots(self):
        self.assertEqual(self.index()[0],0)
        out=self.root/'search'
        source_hash=digest(self.tree)
        self.assertEqual(self.invoke('search','--tree',self.tree,'--index',self.root/'index','--k-low',6,'--k-high',8,
                                    '-n',3,'--w-step',.25,'--annotate-nodes','--save-grid','--output',out),0)
        result=json.loads((out/'run.json').read_text())
        self.assertEqual(result['n_pairs'],10)
        self.assertEqual(result['rank_ties'],'average')
        self.assertEqual(digest(self.tree),source_hash)
        self.assertTrue((out/'nodes.json').exists())
        self.assertEqual(len((out/'weights_2.tsv').read_text().splitlines()),10)
        if (out/'combinations_3.tsv').exists():
            self.assertEqual(self.invoke('plot','heatmap','--results',out,'--ensemble-size',3,'--fixed-k',6,'--output',self.root/'triples.pdf'),0)
        self.assertEqual(self.invoke('plot','heatmap','--results',out,'--ensemble-size',2,'--output',self.root/'heat.pdf'),0)
        self.assertEqual(self.invoke('plot','tree','--results',out,'--output',self.root/'tree.png'),0)
        self.assertGreater((self.root/'heat.pdf').stat().st_size,100)
        self.assertEqual(self.invoke('plot','heatmap','--results',out,'--ensemble-size',3,'--output',self.root/'bad.pdf'),2)

    def test_similarity_and_distance_normalize_identically(self):
        _,out=self.index()
        text=self.table.read_text().splitlines()
        converted=[text[0]]
        for line in text[1:]:
            fields=line.split('\t')
            converted.append('\t'.join(fields[:2]+[str(1-float(x)) for x in fields[2:]]))
        source=self.root/'similarity.tsv'
        source.write_text('\n'.join(converted)+'\n')
        code,other=self.index(source,name='similarity_index',kind='similarity')
        self.assertEqual(code,0)
        np.testing.assert_allclose(load(out)[1],load(other)[1],atol=1e-15)

    def test_bad_pair_coverage_and_no_overwrite(self):
        original=self.table.read_text()
        self.table.write_text(original+original.splitlines()[1]+'\n')
        self.assertEqual(self.index()[0],2)
        self.table.write_text('\n'.join(original.splitlines()[:-1])+'\n')
        self.assertEqual(self.index()[0],2)
        self.table.write_text(original)
        self.assertEqual(self.index()[0],0)
        self.assertEqual(self.index()[0],2)

    def test_explicit_exclusion_and_dry_run(self):
        self.index()
        self.tree.write_text('((a:1,b:2):1,(c:3,d:4):2,e:6,missing:1);')
        exclusion=self.root/'excluded.txt'
        exclusion.write_text('missing\n')
        base=['search','--tree',self.tree,'--index',self.root/'index','--k-low',6,'--k-high',8,'-n',1,'--output',self.root/'dry']
        self.assertEqual(self.invoke(*base),2)
        self.assertEqual(self.invoke(*base,'--exclude-genomes',exclusion,'--dry-run'),0)
        self.assertFalse((self.root/'dry').exists())
        self.assertEqual(self.invoke(*base,'--exclude-genomes',exclusion),0)
        self.assertEqual(json.loads((self.root/'dry/run.json').read_text())['n_genomes'],5)

    def test_no_early_stop_and_improvement_tol_are_mutually_exclusive(self):
        out=self.root/'conflict'
        base=['search','--tree',self.tree,'--index',self.root/'index','--output',out]
        for tolerance in ['0','1e-12']:
            for flags in [('--no-early-stop','--improvement-tol',tolerance),
                          ('--improvement-tol',tolerance,'--no-early-stop')]:
                with self.subTest(flags=flags):
                    with self.assertRaises(SystemExit) as caught:
                        self.invoke(*base,*flags)
                    self.assertEqual(caught.exception.code,2)
                    self.assertFalse(out.exists())

    def test_no_early_stop_selects_best_saved_ensemble_and_node_recommendations(self):
        self.assertEqual(self.index()[0],0)
        self.tree.write_text('(((a:1,b:2):1,c:3):1,d:4,e:6);')
        out=self.root/'complete_search'
        self.assertEqual(self.invoke('search','--tree',self.tree,'--index',self.root/'index',
                                    '--k-low',6,'--k-high',8,'-n',3,'--w-step',.25,
                                    '--no-early-stop','--annotate-nodes','--output',out),0)
        result=json.loads((out/'run.json').read_text())
        self.assertFalse(result['early_stopping'])
        self.assertIsNone(result['improvement_tol'])
        self.assertEqual([s['ensemble_size'] for s in result['stages']],[1,2,3])
        evaluated=[row for size in [1,2,3] for row in read_table(out/f'combinations_{size}.tsv')]
        self.assertEqual(result['recommendation']['rho'],max(float(row['rho']) for row in evaluated))
        selected=[row for row in read_table(out/'recommendations.tsv') if row['selected']=='True']
        self.assertEqual(len(selected),1)
        self.assertEqual(float(selected[0]['rho']),result['recommendation']['rho'])
        nodes=json.loads((out/'nodes.json').read_text())
        self.assertTrue(any(node['status']=='ok' and node['n_genomes']==3 for node in nodes))
        for node in nodes:
            if node['status']=='ok':
                self.assertEqual(len(node['stages']),3)
                best=max((stage['best'] for stage in node['stages']),key=lambda row:row['rho'])
                self.assertEqual(node['recommendation'],best)

    def test_missing_ks_and_index_checksum(self):
        _,out=self.index()
        self.assertEqual(self.invoke('search','--tree',self.tree,'--index',out,'--output',self.root/'bad'),2)
        with (out/'distances.npz').open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'): load(out)

    def test_square_import(self):
        raw=self.root/'raw';raw.mkdir()
        for k in range(6,9):
            (raw/f'jaccard_k{k}.tsv').write_text('#Sources\tc\ta\tb\nc\t0\t0.3\t0.2\na\t0.3\t0\t0.1\nb\t0.2\t0.1\t0\n')
        code,out=self.index(raw)
        self.assertEqual(code,0)
        manifest,values=load(out)
        self.assertEqual(manifest['genomes'],['a','b','c'])
        np.testing.assert_allclose(values[:,0],[.1,.3,.2])

    def test_dashing_wrapper_with_fake_executable(self):
        # Exercise subprocess arguments and matrix normalization without Dashing2.
        fake=self.root/'dashing2'
        fake.write_text('''#!/usr/bin/env python3
import sys
if '--version' in sys.argv:
    print('Dashing2 version 2.1.21');sys.exit(1)
assert '--set' in sys.argv and '--long-kmers' in sys.argv and '--square' in sys.argv
names=open(sys.argv[sys.argv.index('-F')+1]).read().splitlines()
print('#Sources\\t'+'\\t'.join(names))
for i,name in enumerate(names):
    print(name+'\\t'+'\\t'.join('1' if i==j else '.25' for j in range(len(names))))
''')
        fake.chmod(0o755)
        genomes=self.root/'genomes.tsv'
        lines=['genome_id\tpath']
        for name in 'abc':
            genome=self.root/f'{name}.fa';genome.write_text('>seq\nACGT\n')
            lines.append(f'{name}\t{genome.name}')
        genomes.write_text('\n'.join(lines)+'\n')
        out=self.root/'computed'
        self.assertEqual(self.invoke('index','--genomes',genomes,'--dashing2',fake,'--k-low',6,'--k-high',6,'--output',out),0)
        np.testing.assert_allclose(load(out)[1],.75)
        self.assertFalse((out/'dashing2/work_k6').exists())

    def test_dashing_duplicate_basenames_preserve_distances_and_provenance(self):
        # Model Dashing's basename-based cache with actual canonical k-mer sets.
        fake=self.root/'dashing2'
        fake.write_text('''#!/usr/bin/env python3
import gzip
import json
from pathlib import Path
import sys
if '--version' in sys.argv:
    print('Dashing2 version 2.1.21');sys.exit(1)
paths=open(sys.argv[sys.argv.index('-F')+1]).read().splitlines()
cache=Path(sys.argv[sys.argv.index('--outprefix')+1])
k=int(sys.argv[sys.argv.index('-k')+1])
complement=str.maketrans('ACGT','TGCA')
for source in paths:
    opener=gzip.open if source.endswith('.gz') else open
    with opener(source,'rt') as handle:
        seq=''.join(line.strip() for line in handle if not line.startswith('>'))
    words=set()
    for i in range(len(seq)-k+1):
        word=seq[i:i+k]
        words.add(min(word,word.translate(complement)[::-1]))
    (cache/Path(source).name).write_text(json.dumps(sorted(words)))
sets=[set(json.loads((cache/Path(source).name).read_text())) for source in paths]
print('#Sources\\t'+'\\t'.join(paths))
for source,a in zip(paths,sets):
    print(source+'\\t'+'\\t'.join(str(len(a&b)/len(a|b)) for b in sets))
''')
        fake.chmod(0o755)
        # These IDs also collide if naively sanitized into filenames.
        names=['a/b','a?b','c name']
        originals={}
        lines=['genome_id\tpath']
        for i,(name,seq) in enumerate(zip(names,['AAAAAC','AAAACC','CCCCCC'])):
            folder=self.root/f'input_{i}';folder.mkdir()
            source=folder/('genome.fna.gz' if i==2 else 'genome.fna')
            content=f'>seq\n{seq}\n'.encode()
            source.write_bytes(gzip.compress(content,mtime=0) if i==2 else content)
            originals[name]=source
            lines.append(f'{name}\t{source.relative_to(self.root)}')
        hashes={name:digest(source) for name,source in originals.items()}
        manifest=self.root/'genomes.tsv';manifest.write_text('\n'.join(lines)+'\n')
        out=self.root/'computed'
        self.assertEqual(self.invoke('index','--genomes',manifest,'--dashing2',fake,
                                    '--k-low',3,'--k-high',4,'--output',out),0)
        metadata,distances=load(out)
        self.assertEqual(metadata['genomes'],names)
        np.testing.assert_allclose(distances,[[1/3,1/3],[1,1],[1,1]])
        for row in metadata['provenance']['genomes']:
            name=row['genome_id']
            self.assertEqual(row['path'],str(originals[name].resolve()))
            self.assertEqual(row['sha256'],hashes[name])
            staged=Path(row['dashing2_path'])
            self.assertTrue(staged.is_relative_to(out.resolve()))
            self.assertEqual(staged.read_bytes(),originals[name].read_bytes())
        self.assertEqual({name:digest(source) for name,source in originals.items()},hashes)
        for k in [3,4]:self.assertFalse((out/f'dashing2/work_k{k}').exists())


if __name__ == '__main__': unittest.main()
