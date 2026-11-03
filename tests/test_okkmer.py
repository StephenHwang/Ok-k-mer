import contextlib
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
from okkmer.io import digest
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


if __name__ == '__main__': unittest.main()
