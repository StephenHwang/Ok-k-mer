import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from matplotlib.offsetbox import AnnotationBbox
from okkmer.cli import main
from okkmer.plotting import _tree_figure, pyplot
from ete3 import Tree


class TreePlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tree = Tree('(((a:1,b:1):1,c:2):1,(d:1,e:1):2,f:3);',format=1)
        self.metadata = {'ks':list(range(6,11)), 'excluded_genomes':[]}
        self.rows = [dict(node_id=f'node_{i:06d}',recommendation=dict(ks=ks,weights=weights,rho=rho))
                     for i,(ks,weights,rho) in enumerate([
                         ([6],[1.],.9),([6,8],[.25,.75],.8),
                         ([6,8,10],[.2,.3,.5],-.7),([10],[1.],.6)])]
        (self.root/'reference.nwk').write_text(self.tree.write(format=1))
        (self.root/'run.json').write_text(json.dumps(self.metadata))
        (self.root/'nodes.json').write_text(json.dumps(self.rows))

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self,*args):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return main(['plot','tree','--results',str(self.root),*map(str,args)])

    def test_pie_weights_labels_and_default_threshold(self):
        plt = pyplot()
        annotations = {row['node_id']:row for row in self.rows}
        fig = _tree_figure(self.tree,annotations,self.metadata,plt,min_leaves=2)
        ax = fig.axes[0]
        pies = [a for a in ax.artists if isinstance(a,AnnotationBbox)]
        self.assertEqual(len(pies),4)
        for pie,row in zip(pies,self.rows):
            wedges = pie.offsetbox.get_children()
            self.assertEqual(len(wedges),len(row['recommendation']['ks']))
            for wedge,weight in zip(wedges,row['recommendation']['weights']):
                self.assertAlmostEqual(wedge.theta2-wedge.theta1,360*weight)
        labels = [t.get_text() for t in ax.texts]
        self.assertIn('k=6,8,10\nw=0.2,0.3,0.5\nρ=-0.700',labels)
        self.assertEqual(fig.axes[1].get_xlabel(),'k-mer length')
        branch_segments = [(tuple(line.get_xdata()),tuple(line.get_ydata())) for line in ax.lines]
        plt.close(fig)
        for threshold,expected_pies in [(5,1),(100,0)]:
            fig = _tree_figure(self.tree,annotations,self.metadata,plt,min_leaves=threshold)
            ax = fig.axes[0]
            self.assertEqual(sum(isinstance(a,AnnotationBbox) for a in ax.artists),expected_pies)
            self.assertEqual({t.get_text() for t in ax.texts if not t.get_text().startswith('k=')},
                             set(self.tree.get_leaf_names()))
            # Annotation padding may change x positions, but every internal
            # and terminal branch must retain its original vertical endpoints.
            self.assertEqual([tuple(line.get_ydata()) for line in ax.lines],
                             [ys for xs,ys in branch_segments])
            plt.close(fig)

    def test_exclusions_preserve_node_mapping_and_leaf_order(self):
        plt = pyplot()
        self.metadata['excluded_genomes'] = ['a','b']
        fig = _tree_figure(self.tree,{r['node_id']:r for r in self.rows},self.metadata,plt,2)
        labels = [t.get_text() for t in fig.axes[0].texts]
        self.assertNotIn('a',labels)
        self.assertNotIn('b',labels)
        self.assertIn('k=10\nw=1\nρ=0.600',labels)
        self.assertNotIn('k=6,8\nw=0.25,0.75\nρ=0.800',labels)
        leaf_labels = [t for t in fig.axes[0].texts if t.get_text() in 'cdef']
        self.assertEqual([t.get_text() for t in sorted(leaf_labels,key=lambda t:-t.xy[1])],list('cdef'))
        plt.close(fig)

    def test_paired_exports_and_overwrite_preflight(self):
        output = self.root/'annotated'
        self.assertEqual(self.invoke('--output',output,'--dpi',60),0)
        for suffix,magic in [('.pdf',b'%PDF'),('.png',b'\x89PNG')]:
            path = output.with_suffix(suffix)
            self.assertTrue(path.read_bytes().startswith(magic))
        output.with_suffix('.pdf').unlink()
        self.assertEqual(self.invoke('--output',output,'--dpi',60),2)
        self.assertFalse(output.with_suffix('.pdf').exists())
        self.assertEqual(self.invoke('--output',self.root/'one.svg','--min-leaves',2),0)
        self.assertEqual(self.invoke('--output',self.root/'bad.png','--dpi',0),2)
        self.assertEqual(self.invoke('--output',self.root/'bad.png','--min-leaves',0),2)


if __name__ == '__main__':
    unittest.main()
