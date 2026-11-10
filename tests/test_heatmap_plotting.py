import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.patches import Circle

from okkmer.cli import main, parser
from okkmer.plotting import _heatmap_data, _heatmap_figure, pyplot
from okkmer.io import write_table


def row(ks, weights, rho):
    return {'ks':json.dumps(ks),'weights':json.dumps(weights),'rho':str(rho)}


class HeatmapPlotTests(unittest.TestCase):
    def setUp(self):
        self.plt = pyplot()
        self.metadata = {'ks':[6,7,8,9], 'w_step':.05}
        self.pairs = [row([6,8],[.25,.75],.8),row([7,9],[.4,.6],-.2)]
        self.triples = [row([6,7,8],[.2,.3,.5],.9),row([7,8,9],[.1,.6,.3],.7)]

    def tearDown(self):
        self.plt.close('all')

    def test_mirrored_weights_missing_cells_and_winning_row(self):
        matrix,weights = _heatmap_data(self.pairs+[row([6,8],[.8,.2],.5)],self.metadata['ks'],2)
        self.assertAlmostEqual(matrix[0,2],.8)
        self.assertAlmostEqual(matrix[2,0],.8)
        self.assertAlmostEqual(weights[0,2],.25)
        self.assertAlmostEqual(weights[2,0],.75)
        self.assertAlmostEqual(matrix[1,3],-.2)
        self.assertTrue(np.isnan(matrix.diagonal()).all())
        self.assertTrue(np.isnan(matrix[0,1]))
        # Weight lookup follows the fixed k, not its position in the saved tuple.
        for k,expected in [(6,.2),(7,.3),(8,.5)]:
            matrix,weights = _heatmap_data(self.triples[:1],self.metadata['ks'],3,k)
            self.assertTrue(np.allclose(weights[np.isfinite(matrix)],expected))
            self.assertTrue(np.isnan(matrix[self.metadata['ks'].index(k)]).all())

    def test_notebook_style_and_optional_triangular_overlays(self):
        fig = _heatmap_figure(self.pairs,self.metadata,2,[None],False,self.plt)
        ax = fig.axes[0]
        self.assertEqual(ax.images[0].cmap.name,'YlOrRd')
        self.assertEqual(ax.images[0].get_clim(),(0,.8))
        self.assertEqual(ax.get_xlabel(),'$k_2$')
        self.assertEqual(ax.get_ylabel(),'$k_1$')
        self.assertFalse(ax.patches)
        fig = _heatmap_figure(self.pairs,self.metadata,2,[None],True,self.plt)
        circles = fig.axes[0].patches
        self.assertEqual(len(circles),2)
        self.assertTrue(all(isinstance(p,Circle) and p.center[0]>p.center[1] for p in circles))
        self.assertEqual(fig.axes[-1].get_ylabel(),'$k_1$ weight')
        # Shared normalization across panels, with a blank (unsearched) panel.
        fig = _heatmap_figure(self.triples[:1],self.metadata,3,[6,7,9],True,self.plt,vmax=1)
        self.assertEqual([a.get_title() for a in fig.axes[:3]],['$k_1$=6','$k_1$=7','$k_1$=9'])
        self.assertTrue(all(a.images[0].get_clim()==(0,1) for a in fig.axes[:3]))
        self.assertEqual([len(a.patches) for a in fig.axes[:3]],[1,1,0])
        expected = self.plt.get_cmap('Greys',21)(.3)
        np.testing.assert_allclose(fig.axes[1].patches[0].get_facecolor(),expected)

    def test_repeated_lengths_use_saved_lower_order_correlations(self):
        singles = [row([6],[1],.4),row([7],[1],.5),row([8],[1],.6),row([9],[1],float('nan'))]
        matrix,weights = _heatmap_data(self.pairs,self.metadata['ks'],2,lower_rows=singles)
        np.testing.assert_allclose(matrix.diagonal(),[.4,.5,.6,np.nan],equal_nan=True)
        self.assertTrue(np.isnan(weights.diagonal()).all())
        self.assertTrue(np.isnan(matrix[0,1]))  # an unsearched distinct pair
        self.assertAlmostEqual(matrix[0,2],.8)
        # Pair (6,8) must color (6,6,8), (6,8,6), and (6,8,8).
        matrix,weights = _heatmap_data(self.triples,self.metadata['ks'],3,6,
                                       lower_rows=singles+self.pairs)
        for y,x in [(0,2),(2,0),(2,2)]:
            self.assertAlmostEqual(matrix[y,x],.8)
            self.assertTrue(np.isnan(weights[y,x]))
        self.assertAlmostEqual(matrix[0,0],.4)  # (6,6,6)
        self.assertAlmostEqual(matrix[1,2],.9)  # distinct triple unchanged
        self.assertAlmostEqual(weights[1,2],.2)
        self.assertTrue(np.isnan(matrix[0,1]))  # missing pair (6,7)
        self.assertTrue(np.isnan(matrix[1,3]))  # unsearched triple (6,7,9)
        # Fixed k need not be the smaller length of the saved pair.
        matrix,_ = _heatmap_data([],self.metadata['ks'],3,8,lower_rows=singles+self.pairs)
        for y,x in [(0,0),(0,2),(2,0)]:
            self.assertAlmostEqual(matrix[y,x],.8)
        self.assertAlmostEqual(matrix[2,2],.6)
        # Undefined saved pairs remain undefined rather than using singles.
        matrix,_ = _heatmap_data([],self.metadata['ks'],3,6,
                                 lower_rows=singles+[row([6,7],[.5,.5],float('nan'))])
        self.assertTrue(np.isnan(matrix[0,1]))

    def test_reduced_cells_can_render_an_otherwise_empty_panel(self):
        lower = [row([6],[1],.95),*self.pairs]
        fig = _heatmap_figure([],self.metadata,3,[6],True,self.plt,lower_rows=lower)
        data = fig.axes[0].images[0].get_array()
        self.assertAlmostEqual(data[0,0],.95)
        self.assertAlmostEqual(data[0,2],.8)
        self.assertTrue(np.ma.getmaskarray(data)[1,2])
        self.assertFalse(fig.axes[0].patches)  # no arbitrary duplicate-weight split
        self.assertEqual(fig.axes[0].images[0].get_clim(),(0,.95))

    def test_cli_loads_lower_order_files_and_single_k_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'run.json').write_text(json.dumps(self.metadata))
            singles = [row([6],[1],.4)]
            for size,rows in [(1,singles),(2,self.pairs),(3,self.triples)]:
                write_table(root/f'combinations_{size}.tsv',rows,['ks','weights','rho'])
            def render(size):
                args = ['plot','heatmap','--results',str(root),'--output',str(root/'out.pdf'),
                        '--ensemble-size',str(size)]
                if size == 3:
                    args.extend(['--fixed-k','6'])
                with patch('okkmer.plotting.save') as save, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args),0)
                return save.call_args.args[0].axes[0].images[0].get_array()
            self.assertAlmostEqual(render(2)[0,0],.4)
            self.assertAlmostEqual(render(3)[0,2],.8)
            (root/'combinations_1.tsv').rename(root/'single_k.tsv')
            self.assertAlmostEqual(render(2)[0,0],.4)
            (root/'single_k.tsv').unlink()
            self.assertTrue(np.ma.getmaskarray(render(2))[0,0])

    def test_one_k_and_empty_slice(self):
        fig = _heatmap_figure([row([6],[1],.4),row([8],[1],.8)],self.metadata,1,[None],False,self.plt)
        self.assertEqual(fig.axes[0].images[0].get_array().shape,(1,4))
        self.assertEqual(fig.axes[0].get_xlabel(),'$k_1$')
        self.assertEqual(list(fig.axes[0].get_yticks()),[])
        with self.assertRaisesRegex(ValueError,'No evaluated combinations'):
            _heatmap_figure(self.triples[:1],self.metadata,3,[9],False,self.plt)

    def test_cli_defaults_exports_and_validation(self):
        parsed = parser().parse_args(['plot','heatmap','--results','r','--output','o.pdf'])
        self.assertFalse(parsed.show_weights)
        self.assertEqual(parsed.k_radius,0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'run.json').write_text(json.dumps(self.metadata))
            for size,rows in [(2,self.pairs),(3,self.triples)]:
                write_table(root/f'combinations_{size}.tsv',rows,['ks','weights','rho'])
            def invoke(*args):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return main(['plot','heatmap','--results',str(root),'--output',str(root/'plot.pdf'),*args])
            with patch('okkmer.plotting._heatmap_figure', wraps=_heatmap_figure) as render:
                self.assertEqual(invoke('--ensemble-size','3','--fixed-k','7','--k-radius','1','--show-weights'),0)
                self.assertEqual(render.call_args.args[3],[6,7,8])
            self.assertTrue((root/'plot.pdf').read_bytes().startswith(b'%PDF'))
            for flags in [('--ensemble-size','1','--show-weights'),('--k-radius','1'),
                          ('--ensemble-size','3'),('--fixed-k','7'),('--k-radius','-1'),
                          ('--vmax','nan'),('--vmax','0')]:
                self.assertEqual(invoke(*flags),2,flags)


if __name__ == '__main__':
    unittest.main()
