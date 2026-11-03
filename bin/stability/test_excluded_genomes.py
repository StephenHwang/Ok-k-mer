import unittest
import pandas as pd
from ete3 import Tree
from simulation import mask_pairs

class ExclusionTest(unittest.TestCase):
    def test_explicit_exclusion_preserves_tree_and_mask_population(self):
        tree = Tree('((a:1,b:2):1,(c:3,d:4):1,missing:5);', format=1)
        original = tree.write(format=1)
        names = ['a', 'b', 'c', 'd']
        rows = [(a,b,tree.get_distance(a,b)) for i,a in enumerate(names) for b in names[i+1:]]
        df = pd.DataFrame(rows, columns=['genome_1','genome_2','tree_distance'])
        with self.assertRaises(ValueError):
            mask_pairs(tree, df, 0, 0)
        retained, masked = mask_pairs(tree, df, .25, 0, ['missing'])
        self.assertEqual(len(masked), 1)
        self.assertNotIn('missing', masked)
        self.assertEqual(len(retained), 3)
        self.assertEqual(tree.write(format=1), original)
        self.assertTrue(retained.tree_distance.equals(df.loc[retained.index].tree_distance))
