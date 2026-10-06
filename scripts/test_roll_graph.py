#!/usr/bin/env python3
"""Verify boundary dispatch before any rolled query and restore mode on errors."""
import pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from roll_inference import RollPrefixGraph
from prefix_inference import PrefixTextGraph
class RollBoundaryTests(unittest.TestCase):
 def graph(self):
  g=RollPrefixGraph.__new__(RollPrefixGraph);g.position_offset=45;g.cache={'metadata':{'token_ids':list(range(45))}};g.roll_blocks=True
  g.roll_block=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unsafe rolled entry'))
  return g
 def test_boundary_uses_standard_graph(self):
  for n in [88,89]:
   g=self.graph();ids=list(range(45+n))
   def standard(self,tokens,layers):
    self_check=self.roll_blocks is False and tokens==ids and layers==32
    if not self_check:raise AssertionError('fallback graph identity')
    return 'safe'
   with patch.object(PrefixTextGraph,'forward',standard):self.assertEqual(g.forward(ids),'safe')
   self.assertFalse(g.roll_effective);self.assertTrue(g.roll_blocks)
 def test_failure_restores_dispatch_mode(self):
  g=self.graph()
  with patch.object(PrefixTextGraph,'forward',side_effect=RuntimeError('query failure')):
   with self.assertRaisesRegex(RuntimeError,'query failure'):g.forward(list(range(134)))
  self.assertTrue(g.roll_blocks)
 def test_prefix_identity_rejects_before_queries(self):
  g=self.graph()
  with self.assertRaisesRegex(ValueError,'identity'):g.forward([999]*132)
if __name__=='__main__':unittest.main()
