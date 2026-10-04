#!/usr/bin/env python3
"""Check joined dispatch bounds before any new ordinary query."""
import pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from joined_inference import JoinedPrefixGraph
from roll_inference import RollPrefixGraph
class Tests(unittest.TestCase):
 def graph(self):
  g=JoinedPrefixGraph.__new__(JoinedPrefixGraph);g.position_offset=45;g.cache={'metadata':{'token_ids':list(range(45))}}
  g.eight=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unvalidated joined entry'))
  return g
 def test_unsupported_boundary_uses_existing_graph(self):
  for n in [88,89]:
   g=self.graph();ids=list(range(45+n))
   with patch.object(RollPrefixGraph,'forward',return_value='fallback')as old:
    self.assertEqual(g.forward(ids),'fallback');old.assert_called_once_with(ids,32)
   self.assertFalse(g.join_effective)
 def test_invalid_prefix_and_layers_reject_before_query(self):
  for ids,layers in [([999]*132,32),(list(range(132)),31),(list(range(45)),32),(list(range(513)),32)]:
   with self.assertRaisesRegex(ValueError,'identity'):self.graph().forward(ids,layers)
if __name__=='__main__':unittest.main()
