"""Validate the measured graph's shape gate and partial-matrix boundaries."""
import pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from packed_inference import PackedPrefixGraph,MODULE,packed_plan
from tail_inference import TailPrefixGraph
from limit_fallback import standard_command
class Tests(unittest.TestCase):
 def test_all_admitted_shapes_have_valid_matrix_cuts(self):
  for n in range(1,70):
   p=packed_plan(n,27,MODULE)
   self.assertTrue(0<p.front<9216 and p.front%256==0)
   self.assertTrue(0<p.down<2560 and p.down%32==0)
   self.assertTrue(0<p.plain_heads<32 and p.plain_heads%2==0)
  self.assertGreater(packed_plan(69,27,MODULE).down,packed_plan(67,27,MODULE).down)
 def test_unmeasured_scope_is_not_admitted(self):
  for n,p,m in [(0,27,MODULE),(True,27,MODULE),(70,27,MODULE),(67,0,MODULE),(67,28,MODULE),(67,27,'different')]:self.assertIsNone(packed_plan(n,p,m))
 def test_unmeasured_module_uses_verified_fixed_route(self):
  g=PackedPrefixGraph.__new__(PackedPrefixGraph);g.position_offset=27;g.cache={'metadata':{'token_ids':list(range(27)),'wasm_sha256':'different'}}
  with patch.object(TailPrefixGraph,'forward',return_value='fixed') as fixed:
   self.assertEqual(g.forward(list(range(94))),'fixed');fixed.assert_called_once()
  self.assertFalse(g.adaptive_effective)
 def test_changed_prefix_is_rejected_before_query(self):
  g=PackedPrefixGraph.__new__(PackedPrefixGraph);g.position_offset=27;g.cache={'metadata':{'token_ids':list(range(27)),'wasm_sha256':MODULE}}
  with self.assertRaisesRegex(ValueError,'prefix identity'):g.forward([999]*94)
 def test_fallback_removes_packed_graph_but_keeps_compatible_bridge(self):
  cmd=standard_command(['--packed-start','--tail-start','--join-start','--roll-start','--bridge-binary','bridge','--cache','cache'],pathlib.Path('/tmp/fallback'))
  self.assertFalse(set(['--packed-start','--tail-start','--join-start','--roll-start']) & set(cmd));self.assertIn('--bridge-binary',cmd)
if __name__=='__main__':unittest.main()
