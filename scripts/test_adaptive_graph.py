"""Check profile selection and fallback identity before any adaptive query."""
import pathlib, sys, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from adaptive_inference import AdaptivePrefixGraph, MEASURED_MODULE, short_plan
from tail_inference import TailPrefixGraph
from limit_fallback import standard_command
class Tests(unittest.TestCase):
    def graph(self,p=27,module=MEASURED_MODULE):
        g=AdaptivePrefixGraph.__new__(AdaptivePrefixGraph);g.position_offset=p
        g.cache={'metadata':{'token_ids':list(range(p)),'wasm_sha256':module}}
        g.send=lambda *a,**k: (_ for _ in ()).throw(AssertionError('unexpected query'))
        return g
    def test_unmeasured_shape_or_module_uses_fixed_route(self):
        for n,p,module in [(70,27,MEASURED_MODULE),(67,45,MEASURED_MODULE),(67,27,'different')]:
            g=self.graph(p,module);ids=list(range(p+n))
            with patch.object(TailPrefixGraph,'forward',return_value='fixed') as fixed:
                self.assertEqual(g.forward(ids),'fixed');fixed.assert_called_once_with(ids,32)
            self.assertFalse(g.adaptive_effective)
    def test_prefix_rejected_before_dispatch(self):
        for ids,layers in [([999]*94,32),(list(range(94)),31),(list(range(27)),32),(list(range(513)),32)]:
            with self.assertRaisesRegex(ValueError,'prefix identity'):self.graph().forward(ids,layers)
    def test_plan_alignment_and_budget_boundary(self):
        for n in range(1,70):
            p=short_plan(n,27)
            self.assertTrue(0<p.front<9216 and p.front%256==0)
            self.assertTrue(0<p.heads<32 and p.heads%2==0)
            self.assertTrue(0<p.plain_heads<32 and p.plain_heads%2==0)
        self.assertLess(short_plan(69,27).plain_heads,short_plan(67,27).plain_heads)
        for n,p in [(0,27),(True,27),(67,0),(67,133)]:
            with self.assertRaises(ValueError):short_plan(n,p)
    def test_fallback_strips_adaptive_route(self):
        cmd=standard_command(['--adaptive-start','--tail-start','--join-start','--roll-start','--cache','verified-cache'],pathlib.Path('/tmp/fallback'))
        self.assertFalse(set(['--adaptive-start','--tail-start','--join-start','--roll-start']) & set(cmd))
        self.assertIn('verified-cache',cmd)
if __name__=='__main__':unittest.main()
