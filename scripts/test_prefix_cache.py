#!/usr/bin/env python3
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from prefix_inference import PrefixTextGraph,load_cache
from full_inference import TextGraph
class Transport:
    wire_codec='bf16-exact';max_floats=900000
class PrefixTests(unittest.TestCase):
    def graph(self):
        cache=dict(metadata=dict(token_ids=[1,2]),states=[dict(conv=np.arange(3*8192,dtype=np.float32).reshape(3,8192),delta=np.arange(32*128*128,dtype=np.float32).reshape(32,128,128))],hidden=[np.ones((2,2560),dtype=np.float32)])
        return PrefixTextGraph(Transport(),dict(tensors=[]),cache=cache,arithmetic='int8')
    def test_state_offsets_and_copy(self):
        g=self.graph();g.current_layer=0
        c=g.initial_conv(4096,8);np.testing.assert_array_equal(c,g.cache['states'][0]['conv'][:,4096:4104]);c[:]=0;self.assertTrue(g.cache['states'][0]['conv'].any())
        d=g.initial_delta(8,8);np.testing.assert_array_equal(d,g.cache['states'][0]['delta'][8:16].reshape(8,16384))
    def test_mismatched_tokens_and_no_suffix_rejected(self):
        g=self.graph()
        for ids in [[1,3,4],[1,2]]:
            with self.assertRaises(ValueError):g.forward(ids)
    def test_full_length_limit_checked_before_suffix(self):
        with self.assertRaises(ValueError):self.graph().forward([1,2]+[4]*511)
    def test_no_history_defaults(self):
        g=TextGraph(Transport(),dict(tensors=[]));self.assertEqual(g.position_offset,0)
        self.assertTrue(np.all(g.initial_delta(0,2)==0));self.assertTrue(np.all(g.initial_conv(0,4)==0))
if __name__=='__main__':unittest.main()
