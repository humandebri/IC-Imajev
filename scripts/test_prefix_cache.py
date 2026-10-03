#!/usr/bin/env python3
import json,pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from prefix_inference import PrefixTextGraph,load_cache,file_hash,graph_hash,saved_delta_state_version
from full_inference import TextGraph
class Transport:
    wire_codec='bf16-exact';max_floats=900000
class PrefixTests(unittest.TestCase):
    def write_cache(self,directory,version):
        directory=pathlib.Path(directory);(directory/'states').mkdir()
        for i in range(32):
            np.save(directory/f'layer-{i:02d}.npy',np.ones((1,2560),np.float32))
            if (i+1)%4==0:
                state=dict(keys=np.ones((1,4,256),np.float32),values=np.ones((1,4,256),np.float32),positions=np.arange(1,dtype=np.int32))
            else:
                state=dict(conv=np.full((3,8192),3.,np.float32))
                state.update(dict(delta=np.full((32,128,128),7.,np.float32)) if version==1 else dict(delta_log=np.full(6176,0.5,np.float32)))
            np.savez_compressed(directory/f'states/layer-{i:02d}.npz',**state)
        names={f'layer-{i:02d}.npy' for i in range(32)}|{f'states/layer-{i:02d}.npz' for i in range(32)}
        metadata=dict(version=version,token_ids=[1],model='m',pack_hash='p',wasm_sha256='w',graph_sha256=graph_hash(),files={name:file_hash(directory/name) for name in names})
        (directory/'cache.json').write_text(json.dumps(metadata))
        return metadata
    def test_saved_dense_and_log_caches_load_and_continue_with_correct_state(self):
        for version in [1,2]:
            with self.subTest(version=version),tempfile.TemporaryDirectory() as directory:
                self.write_cache(directory,version)
                self.assertEqual(saved_delta_state_version(directory),version)
                cache=load_cache(directory,dict(model='m',pack_hash='p'),'w')
                g=PrefixTextGraph(Transport(),dict(tensors=[]),cache=cache,arithmetic='int8',fuse_delta=True,fuse_delta_projected=True,fuse_delta_full_log=True,compact_heads=True,retain_terminal_state=False)
                g.current_layer=0;calls=[]
                def dense(x,*args):
                    np.testing.assert_array_equal(g.initial_delta(0,2),7.)
                    np.testing.assert_array_equal(g.initial_conv(0,4),3.)
                    self.assertEqual(len(x),1);calls.append('dense')
                def logged(x,*args):
                    np.testing.assert_array_equal(g.initial_delta_log(),0.5)
                    self.assertEqual(len(x),1);calls.append('log')
                g.delta_projected=dense;g.delta_full_log=logged
                g.delta(np.ones((1,2560),np.float32),'layer.linear_attn',0)
                self.assertEqual(calls,['log' if version==2 else 'dense'])
                if version==2:
                    with self.assertRaisesRegex(ValueError,'representation mismatch'):
                        PrefixTextGraph(Transport(),dict(tensors=[]),cache=cache,arithmetic='int8')
    def test_cache_version_mismatch_and_mixed_state_formats_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata=self.write_cache(directory,1)
            metadata['version']=2
            path=pathlib.Path(directory)/'cache.json';path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'representation version'):load_cache(directory,dict(model='m',pack_hash='p'),'w')
            name='states/layer-00.npz'
            np.savez_compressed(pathlib.Path(directory)/name,conv=np.ones((3,8192),np.float32),delta_log=np.ones(6176,np.float32))
            metadata['version']=1;metadata['files'][name]=file_hash(pathlib.Path(directory)/name);path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'representation version'):saved_delta_state_version(directory)
            with self.assertRaisesRegex(ValueError,'representation version'):load_cache(directory,dict(model='m',pack_hash='p'),'w')
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
