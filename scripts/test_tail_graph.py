#!/usr/bin/env python3
"""Tail routing boundaries and 13 encoded requests, independent of model math."""
import pathlib,sys,tempfile,types,unittest,subprocess
from unittest.mock import patch
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'client'),str(ROOT/'scripts')]
from tail_inference import TailPrefixGraph
from joined_inference import JoinedPrefixGraph,C,H,KV
from test_mlp_stream_codec import carry
from test_mlp_delta_stream_codec import reply
import test_mlp_attention_finish_codec as finish_fixtures

class Tests(unittest.TestCase):
 def test_cli_rejects_bad_tail_configuration_before_files_or_network(self):
  for flags in [['--tail-start'],['--tail-heads28','8'],['--tail-start','--join-start','--roll-start','--tail-heads28','7']]:
   result=subprocess.run([sys.executable,str(ROOT/'scripts/run_prefix_canister.py'),'--canister','invalid','--cache','missing',*flags],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertNotIn('Traceback',result.stderr);self.assertIn('tail',result.stderr)
 def test_bad_chunks_reject_before_parent_initialization(self):
  for name,value in [('tail_heads28',True),('tail_heads28',7),('tail_heads29',32),('tail_front28',6145),('tail_front30',H),('tail_down29',True),('tail_down29',C),('tail_down29',1025)]:
   with patch.object(JoinedPrefixGraph,'__init__')as init:
    with self.assertRaises(ValueError):TailPrefixGraph(**{name:value})
    init.assert_not_called()
 def test_unsupported_boundary_and_bad_prefix_reject_before_new_query(self):
  for n in [88,89]:
   g=TailPrefixGraph.__new__(TailPrefixGraph);g.position_offset=45;g.cache={'metadata':{'token_ids':list(range(45))}}
   with patch.object(JoinedPrefixGraph,'forward',return_value='old')as old:
    self.assertEqual(g.forward(list(range(45+n))),'old');old.assert_called_once_with(list(range(45+n)),32)
   self.assertFalse(g.tail_effective)
  for ids,layers in [([999]*132,32),(list(range(132)),31),(list(range(45)),32),(list(range(513)),32)]:
   with self.assertRaisesRegex(ValueError,'identity'):g.forward(ids,layers)
 def test_tail_encodes_thirteen_queries_and_keeps_state_and_terminal_exports(self):
  from mlp_stream_codec import length
  from delta_mlp_start_codec import CONV
  for n in [1,80,87]:
   with tempfile.TemporaryDirectory()as name:
    g=TailPrefixGraph.__new__(TailPrefixGraph);g.position_offset=45;g.layers=[];g.tail_heads28=6;g.tail_front28=5888;g.tail_heads29=24;g.tail_front30=512;g.tail_down29=1024
    g.t=types.SimpleNamespace(directory=pathlib.Path(name),measurements=[],index=0,model='a'*64,pack_hash='b'*64,input_hash='c'*64)
    g.cache={'states':[dict(conv=np.zeros((3,8192),np.float32),delta_log=np.zeros(45*6176,np.float32))if i%4!=3 else dict(keys=np.zeros((45,4,256),np.float32),values=np.zeros((45,4,256),np.float32),positions=np.arange(45,dtype=np.int32))for i in range(32)]}
    g.recorded_hidden=lambda hidden,layer:hidden
    calls=[]
    def send(layer,op,codec,dims,encode,*values,**kwargs):
     h=g.header(op,codec,dims,layer);packet=encode(h,*values,**kwargs);self.assertLess(len(packet),2_000_000);calls.append((layer,op,dims))
     if op.startswith('delta_partial_'):
      offset=4+int.from_bytes(packet[:4],'little');self.assertEqual(packet[offset],{23:10,25:12,27:14,28:4}[layer])
     g.t.measurements.append(dict(ok=dict(instructions=1,request_bytes=len(packet),reply_bytes=1)));g.t.index+=1
     if op=='mlp_finish_attention_mlp_front':return np.concatenate([np.zeros(n*C,np.float32),carry(n,dims[3]),np.zeros(n*KV,np.float32)])
     if op in ['mlp_complete_delta_partial','mlp_full_delta_partial']:return reply(n,dims[3])[1]
     if op=='delta_partial_mlp_prepare_down':return np.concatenate([finish_fixtures.Tests().carry(n),np.zeros(3*(32-dims[3])*256,np.float32)])
     if op=='mlp_finish_delta_log_mlp_front':return np.concatenate([np.zeros(n*C,np.float32),carry(n,dims[3]),np.zeros(CONV,np.float32)])
     if op=='delta_partial_mlp_front':return np.concatenate([carry(n,dims[5]),np.zeros(3*(32-dims[3])*256,np.float32)])
     if op=='mlp_stream_complete_attention_full':return np.zeros(n*(2*C+KV),np.float32)
     if op=='delta_partial_mlp_full':return np.zeros(2*n*C+3*(32-dims[3])*256,np.float32)
     if op=='delta_mlp_stream_prepare':return np.concatenate([carry(n,dims[1]),np.zeros(CONV,np.float32)])
     if op=='mlp_stream_complete_terminal':return np.zeros(2*C+n*KV,np.float32)
     raise AssertionError(op)
    g.send=send
    (g.t.directory/'layer-30.npy').write_bytes(b'stale')
    out=g.tail(finish_fixtures.Tests().carry(n),n,0,0.)
    self.assertEqual(len(calls),13);self.assertEqual(out.shape,(1,C));self.assertEqual([r['layer']for r in g.layers],list(range(22,32)))
    self.assertFalse((g.t.directory/'layer-30.npy').exists());self.assertEqual(np.load(g.t.directory/'layer-31.npy').shape,(1,C))
    for layer in [24,25,26,28,29,30]:
     with np.load(g.t.directory/'states'/f'layer-{layer:02d}.npz')as z:self.assertEqual(z['conv'].shape,(3,8192))
    for layer in [23,27,31]:
     with np.load(g.t.directory/'states'/f'layer-{layer:02d}.npz')as z:self.assertEqual(z['keys'].shape,(45+n,4,256))
    self.assertEqual(calls[-2][0],29);self.assertEqual(calls[-2][1],'mlp_finish_delta_log_mlp_front');self.assertEqual(calls[-1][1],'mlp_stream_complete_terminal')
 def test_forward_preserves_first37_queries_and_appends13(self):
  g=TailPrefixGraph.__new__(TailPrefixGraph);g.position_offset=45;g.cache={'metadata':{'token_ids':list(range(45))}};calls=[];count=0
  def block(kind,size):
   def run(layer,*args,**kwargs):
    nonlocal count
    calls.append((kind,layer));count+=size;return np.zeros((87,C),np.float32),np.zeros((87,C),np.float32)
   return run
  g.eight=block('eight',8);g.remainder=block('remainder',5)
  def entry(layer,*args):
   nonlocal count
   calls.append(('entry',layer));count+=3;return 'carry',count,0.
  def tail(carry,n,start,clock):
   nonlocal count
   self.assertEqual((carry,n,start),('carry',87,37));count+=13;return 'done'
  g.entry=entry;g.tail=tail
  self.assertEqual(g.forward(list(range(132))),'done');self.assertEqual(count,50)
  self.assertEqual(calls,[('eight',0),('remainder',5),('eight',8),('remainder',13),('eight',16),('entry',21)])

if __name__=='__main__':unittest.main()
