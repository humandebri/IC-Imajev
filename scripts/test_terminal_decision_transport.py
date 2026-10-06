#!/usr/bin/env python3
"""Fused ordinary-query routing, client-held checkpoint and option identity."""
import json,pathlib,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode
from full_inference import JournalTransport

class TerminalDecisionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.directory=pathlib.Path(self.temp.name);self.calls=[]
 def transport(self,options=('yes','no'),fused=True):
  t=JournalTransport.__new__(JournalTransport)
  t.directory=self.directory;t.index=0;t.measurements=[];t.replayed=0;t.model='a'*64;t.pack_hash='b'*64;t.input_hash='c'*64;t.frame_version=2;t.wire_codec='bf16-block256-exact-v1';t.fuse_terminal_decision=fused;t.decision_options=list(options)
  def command(cmd):
   self.calls.append(cmd)
   self.assertEqual(cmd['op'],'terminal_step_decision');self.assertEqual(cmd['options'],list(options))
   h,_=decode(pathlib.Path(cmd['input']).read_bytes());h['step']+=1
   pathlib.Path(cmd['output']).write_bytes(encode(h,np.array([0.,-0.,1.],dtype=np.float32)))
   return dict(ok=dict(instructions=100,stable_read_bytes=0,request_bytes=10,reply_bytes=20,decision=dict(value='yes',abstained=False,probabilities=[.8,.1],unknown_probability=.1,raw_logits=[1.,0.,-1.],instructions=5,calibration_version='p3-r2-s000291-authored')),wall_seconds=.01)
  t.command=command
  return t
 def run_terminal(self,t):
  return t.run('terminal_attention_mlp_integer',np.zeros(5120,np.float32),[1,0],tensor='model.language_model.layers.31.self_attn.q_proj.weight')
 def test_one_query_and_replay_restores_both_state_and_decision(self):
  t=self.transport();out=self.run_terminal(t)
  self.assertEqual(len(self.calls),1);self.assertEqual(t.terminal_decision['instructions'],5);self.assertEqual(t.measurements[0]['ok']['instructions'],100)
  replay=self.transport();again=self.run_terminal(replay)
  self.assertEqual(out.tobytes(),again.tobytes());self.assertEqual(replay.terminal_decision,t.terminal_decision);self.assertEqual(len(self.calls),1);self.assertEqual(replay.replayed,1)
 def test_checkpoint_binds_options_order_and_query_mode(self):
  self.run_terminal(self.transport())
  for options,fused in [(('no','yes'),True),(('yes','unknown'),True),(('yes','no'),False)]:
   with self.assertRaisesRegex(ValueError,'checkpoint'):self.run_terminal(self.transport(options,fused))
  self.assertEqual(len(self.calls),1)
 def test_missing_typed_decision_in_checkpoint_rejected(self):
  self.run_terminal(self.transport());p=self.directory/'000000.metric.json';r=json.loads(p.read_text());del r['ok']['decision'];p.write_text(json.dumps(r))
  with self.assertRaisesRegex(ValueError,'checkpoint'):self.run_terminal(self.transport())

if __name__=='__main__':unittest.main()
