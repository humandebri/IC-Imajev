#!/usr/bin/env python3
"""Separate reference capture; these tensors are never used as inference input."""
import pathlib,sys,json
import mlx.core as mx
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import reference
from mlx_vlm.models.qwen3_5 import language
from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta
output=ROOT/'artifacts/full-reference';output.mkdir(parents=True,exist_ok=True)
if (output/'complete.json').exists():raise SystemExit('Refusing to overwrite reference capture')
engine=reference.MLXDirect(str(ROOT/'artifacts/base-bundle.json'),adapter=str(ROOT/'checkpoints/adapter/mlx'))
layer_ids={id(layer):i for i,layer in enumerate(engine.model.language_model.model.layers)}
def array(x):mx.eval(x);return np.array(x.astype(mx.float32))
original=language.Qwen3_5DecoderLayer.__call__
def layer_hook(self,x,*args,**kwargs):
 i=layer_ids[id(self)];out=original(self,x,*args,**kwargs);np.savez(output/f'layer-{i:02d}.npz',input=array(x),output=array(out));print('captured layer',i,flush=True);return out
language.Qwen3_5DecoderLayer.__call__=layer_hook
conv=engine.model.language_model.model.layers[0].linear_attn.conv1d;original_conv=type(conv).__call__
def conv_hook(self,x,*args,**kwargs):
 out=original_conv(self,x,*args,**kwargs)
 if self is conv:np.savez(output/'conv0.npz',input=array(x),output=array(out),weight=array(self.weight))
 return out
type(conv).__call__=conv_hook
original_delta=language.gated_delta_update;first=True
def delta_hook(q,k,v,a,b,A_log,dt_bias,**kwargs):
 global first
 out,state=original_delta(q,k,v,a,b,A_log,dt_bias,**kwargs)
 if first:
  first=False;g,beta=_compute_g_beta(A_log,a,b,dt_bias)
  np.savez(output/'delta0.npz',**{key:array(value) for key,value in dict(q=q,k=k,v=v,a=a,b=b,A_log=A_log,dt_bias=dt_bias,g=g,beta=beta,output=out,state=state).items()})
 return out,state
language.gated_delta_update=delta_hook
original_swiglu=language.swiglu;first_mlp=True
def swiglu_hook(gate,up):
 global first_mlp
 out=original_swiglu(gate,up)
 if first_mlp:first_mlp=False;np.savez(output/'mlp0.npz',gate=array(gate),up=array(up),output=array(out))
 return out
language.swiglu=swiglu_hook
original_prepare=language.Qwen3_5Attention._prepare_projected_qkv;first_attention=True
def prepare_hook(self,*args,**kwargs):
 global first_attention
 out=original_prepare(self,*args,**kwargs)
 if first_attention:
  first_attention=False;q,k,v,gate,mask=out;np.savez(output/'attention3.npz',q=array(q),k=array(k),v=array(v),gate=array(gate))
 return out
language.Qwen3_5Attention._prepare_projected_qkv=prepare_hook
record=json.loads((ROOT/'artifacts/reference-first.json').read_text())['records'][0]
ids=mx.array([record['token_ids']]);result=engine.model.language_model(ids,return_hidden=True,skip_logits=True)
hidden=array(result.hidden_states[-1]);np.save(output/'final-hidden.npy',hidden)
expected=np.asarray(record['hidden'],dtype=np.float32);error=float(np.abs(hidden[0,-1]-expected).max())
assert error==0.,('capture changed reference',error)
(output/'complete.json').write_text(json.dumps(dict(model_lock=record.get('model_lock_sha256',json.loads((ROOT/'artifacts/reference-first.json').read_text())['model_lock_sha256']),tokens=len(record['token_ids']),final_hidden_matches_record=True,layers=32),indent=2)+'\n')
print('full reference captured; final hidden exactly matches first fixture',flush=True)
