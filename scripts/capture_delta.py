#!/usr/bin/env python3
"""Capture actual first-layer recurrent operands from pinned official MLX forward."""
import json,pathlib,sys
import mlx.core as mx
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import reference
from mlx_vlm.models.qwen3_5 import language
original=language.gated_delta_update
captured=False
def hook(q,k,v,a,b,A_log,dt_bias,**kw):
 global captured
 out,state=original(q,k,v,a,b,A_log,dt_bias,**kw)
 if not captured:
  captured=True;g,beta=language.gated_delta_update.__globals__.get('unused',(None,None)) if False else (None,None)
  from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta
  g,beta=_compute_g_beta(A_log,a,b,dt_bias);mx.eval(q,k,v,g,beta,out)
  arrays={name:np.array(value.astype(mx.float32)) for name,value in [('q',q),('k',k),('v',v),('g',g),('beta',beta),('output',out)]}
  np.savez(ROOT/'artifacts/delta-layer0.npz',**arrays)
  print('captured', {k:list(v.shape) for k,v in arrays.items()},flush=True)
 return out,state
language.gated_delta_update=hook
sys.argv=['capture_delta.py','--first-only','--output','artifacts/capture-reference.json']
reference.main()
