#!/usr/bin/env python3
import pathlib,sys
import mlx.core as mx
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import reference
from mlx_vlm.trainer.lora_layers import LoRALinear
original=LoRALinear.__call__;captured=False
def hook(self,x):
 global captured
 out=original(self,x)
 if not captured:
  captured=True;mx.eval(x,out);np.savez(ROOT/'artifacts/projection-layer0.npz',x=np.array(x.astype(mx.float32))[0],output=np.array(out.astype(mx.float32))[0,:,:256]);print('projection',x.shape,out.shape,flush=True)
 return out
LoRALinear.__call__=hook
sys.argv=['capture_projection.py','--first-only','--output','artifacts/projection-reference.json'];reference.main()
