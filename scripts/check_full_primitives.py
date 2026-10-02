#!/usr/bin/env python3
"""Real operands + MLX kernels versus the shared native Rust evaluator."""
import hashlib,json,pathlib,sys,subprocess
import numpy as np
import mlx.core as mx
import mlx.nn as nn
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from transport import encode,decode,atomic
from checkpoint import Checkpoint
m=json.loads((ROOT/'checkpoints/full.manifest.json').read_text());directory=ROOT/'artifacts/full-primitives';directory.mkdir(parents=True,exist_ok=True);results=[]
def run(op,x,dims=(),scalars=(),tensor='',aux=()):
 h=dict(version=1,model=m['model'],pack_hash=m['pack_hash'],input_hash='0'*64,step=len(results),op=op,tensor=tensor,dims=list(dims),scalars=list(scalars));
 if aux:h['aux']=list(aux)
 a=directory/(op+'.request.bin');b=directory/(op+'.response.bin');atomic(a,encode(h,x));subprocess.run([str(ROOT/'target/release/primitive'),str(a),str(b),str(ROOT/'checkpoints/full.manifest.json'),str(ROOT/'checkpoints/full.pack')],check=True)
 return decode(b.read_bytes())[1]
def compare(op,actual,gold,**extra):
 gold=np.asarray(gold,dtype=np.float32).ravel();actual=actual[:len(gold)];error=abs(actual-gold);record=dict(op=op,max_error=float(error.max()),mean_error=float(error.mean()),bitwise_equal=bool(np.array_equal(actual.view(np.uint32),gold.view(np.uint32))),**extra);results.append(record);print(record,flush=True)
def bf(x):return mx.array(x).astype(mx.bfloat16)
def array(x):mx.eval(x);return np.array(x.astype(mx.float32))
prefix='model.language_model.layers.0.linear_attn';conv=np.load(ROOT/'artifacts/full-reference/conv0.npz');delta=np.load(ROOT/'artifacts/full-reference/delta0.npz');mlp=np.load(ROOT/'artifacts/full-reference/mlp0.npz');n=132
x=conv['input'][0,:,4096:4352];y=run('conv_state',x,[n,256,4,4096],tensor=prefix+'.conv1d.weight');compare('conv_state',y,delta['v'][0,:,:2]);assert np.array_equal(y[n*256:].reshape(3,256),x[-3:])
convolved=array(nn.silu(bf(conv['output'])))[0]
q=run('rms_scaled',convolved[:,:2048],[n*16,128],[1e-6,1/128]);compare('rms_scaled_q',q,delta['q'])
k=run('rms_scaled',convolved[:,2048:4096],[n*16,128],[1e-6,128**-0.5]);compare('rms_scaled_k',k,delta['k'])
gates=run('delta_gates',np.concatenate([delta['a'].ravel(),delta['b'].ravel()]),[n,32],tensor=prefix+'.A_log',aux=[prefix+'.dt_bias']);compare('delta_decay',gates[:n*32],delta['g']);compare('delta_beta',gates[n*32:],delta['beta'])
y=run('swiglu_bf16',np.concatenate([mlp['gate'][0,:24].ravel(),mlp['up'][0,:24].ravel()]),[24*9216]);compare('swiglu_bf16',y,mlp['output'][0,:24])
attention=np.load(ROOT/'artifacts/full-reference/attention3.npz');q=attention['q'][0,0];k=attention['k'][0,0];v=attention['v'][0,0];expected=array(mx.fast.scaled_dot_product_attention(bf(q)[None,None],bf(k)[None,None],bf(v)[None,None],scale=1/16,mask='causal'))
y=run('attention_bf16',np.concatenate([q.ravel(),k.ravel(),v.ravel()]),[n,256]);compare('attention_bf16',y,expected)
from mlx_vlm.models.qwen3_5.language import Qwen3_5RMSNormGated
ck=Checkpoint(ROOT/'checkpoints/base/model.safetensors-00002-of-00002.safetensors');norm=Qwen3_5RMSNormGated(128);norm.weight=bf(ck.tensor(prefix+'.norm.weight'));rng=np.random.default_rng(41);out=delta['output'][0,:20];z=array(bf(rng.normal(size=out.shape).astype(np.float32)));expected=array(norm(bf(out),bf(z)));y=run('gated_norm',np.concatenate([out.ravel(),z.ravel()]),[20*32,128],[1e-6],tensor=prefix+'.norm.weight');compare('gated_norm',y,expected)
# RoPE split-half rotation; all text axes share the same absolute positions.
x=array(bf(rng.normal(size=(n,256)).astype(np.float32)));freq=1/(10000000.**(mx.arange(0,64,2).astype(mx.float32)/64));angles=mx.arange(n).astype(mx.float32)[:,None]*freq[None,:];cos=mx.concatenate([mx.cos(angles),mx.cos(angles)],axis=1);sin=mx.concatenate([mx.sin(angles),mx.sin(angles)],axis=1);rot=mx.array(x[:,:64]);expected=mx.concatenate([(rot*cos+mx.concatenate([-rot[:,32:],rot[:,:32]],axis=1)*sin).astype(mx.bfloat16),bf(x[:,64:])],axis=1)
y=run('rope',x,[n,256,64,0],[10000000.]);compare('rope',y,array(expected))
(directory/'report.json').write_text(json.dumps(dict(scope='Native Rust versus official MLX primitive kernels, real operands where available; not full-model judgment',results=results),indent=2)+'\n')
