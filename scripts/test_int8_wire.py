#!/usr/bin/env python3
import hashlib,pathlib,struct,subprocess,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode,int8_prefix
H=dict(version=1,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding='int8-block256-v1')
class Tests(unittest.TestCase):
 def test_block_error_bound_zero_and_boundary_lengths(self):
  for count in (0,1,255,256,257,511,1025,900000):
   v=np.random.default_rng(count).normal(0,3,count).astype(np.float32);v[::37]=0
   _,q=decode(encode(H,v));self.assertEqual(q.shape,v.shape)
   for start in range(0,count,256):
    block=v[start:start+256];scale=max(float(abs(block).max())/127,np.finfo(np.float32).tiny)
    self.assertLessEqual(float(abs(q[start:start+256]-block).max()),scale/2+2e-6)
  _,q=decode(encode(H,np.zeros(1025,dtype=np.float32)));self.assertTrue(np.all(q==0))
 def test_delta_recurrent_state_and_gates_are_bitwise_preserved(self):
  n,k,v=2,128,128
  for count,prefix in [(n*(2*k+v+2)+k*v,n*(2*k+v)),(n*v+k*v,n*v)]:
   a=np.random.default_rng(count).normal(size=count).astype(np.float32);h={**H,'op':'delta_bf16','dims':[n,k,v]}
   _,got=decode(encode(h,a));np.testing.assert_array_equal(a[prefix:].view(np.uint32),got[prefix:].view(np.uint32))
  for op,a,d in [('embed',np.array([248319,1],dtype=np.float32),[2,2560]),('delta_gates',np.array([.9765625,.999998,-.00031],dtype=np.float32),[])]:
   _,got=decode(encode({**H,'op':op,'dims':d},a));np.testing.assert_array_equal(got.view(np.uint32),a.view(np.uint32))
 def test_nonfinite_scale_range_count_and_length_rejected(self):
  raw=encode(H,[1.]);n=struct.unpack('<I',raw[:4])[0];payload=4+n
  def mutate(offset,data):
   b=bytearray(raw[:-32]);b[offset:offset+len(data)]=data;return bytes(b)+hashlib.sha256(b).digest()
  for b in [mutate(payload+8,struct.pack('<f',0)),mutate(payload+8,struct.pack('<f',float('nan'))),mutate(payload+12,bytes([128])),mutate(payload+4,struct.pack('<I',0))]:
   with self.assertRaises(ValueError):decode(b)
 def test_python_rust_encoder_and_decoder_agree(self):
  # bf16 op is identity for exact BF16 inputs; compare complete encoded output.
  v=np.random.default_rng(11).integers(-64,64,1025).astype(np.float32)/8
  with tempfile.TemporaryDirectory(prefix='imajev-int8-wire-') as name:
   p=pathlib.Path(name);r=p/'r';s=p/'s';r.write_bytes(encode(H,v));subprocess.run([str(ROOT/'target/release/primitive'),str(r),str(s)],check=True)
   _,inputq=decode(r.read_bytes());_,got=decode(s.read_bytes())
   bits=inputq.view(np.uint32);rounded=((bits+0x7fff+((bits>>16)&1))&0xffff0000).view(np.float32)
   _,expected=decode(encode({**H,'step':1},rounded));np.testing.assert_array_equal(got.view(np.uint32),expected.view(np.uint32))
if __name__=='__main__':unittest.main()
