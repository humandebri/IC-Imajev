#!/usr/bin/env python3
import hashlib,json,pathlib,sys,subprocess,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode
class WireTests(unittest.TestCase):
 def test_mixed_random_f32_bit_roundtrip(self):
  bits=np.random.default_rng(42).integers(0,2**32,10001,dtype=np.uint32);bits[::2]&=0xffff0000;v=bits.view(np.float32);v=v[np.isfinite(v)]
  for codec in ['', 'bf16-exact']:
   h=dict(encoding=codec);got=decode(encode(h,v))[1];np.testing.assert_array_equal(got.view(np.uint32),v.view(np.uint32))
 def test_bf16_capacity_and_f32_exception_bounds(self):
  b=encode({'encoding':'bf16-exact'},np.ones(900000,dtype=np.float32));self.assertLess(len(b),2000000);self.assertEqual(len(decode(b)[1]),900000)
  with self.assertRaisesRegex(ValueError,'state size'):encode({'encoding':'bf16-exact'},np.full(500000,1.0000001,dtype=np.float32))
 def test_python_rust_codec_interoperation(self):
  # bf16 op outputs exact BF16; mixed input ensures full-F32 exception path.
  h=dict(version=1,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding='bf16-exact')
  v=np.array([0.,-0.,1.0000001,1e-40,4.25],dtype=np.float32)
  with tempfile.TemporaryDirectory(prefix='imajev-wire-') as d:
   request=pathlib.Path(d)/'r.bin';response=pathlib.Path(d)/'s.bin';request.write_bytes(encode(h,v))
   subprocess.run([str(ROOT/'target/release/primitive'),str(request),str(response)],check=True)
   rh,got=decode(response.read_bytes());self.assertEqual(rh['encoding'],'bf16-exact');self.assertEqual(rh['step'],1)
   expected=(v.view(np.uint32)+0x7fff+((v.view(np.uint32)>>16)&1))&0xffff0000
   np.testing.assert_array_equal(got.view(np.uint32),expected)
if __name__=='__main__':unittest.main()
