#!/usr/bin/env python3
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import encode,decode
H=dict(version=1,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='lora_project',tensor='',dims=[1,1,512,0],aux=[],scalars=[2.],encoding='int8-block256-v1')
class CodecTests(unittest.TestCase):
 def test_padded_virtual_tiles_preserve_all_quantized_bits(self):
  cases=0
  for n in [1,2,3,4,5,36,132]:
   for width in [1,2,7,31,32,63,64,65,255,256,257,441,1236]:
    chunks=[];expected=[];lengths=[]
    for w in [width,width,max(1,width-1)]:
     x=(np.sin(np.arange(n*w,dtype=np.float32)*np.float32(.017))*np.float32(2.123)).astype(np.float32);lengths.append(len(x));_,old=decode(encode(H,x));expected.append(old);chunks.append(np.pad(x,(0,(-len(x))%256)))
    _,got=decode(encode({**H,'op':'lora_grouped','dims':[n,width,512,0,sum([width,width,max(1,width-1)])]},np.concatenate(chunks)));offset=0
    for length,chunk,old in zip(lengths,chunks,expected):
     np.testing.assert_array_equal(got[offset:offset+length].view(np.uint32),old.view(np.uint32));self.assertTrue(np.all(got[offset+length:offset+len(chunk)]==0));offset+=len(chunk)
    cases+=1
  self.assertEqual(cases,91)
if __name__=='__main__':unittest.main()
