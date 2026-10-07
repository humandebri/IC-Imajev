#!/usr/bin/env python3
"""Independent ordered F32 recurrence, vectorized across independent heads/value lanes."""
from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
def check(path):
 raw=Path(path).read_bytes();assert raw[:4]==b'DLC1'
 n,key_major,heads,step=struct.unpack_from('<IIIQ',raw,4)
 assert 1<=n<=90 and key_major==1 and heads==32
 a=np.frombuffer(raw,dtype='<f4',offset=24);assert len(a)==2*32*16384+4*32*n*128+2*32*n
 assert np.isfinite(a).all()
 offset=0
 def take(size,shape):
  nonlocal offset
  v=a[offset:offset+size].reshape(shape);offset+=size;return v
 state=take(32*16384,(32,128,128)).copy();final=take(32*16384,(32,128,128))
 q,k,v=[take(32*n*128,(32,n,128))for _ in range(3)]
 g,b=[take(32*n,(32,n))for _ in range(2)]
 expected=take(32*n*128,(32,n,128));actual=np.empty_like(expected)
 for t in range(n):
  np.multiply(state,g[:,t,None,None],out=state)
  memory=np.zeros((32,128),dtype=np.float32)
  for i in range(128):memory=np.add(memory,np.multiply(state[:,i,:],k[:,t,i,None]))
  innovation=np.multiply(np.subtract(v[:,t,:],memory),b[:,t,None])
  output=np.zeros((32,128),dtype=np.float32)
  for i in range(128):
   state[:,i,:]=np.add(state[:,i,:],np.multiply(k[:,t,i,None],innovation))
   output=np.add(output,np.multiply(state[:,i,:],q[:,t,i,None]))
  actual[:,t,:]=output
 assert np.array_equal(state.view(np.uint32),final.view(np.uint32)),('dense state',step,np.count_nonzero(state.view(np.uint32)!=final.view(np.uint32)))
 assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),('delta output',step,np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32)))
 return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),n=n,step=step,layer=step//2,heads=32,dense_values=state.size,output_values=actual.size,dense_state_bit_equal=True,output_bit_equal=True,numpy=np.__version__)
if __name__=='__main__':print(json.dumps(check(sys.argv[1]),indent=2))
