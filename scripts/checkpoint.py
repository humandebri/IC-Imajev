"""Minimal mmap safetensors reader, preserving explicit BF16/F32 precision."""
import json,mmap,pathlib,struct
import numpy as np
class Checkpoint:
 def __init__(self,path):
  self.file=open(path,'rb');self.map=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ);n,=struct.unpack('<Q',self.map[:8]);self.offset=8+n;self.header=json.loads(self.map[8:self.offset])
 def tensor(self,name):
  t=self.header[name];a,z=t['data_offsets'];a+=self.offset
  if t['dtype']=='BF16':return (np.frombuffer(self.map,dtype='<u2',count=(z-t['data_offsets'][0])//2,offset=a).astype(np.uint32)<<16).view(np.float32).reshape(t['shape'])
  if t['dtype']=='F32':return np.frombuffer(self.map,dtype='<f4',count=(z-t['data_offsets'][0])//4,offset=a).reshape(t['shape']).copy()
  raise ValueError(t['dtype'])
