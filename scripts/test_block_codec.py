import hashlib
import pathlib
import struct
import sys
import unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import encode,decode


class BlockCodecTests(unittest.TestCase):
    def header(self):return dict(version=1,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding='bf16-block256-exact-v1')
    def test_signed_zero_exceptions_tails_and_large_pure_frame(self):
        for n in [0,1,255,256,257,511,512,513,899999]:
            x=np.full(n,-0.,np.float32)
            if n:x[0]=np.array([1],dtype=np.uint32).view(np.float32)[0]
            if n>256:x[256]=np.float32(1.0000001)
            _,got=decode(encode(self.header(),x))
            np.testing.assert_array_equal(got.view(np.uint32),x.view(np.uint32))
    def test_noncanonical_full_block_and_count_bounds(self):
        b=encode(self.header(),np.array([1.0000001],np.float32));n,=struct.unpack('<I',b[:4]);raw=bytearray(b[:-32]);raw[4+n+5:4+n+9]=struct.pack('<f',1.)
        invalid=bytes(raw)+hashlib.sha256(raw).digest()
        with self.assertRaisesRegex(ValueError,'noncanonical'):decode(invalid)
        with self.assertRaises(ValueError):encode(self.header(),np.zeros(900001,np.float32))
        with self.assertRaises(ValueError):encode(self.header(),np.array([np.nan],np.float32))
        with self.assertRaises(ValueError):encode(self.header(),np.full(600000,1.0000001,np.float32))


if __name__=='__main__':unittest.main()
