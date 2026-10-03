import hashlib
import json
import pathlib
import struct
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'client'))
from prefix_hybrid import NAME, encode_request
from transport import decode, encode


class HybridFrameTests(unittest.TestCase):
    def header(self, n=87):
        return dict(version=1, model='a'*64, pack_hash='b'*64, input_hash='c'*64,
                    step=0, op='delta_full_hybrid_integer', tensor='test',
                    dims=[n,32,45,0], scalars=[], encoding=NAME)

    def packet(self):
        # Envelope policy fixture. Numeric packet validation is in Rust.
        return struct.pack('<4sII', b'NPF1', 45, (1<<18)-1) + bytes(1_440_278-12)

    def test_real_capacity_and_lossless_bf16_segments(self):
        for n in [1,45,80,87,89,90]:
            h=self.header(n);v=np.full(n*2560+3*8192, -0., dtype=np.float32)
            b=encode_request(h,v,self.packet());self.assertLessEqual(len(b),2_000_000)
            start=4+struct.unpack('<I',b[:4])[0]
            self.assertEqual(b[start],1)
            self.assertEqual(np.frombuffer(b[start+5:start+5+len(v)*2],dtype='<u2').tobytes(),np.full(len(v),0x8000,dtype='<u2').tobytes())

    def test_input_and_packet_mismatch_rejected(self):
        h=self.header(1);v=np.zeros(2560+3*8192,dtype=np.float32)
        v[0]=1.00001
        with self.assertRaisesRegex(ValueError,'BF16'):encode_request(h,v,self.packet())
        v[0]=0.;h['dims'][2]=44
        with self.assertRaisesRegex(ValueError,'packet identity'):encode_request(h,v,self.packet())
        h=self.header(1);h['dims'][3]=1
        with self.assertRaisesRegex(ValueError,'bounds'):encode_request(h,v,self.packet())

    def test_reply_keeps_identity_and_exact_f32_values(self):
        h=self.header(1);h['step']=1
        base=dict(h,encoding='bf16-block256-exact-v1')
        v=np.array([-0.,1.0000001,0.5],dtype=np.float32)
        encoded=encode(base,v);offset=4+struct.unpack('<I',encoded[:4])[0]
        raw=json.dumps(h,separators=(',',':')).encode()
        body=struct.pack('<I',len(raw))+raw+b'\x00'+encoded[offset:-32]
        returned,out=decode(body+hashlib.sha256(body).digest())
        self.assertEqual(returned,h);self.assertEqual(out.tobytes(),v.tobytes())


if __name__=='__main__':unittest.main()
