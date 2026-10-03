import hashlib
import json
import pathlib
import struct
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'client'))
from prefix_start import NAME, encode_request
from projection_codec import block_pack
from transport import decode


class PrefixStartTests(unittest.TestCase):
    def header(self, n):
        return dict(version=1, model='a'*64, pack_hash='b'*64, input_hash='c'*64,
                    step=1, op='prefix_start_integer', encoding=NAME, dims=[n,45],
                    tensor='model.language_model.embed_tokens.weight', scalars=[])

    def test_two_segments_preserve_real_suffix_boundary_and_signed_zero(self):
        for n in [1,80,87,89]:
            h=self.header(n);a=np.full(n*2560,-0.,dtype='<f4');b=np.zeros(n*2560+3*8192,dtype='<f4')
            b[0]=1.0000001
            first=block_pack(a);payload=b'\0'+struct.pack('<I',len(first))+first+block_pack(b)
            header=json.dumps(h,separators=(',',':')).encode()
            body=struct.pack('<I',len(header))+header+payload
            returned,out=decode(body+hashlib.sha256(body).digest())
            self.assertEqual(returned,h);self.assertEqual(out.tobytes(),a.tobytes()+b.tobytes())

    def test_ids_and_conv_are_exact_and_request_fits(self):
        h=self.header(89);packet=struct.pack('<4sII',b'NPF1',45,0)+bytes(1440278-12)
        ids=np.arange(89,dtype='<f4');conv=np.full(3*8192,-0.,dtype='<f4')
        frame=encode_request(h,np.concatenate([ids,conv]),packet)
        start=4+struct.unpack('<I',frame[:4])[0]
        self.assertLessEqual(len(frame),2000000)
        self.assertEqual(np.frombuffer(frame,dtype='<u4',count=89,offset=start+5).tobytes(),np.arange(89,dtype='<u4').tobytes())
        self.assertEqual(frame[start+5+89*4:start+5+89*4+len(conv)*2],np.full(len(conv),0x8000,dtype='<u2').tobytes())
        with self.assertRaisesRegex(ValueError,'direction'):decode(frame)

    def test_invalid_values_and_metadata_reject(self):
        h=self.header(1);packet=struct.pack('<4sII',b'NPF1',45,0);v=np.zeros(1+3*8192,dtype='<f4')
        for bad in [-1.,1.5,16777218.,float('nan')]:
            v[0]=bad
            with self.assertRaises(ValueError):encode_request(h,v,packet)
        v[0]=1.;v[1]=1.0000001
        with self.assertRaisesRegex(ValueError,'BF16'):encode_request(h,v,packet)
        v[1]=0.;h['dims'][0]=90
        with self.assertRaisesRegex(ValueError,'shape'):encode_request(h,v,packet)


if __name__=='__main__':unittest.main()
