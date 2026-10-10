#!/usr/bin/env python3
"""Versioned frame digests, Rust interop and removal of duplicate encoding."""
import hashlib,pathlib,struct,subprocess,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));sys.path.insert(0,str(ROOT/'scripts'))
from transport import encode,decode
from test_journal import FakeJournal
class FrameTests(unittest.TestCase):
 def header(self,version,codec):return dict(version=version,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding=codec)
 def test_both_versions_interoperate_with_rust_and_preserve_payload(self):
  values=np.array([0.,-0.,1.0000001,1e-40,4.25],np.float32)
  expected=(values.view(np.uint32)+0x7fff+((values.view(np.uint32)>>16)&1))&0xffff0000
  for codec in ['','bf16-exact','bf16-block256-exact-v1']:
   for version in [1,2]:
    with self.subTest(version=version,codec=codec),tempfile.TemporaryDirectory() as directory:
     body=encode(self.header(version,codec),values)
     np.testing.assert_array_equal(decode(body)[1].view(np.uint32),values.view(np.uint32))
     request=pathlib.Path(directory)/'r.bin';response=pathlib.Path(directory)/'s.bin';request.write_bytes(body)
     subprocess.run([str(ROOT/'target/release/primitive'),str(request),str(response)],check=True)
     rh,got=decode(response.read_bytes());self.assertEqual(rh['version'],version);self.assertEqual(rh['step'],1)
     np.testing.assert_array_equal(got.view(np.uint32),expected)
 def test_host_checksum_version_preserves_payload_and_uses_sha256(self):
  values=np.array([0.,-0.,1.0000001,1e-40,4.25],np.float32)
  for codec in ['','bf16-exact','bf16-block256-exact-v1']:
   with self.subTest(codec=codec):
    header=self.header(3,codec);body=encode(header,values)
    self.assertEqual(body[-32:],hashlib.sha256(body[:-32]).digest())
    decoded,got=decode(body);self.assertEqual(decoded,header)
    np.testing.assert_array_equal(got.view(np.uint32),values.view(np.uint32))
 def test_unknown_version_corruption_and_wrong_digest_are_rejected(self):
  for version in [0,4,True,'2']:
   with self.assertRaisesRegex(ValueError,'version'):encode(self.header(version,''),[1.])
  for version in [1,2,3]:
   with self.subTest(version=version):
    b=encode(self.header(version,''),[1.]);n,=struct.unpack('<I',b[:4])
    for offset in [4,4+n,len(b)-1]:
     bad=bytearray(b);bad[offset]^=1
     with self.assertRaises((ValueError,UnicodeError)):decode(bad)
    with self.assertRaisesRegex(ValueError,'checksum'):decode(b[:-32]+bytes(32))
    changed=b.replace(f'"version":{version}'.encode(),f'"version":{version % 3 + 1}'.encode(),1)
    with self.assertRaisesRegex(ValueError,'checksum'):decode(changed)
  b=encode(self.header(2,''),[1.])
  with self.assertRaisesRegex(ValueError,'checksum'):decode(b[:-32]+hashlib.sha256(b[:-32]).digest())
  with self.assertRaisesRegex(ValueError,'header'):decode(struct.pack('<I',16385)+b[4:])
 def test_journal_encodes_request_once_and_binds_checksum_on_resume(self):
  for version in [1,2,3]:
   with tempfile.TemporaryDirectory() as directory:
    path=pathlib.Path(directory);t=FakeJournal(path);t.frame_version=version
    with patch('full_inference.encode',wraps=encode) as journal_encode,patch('transport.encode',wraps=encode) as transport_encode:
     t.run('bf16',[1.,2.]);self.assertEqual(journal_encode.call_count,1);self.assertEqual(transport_encode.call_count,0)
    resumed=FakeJournal(path);resumed.frame_version=version;resumed.run('bf16',[1.,2.]);self.assertEqual(resumed.replayed,1);self.assertEqual(resumed.calls,0)
    changed=FakeJournal(path);changed.frame_version=version % 3 + 1
    with self.assertRaisesRegex(ValueError,'input mismatch'):changed.run('bf16',[1.,2.])
if __name__=='__main__':unittest.main()
