import struct,unittest
import numpy as np
from prepare_strassen_pack import transform_weights,decode_corrections
class StrassenPackTests(unittest.TestCase):
 def test_sparse_int8_corrections_reconstruct_exact_integer_products(self):
  rng=np.random.default_rng(42);w=rng.integers(-128,128,(8,512),dtype=np.int16).astype(np.int8);q=rng.integers(-127,128,(8,512),dtype=np.int16).astype(np.int32)
  packed,csr=transform_weights(w);flags=decode_corrections(csr,packed.shape);b=packed.astype(np.int32)+flags.astype(np.int32)*256
  a=q.reshape(4,2,2,256);a11,a12,a21,a22=a[:,0,:,:128],a[:,0,:,128:],a[:,1,:,:128],a[:,1,:,128:]
  operands=np.stack([a11+a22,a21+a22,a11,a22,a11+a12,a21-a11,a12-a22]);result=np.zeros((8,8),np.int32)
  for block in range(2):
   p=[operands[m,:,block]@b[m,:,block].T for m in range(7)]
   result[0::2,0::2]+=p[0]+p[3]-p[4]+p[6];result[0::2,1::2]+=p[2]+p[4];result[1::2,0::2]+=p[1]+p[3];result[1::2,1::2]+=p[0]-p[1]+p[2]+p[5]
  np.testing.assert_array_equal(result,q@w.astype(np.int32).T)
 def test_both_signed_overflow_directions_and_bad_csr_are_detected(self):
  w=np.full((8,256),-128,np.int8);w[4:]=127;packed,csr=transform_weights(w);flags=decode_corrections(csr,packed.shape);self.assertTrue(np.any(flags==-1));self.assertTrue(np.any(flags==1))
  with self.assertRaises(ValueError):decode_corrections(csr[:-1],packed.shape)
  bad=bytearray(csr);records,=struct.unpack('<I',bad[:4]);entry_start=4+4*(records+1);bad[entry_start]=128
  with self.assertRaises(ValueError):decode_corrections(bad,packed.shape)
 def test_unaligned_or_wrong_precision_inputs_are_rejected(self):
  for w in [np.zeros((7,256),np.int8),np.zeros((8,128),np.int8),np.zeros((8,256),np.int16)]:
   with self.assertRaises(ValueError):transform_weights(w)
if __name__=='__main__':unittest.main()
