"""Cache-policy regression checks; numeric codec parity has separate canister evidence."""
import json
import pathlib
import struct
import tempfile
import unittest
from unittest.mock import patch

from prepare_prefix_reuse import CODEC_MODULE, prepare_cache, sha


class PrefixReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = pathlib.Path(self.temp.name)
        self.identity = dict(version=1, layers={'0': 'source'}, input_hash='original', tokens=45, codec_module=CODEC_MODULE)
        # Only envelope/hash policy is under test here, not dense codec decoding.
        self.packet = struct.pack('<4sII', b'NPF1', 45, (1 << 18) - 1)
        self.save()
        self.mock = patch('prepare_prefix_reuse.source_identity', return_value=self.identity)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def save(self):
        (self.directory / 'layer-00.npf1').write_bytes(self.packet)
        (self.directory / 'cache.json').write_text(json.dumps(dict(identity=self.identity,
            packets={'0': dict(sha256=sha(self.packet), bytes=len(self.packet))})))

    def run_cache(self, module=CODEC_MODULE):
        def forbidden(*_):
            self.fail('a valid cache hit must never call prepare_prefix')
        return prepare_cache(self.directory, self.directory, 'dedicated', lambda: module, forbidden)

    def test_hit_skips_all_preparation_queries(self):
        result = self.run_cache()
        self.assertTrue(result['cache_hit'])
        self.assertEqual(result['preparation_queries'], 0)
        self.assertEqual(result['preparation_instructions'], 0)

    def test_packet_tamper_rejected_before_query(self):
        (self.directory / 'layer-00.npf1').write_bytes(self.packet + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'packet hash'):
            self.run_cache()

    def test_different_input_rejected(self):
        self.identity['input_hash'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.run_cache()

    def test_different_module_rejected(self):
        with self.assertRaisesRegex(ValueError, 'module mismatch'):
            self.run_cache('another-module')

    def test_short_prefix_cache_hit(self):
        self.identity['tokens'] = 27
        self.packet = struct.pack('<4sII', b'NPF1', 27, (1 << 18) - 1)
        self.save()
        self.assertTrue(self.run_cache()['cache_hit'])

    def test_packet_token_identity_rejected(self):
        self.packet = struct.pack('<4sII', b'NPF1', 27, (1 << 18) - 1)
        self.save()
        with self.assertRaisesRegex(ValueError, 'packet header'):
            self.run_cache()


if __name__ == '__main__':
    unittest.main()
