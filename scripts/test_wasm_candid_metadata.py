"""Public metadata must preserve executable bytes and replace stale interfaces."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from generate_wasm_candid import embed_public_candid


# A valid module exporting answer() -> 42.
MODULE = bytes.fromhex(
    '0061736d01000000 0105016000017f 03020100 '
    '070a0106616e737765720000 0a06010400412a0b'
)


def custom(name, content):
    name = name.encode()
    payload = bytes([len(name)]) + name + content
    return b'\0' + bytes([len(payload)]) + payload


class CandidMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wasm = Path(self.temp.name) / 'module.wasm'
        self.did = Path(self.temp.name) / 'service.did'
        # Exercise multi-byte section lengths and UTF-8 content.
        self.did.write_text('// ' + '公開' * 100 + '\nservice : { answer : () -> (int32) query; }\n')

    def test_preserves_code_and_unrelated_metadata_and_is_repeatable(self):
        original = MODULE + custom('kept', b'unchanged')
        self.wasm.write_bytes(original)
        embed_public_candid(self.wasm, self.did)
        published = self.wasm.read_bytes()
        self.assertTrue(published.startswith(original))
        embed_public_candid(self.wasm, self.did)
        self.assertEqual(self.wasm.read_bytes(), published)
        subprocess.run(['node', '-e', '''
const fs = require('node:fs');
const mod = new WebAssembly.Module(fs.readFileSync(process.argv[1]));
if (new WebAssembly.Instance(mod).exports.answer() !== 42) throw Error('code changed');
''', str(self.wasm)], check=True)

    def test_replaces_all_stale_public_and_private_sections(self):
        self.wasm.write_bytes(MODULE + custom('icp:private candid:service', b'old') +
                              custom('icp:public candid:service', b'stale') +
                              custom('icp:public candid:service', b'duplicate'))
        embed_public_candid(self.wasm, self.did)
        published = self.wasm.read_bytes()
        self.wasm.write_bytes(MODULE)
        embed_public_candid(self.wasm, self.did)
        self.assertEqual(self.wasm.read_bytes(), published)

    def test_rejects_invalid_module_without_overwriting_it(self):
        for invalid in (b'invalid', MODULE + b'\0\x7f', MODULE + b'\0\x01\x02'):
            with self.subTest(invalid=invalid):
                self.wasm.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    embed_public_candid(self.wasm, self.did)
                self.assertEqual(self.wasm.read_bytes(), invalid)


if __name__ == '__main__':
    unittest.main()
