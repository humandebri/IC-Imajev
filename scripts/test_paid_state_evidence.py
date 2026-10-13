import hashlib
import errno
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from paid_state_evidence import read_state, validate, digest_file, capture
from paid_instruction_accounting import decode_snapshot
from test_paid_instruction_accounting import blob, integer


class StateTests(unittest.TestCase):
    def fixture(self, directory):
        memory = bytearray(4*1024*1024)
        address, arrays, text, floats = 1024*1024+64, 2*1024*1024, 2*1024*1024+1024, 3*1024*1024
        struct.pack_into('<Q9I', memory, address, 9,32,arrays,32,32,arrays+384,32,2560,floats,2560)
        for i in range(64):
            struct.pack_into('<III',memory,arrays+i*12,64,text+i*64,64)
            memory[text+i*64:text+(i+1)*64] = f'{i:064x}'.encode()
        values = [-0.0]+[i/8 for i in range(1,2560)]
        memory[floats:floats+10240] = struct.pack('<2560f',*values)
        (directory/'vmemory_0.bin').write_bytes(memory)
        raw = blob(7,blob(6,bytes.fromhex('a'*64)))+integer(34,3)+integer(63,100)
        (directory/'canister.pbuf').write_bytes(raw)
        snapshot = decode_snapshot(raw,1,'target','a'*64)
        state = read_state(directory,9,4096)
        evidence = dict(directory=directory.name,page_size=4096,snapshot=snapshot,state=state,
            files={p.name:dict(bytes=p.stat().st_size,sha256=digest_file(p)) for p in directory.iterdir()})
        return memory,address,floats,evidence

    def test_original_words_hash_order_and_signed_zero_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'heap';directory.mkdir();_,_,_,evidence=self.fixture(directory)
            state=validate(directory.parent,evidence,evidence['snapshot'],9)
            self.assertEqual(state['hidden_hashes'],[f'{i:064x}' for i in range(32)])
            self.assertEqual(state['state_hashes'],[f'{i:064x}' for i in range(32,64)])
            self.assertTrue(state['final_hidden_bits'].startswith('00000080'))

    def test_cross_volume_capture_copies_identical_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'checkpoints/0000000000000001/canister_states/target'
            source.mkdir(parents=True); _,_,_,original=self.fixture(source)
            output=root/'retained'; output.mkdir()
            meter=SimpleNamespace(root=root/'checkpoints',hex='target')
            with patch('paid_state_evidence.os.link',side_effect=OSError(errno.EXDEV,'Cross-device link')):
                evidence=capture(meter,original['snapshot'],dict(job_id=9),output/'heap',output)
            self.assertEqual(validate(output,evidence,original['snapshot'],9),original['state'])
            for name in evidence['files']:
                self.assertEqual((output/'heap'/name).read_bytes(),(source/name).read_bytes())

    def test_capture_does_not_hide_other_link_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'checkpoints/0000000000000001/canister_states/target'
            source.mkdir(parents=True); _,_,_,original=self.fixture(source)
            output=root/'retained'; output.mkdir()
            meter=SimpleNamespace(root=root/'checkpoints',hex='target')
            with patch('paid_state_evidence.os.link',side_effect=PermissionError(errno.EACCES,'Denied')):
                with self.assertRaises(PermissionError):
                    capture(meter,original['snapshot'],dict(job_id=9),output/'heap',output)

    def test_changed_bytes_metadata_paths_and_ambiguous_records_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'heap';directory.mkdir();memory,address,floats,evidence=self.fixture(directory)
            for edit in [lambda e:e.update(directory='../heap'),lambda e:e['snapshot'].update(canister_version=4),
                         lambda e:e['state'].update(final_hidden_bits='changed')]:
                changed=json.loads(json.dumps(evidence));edit(changed)
                with self.assertRaises(ValueError):validate(directory.parent,changed,evidence['snapshot'],9)
            struct.pack_into('<Q9I',memory,address+64,9,32,2*1024*1024,32,32,2*1024*1024+384,32,2560,floats+4,2560)
            (directory/'vmemory_0.bin').write_bytes(memory)
            with self.assertRaisesRegex(ValueError,'unambiguous'):read_state(directory,9,4096)
            with self.assertRaisesRegex(ValueError,'checksum'):validate(directory.parent,evidence,evidence['snapshot'],9)

    def test_overlay_precedence_and_invalid_format_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'heap';directory.mkdir();memory,address,_,_=self.fixture(directory)
            page=address//4096;data=bytearray(memory[page*4096:(page+1)*4096])
            struct.pack_into('<Q',data,address%4096,10)
            path=directory/'0000000000000002_0000_vmemory_0.overlay'
            original=bytes(data)+struct.pack('<QQQ',page,page+1,0)+struct.pack('<QI',1,0)
            path.write_bytes(original)
            self.assertEqual(read_state(directory,10,4096)['job_id'],10)
            with self.assertRaises(ValueError):read_state(directory,9,4096)
            path.write_bytes(original[:-4]+struct.pack('<I',1))
            with self.assertRaisesRegex(ValueError,'format'):read_state(directory,10,4096)


if __name__ == '__main__':
    unittest.main()
