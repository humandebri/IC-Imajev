#!/usr/bin/env python3
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from proof_inputs import read_bytes, read_report, residual_cases, selections, validate_join_settings


class ProofInputsTests(unittest.TestCase):
    def test_reference_is_frozen_before_parsing_and_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / 'report.json'
            path.write_text('{"queries": []}')
            hashes = {}
            self.assertEqual(read_report(path, root, hashes), {'queries': []})
            self.assertIn('report.json', hashes)
            path.write_text('{"queries": [1]}')
            with self.assertRaises(ValueError):
                read_report(path, root, hashes)

    def test_binary_reference_cannot_change_or_be_parsed_from_another_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory); path = root / 'tensor.bin'; hashes = {}
            path.write_bytes(b'original')
            raw = read_bytes(path, root, hashes)
            path.write_bytes(b'changed')
            self.assertEqual(raw, b'original')
            with self.assertRaises(ValueError):
                read_bytes(path, root, hashes)

    def test_join_bounds_before_queries(self):
        good = dict(front=1792, attention_front=6272, down_rows=1408,
                    residual_raw_threshold=.1, residual_dictionary=True,
                    compress_residual=True, entry_heads=20, entry_front=4096)
        validate_join_settings(SimpleNamespace(**good))
        validate_join_settings(SimpleNamespace(**dict(good, entry_front=4224)))
        for key, value in [('front', 0), ('front', 9216), ('attention_front', 6273),
                           ('down_rows', 2560), ('down_rows', 33),
                           ('entry_heads', 21), ('entry_heads', 32),
                           ('entry_front', 1), ('entry_front', 9216),
                           ('residual_raw_threshold', float('nan')),
                           ('compress_residual', False)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_join_settings(SimpleNamespace(**dict(good, **{key: value})))

    def test_all_42_residual_cases_required_without_duplicates(self):
        cases = [dict(label=label, layer=layer, begin=begin, success=True)
                 for label in ('617', 'insufficient', 'maximum')
                 for layer in (3, 7, 11, 15, 19, 23, 27)
                 for begin in (5120, 5376)]
        self.assertEqual(len(residual_cases({'cases': cases})), 42)
        for invalid in ([], cases[:-1], cases + [cases[0]],
                        [dict(c, success=False) for c in cases]):
            with self.assertRaises(ValueError):
                residual_cases({'cases': invalid})

    def test_real_reference_and_invalid_cli_selections(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        path = root / 'artifacts/prefix_codec/attention-q4-all-v2/report.json'
        if path.exists():
            self.assertEqual(len(residual_cases(json.loads(path.read_text()))), 42)
        self.assertEqual(selections('24,26', ('24', '26'), 'heads'), ['24', '26'])
        for value in ('', '24,', '24,24', '../617'):
            with self.assertRaises(ValueError):
                selections(value, ('24', '26'), 'heads')


if __name__ == '__main__':
    unittest.main()
