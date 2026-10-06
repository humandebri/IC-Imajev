"""Attention checkpoint replay must not resend or replace saved evidence."""
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
import query32_balanced
import query32_templates
from transport import encode


class AttentionCheckpointTests(unittest.TestCase):
    modules = (query32_balanced, query32_templates)

    def graph(self, module, directory):
        graph = module.Query32PrefixGraph.__new__(module.Query32PrefixGraph)
        graph.position_offset = 27
        graph.t = SimpleNamespace(index=3, directory=directory, measurements=[], replayed=0)
        # Use actual frame checksums/decoding; projection arithmetic is outside
        # these tests. The bridge supplies the same deterministic request bytes.
        def header(op, codec, dims, layer):
            return dict(version=3, model='a' * 64, pack_hash='b' * 64,
                        input_hash='c' * 64, step=graph.t.index, op=op,
                        encoding='bf16-exact', dims=dims, scalars=[2., 1e-6],
                        tensor=f'layer-{layer}', aux=[])
        graph.header = header
        graph.prefix_kv = lambda layer: []

        def command(cmd):
            target = header('mlp_stream_prepare', '', [1, 0, cmd['front']], 3)
            target['step'] += 1
            pathlib.Path(cmd['output']).write_bytes(encode(target, [2.]))
            pathlib.Path(cmd['hidden']).write_bytes(np.full(module.C, 0x3f80, dtype='<u2').tobytes())
            pathlib.Path(cmd['kv']).write_bytes(np.full(module.KV, 0x4000, dtype='<u2').tobytes())
            return dict(ok=dict(instructions=123, request_bytes=456, reply_bytes=789))
        graph.t.command = Mock(side_effect=command)
        return graph

    def call(self, module, graph, next_front=5120):
        with patch.object(module, 'bridge', return_value=b'fixed-request'):
            return graph.attention_front(2, 1, 4608, next_front, [])

    def saved(self, module, directory):
        graph = self.graph(module, directory)
        self.call(module, graph)
        self.assertEqual(graph.t.command.call_count, 1)
        self.assertEqual(graph.t.index, 4)
        self.assertEqual(graph.t.replayed, 0)
        return {p.name: p.read_bytes() for p in directory.iterdir()}

    def test_complete_checkpoint_replays_without_writes_or_network(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as folder:
                directory = pathlib.Path(folder)
                before = self.saved(module, directory)
                graph = self.graph(module, directory)
                values, hidden, keys = self.call(module, graph)
                graph.t.command.assert_not_called()
                self.assertEqual(graph.t.index, 4)
                self.assertEqual(graph.t.replayed, 1)
                self.assertTrue(graph.t.measurements[0]['replayed'])
                self.assertEqual(graph.t.measurements[0]['ok']['instructions'], 123)
                np.testing.assert_array_equal(values, [2.])
                np.testing.assert_array_equal(hidden, np.ones(module.C))
                np.testing.assert_array_equal(keys, np.full(module.KV, 2.))
                self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_invalid_checkpoint_is_rejected_without_mutation(self):
        corruptions = ('request.bin', 'expected.bin', 'response.bin', 'previous.bf16', 'kv.bf16', 'metric.json')
        for module in self.modules:
            for suffix in corruptions:
                with self.subTest(module=module.__name__, suffix=suffix), tempfile.TemporaryDirectory() as folder:
                    directory = pathlib.Path(folder)
                    self.saved(module, directory)
                    path = directory / ('000003.' + suffix)
                    if suffix == 'metric.json':
                        metric = json.loads(path.read_text())
                        metric['tensor'] = 'wrong-layer'
                        path.write_text(json.dumps(metric))
                    else:
                        path.write_bytes(path.read_bytes() + b'changed')
                    before = {p.name: p.read_bytes() for p in directory.iterdir()}
                    graph = self.graph(module, directory)
                    with self.assertRaises(ValueError):
                        self.call(module, graph)
                    graph.t.command.assert_not_called()
                    self.assertEqual(graph.t.index, 3)
                    self.assertEqual(graph.t.replayed, 0)
                    self.assertEqual(graph.t.measurements, [])
                    self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_changed_next_front_is_rejected(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as folder:
                directory = pathlib.Path(folder)
                before = self.saved(module, directory)
                graph = self.graph(module, directory)
                with self.assertRaisesRegex(ValueError, 'input mismatch'):
                    self.call(module, graph, next_front=4864)
                graph.t.command.assert_not_called()
                self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_missing_checkpoint_files_are_rejected(self):
        for module in self.modules:
            for suffix in ('request.bin', 'expected.bin', 'previous.bf16', 'kv.bf16'):
                with self.subTest(module=module.__name__, suffix=suffix), tempfile.TemporaryDirectory() as folder:
                    directory = pathlib.Path(folder)
                    self.saved(module, directory)
                    (directory / ('000003.' + suffix)).unlink()
                    graph = self.graph(module, directory)
                    with self.assertRaises(ValueError):
                        self.call(module, graph)
                    graph.t.command.assert_not_called()
                    self.assertFalse((directory / ('000003.' + suffix)).exists())

    def test_reply_identity_checked_even_with_matching_hash(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as folder:
                directory = pathlib.Path(folder)
                self.saved(module, directory)
                graph = self.graph(module, directory)
                target = graph.header('mlp_stream_prepare', '', [1, 0, 5120], 3)
                target['step'] = 9
                path = directory / '000003.response.bin'
                path.write_bytes(encode(target, [2.]))
                metric_path = directory / '000003.metric.json'
                metric = json.loads(metric_path.read_text())
                metric['outputs_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                metric_path.write_text(json.dumps(metric))
                with self.assertRaisesRegex(ValueError, 'reply identity'):
                    self.call(module, graph)
                graph.t.command.assert_not_called()
                self.assertEqual(graph.t.replayed, 0)
                self.assertEqual(graph.t.index, 3)


if __name__ == '__main__':
    unittest.main()
