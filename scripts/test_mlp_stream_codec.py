#!/usr/bin/env python3
"""Carry encoding parity and rejection boundaries, with optional saved evidence."""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'client'))
from mlp_stream_codec import C, H, R, NAME, encode_payload, decode_payload, layout
from transport import Transport, decode


def header(n, done):
    return dict(encoding=NAME, op='mlp_stream_finish' if done == H else 'mlp_stream_next',
                dims=[n, done, 0 if done == H else 256], scalars=[2., 1e-6],
                tensor='model.language_model.layers.0.post_attention_layernorm.weight',
                aux=['model.language_model.layers.1.input_layernorm.weight'])


def carry(n, done):
    return np.concatenate([np.full(n * C, -0., np.float32),
                           np.full(n * C, -127., np.float32),
                           np.full(n * C // 256, .0123, np.float32),
                           np.full(n * 2 * R, .1234567, np.float32),
                           np.full(n * (done//256*256), 127., np.float32),
                           np.full(n * (done%256), -0., np.float32),
                           np.full(n * (done // 256), .009, np.float32),
                           np.full(n * R, -.1234567, np.float32)])


class StreamCodecTests(unittest.TestCase):
    def test_lossless_bits_and_bounded_largest_carry(self):
        for n in [1, 7, 87, 89]:
            for done in [256, 4608, H]:
                h, x = header(n, done), carry(n, done)
                p = encode_payload(h, x)
                self.assertLess(len(p), 2_000_000)
                self.assertEqual(decode_payload(h, p).tobytes(), x.tobytes())

    def test_half_block_pending_is_bf16_and_scales_use_per_token_floor(self):
        for n in [1,7,87,89]:
            for done in [128,384,6016,H-128]:
                h,x=header(n,done),carry(n,done)
                h["dims"][2]=128
                p=encode_payload(h,x)
                self.assertEqual(p[0],2)
                self.assertEqual(decode_payload(h,p).tobytes(),x.tobytes())
                self.assertLess(len(p)+16424,2_000_000)
                with self.assertRaises(ValueError):decode_payload(h,bytes([1])+p[1:])
                with self.assertRaises(ValueError):decode_payload(h,p[:-1])
                offset=n*(2*C+C//256+2*R+done//256*256)
                for value in [.1234567,np.inf,np.nan]:
                    bad=x.copy();bad[offset]=value
                    with self.assertRaises(ValueError):encode_payload(h,bad)

    def test_complete_carry_and_plain_reply_are_lossless_and_bounds_are_strict(self):
        h = header(89, 4608)
        h.update(op='mlp_stream_complete', dims=[89, 4608, 4608])
        x = carry(89, 4608)
        self.assertEqual(decode_payload(h, encode_payload(h, x)).tobytes(), x.tobytes())
        plain = np.zeros(2 * 89 * C, np.float32)
        self.assertEqual(decode_payload(h, encode_payload(h, plain)).tobytes(), plain.tobytes())
        for dims in [[1, 0, H], [1, H, 0], [1, 256, 256], [1, H - 256, 512]]:
            h['dims'] = dims
            with self.assertRaises(ValueError):
                layout(h)

    def test_encoder_rejects_scales_and_noncanonical_integers(self):
        n, done = 1, 256
        h, x = header(n, done), carry(n, done)
        for offset in [2 * C, 2 * C + C // 256 + 2 * R + done]:
            for value in [0., -0., -1., np.nan, np.inf]:
                bad = x.copy()
                bad[offset] = value
                with self.assertRaises(ValueError):
                    encode_payload(h, bad)
        for offset in [C, 2 * C + C // 256 + 2 * R]:
            for value in [-128., 128., .5, -0.]:
                bad = x.copy()
                bad[offset] = value
                with self.assertRaises(ValueError):
                    encode_payload(h, bad)

    def test_frame_version_rejects_bool_and_float_before_bridge_creation(self):
        for value in [True, False, 1., 3., '3', 0, 4]:
            with self.assertRaises(ValueError):
                Transport('', '', '', '', '', '', frame_version=value)

    def test_saved_canister_frames_keep_exact_payload_bytes(self):
        count = 0
        for label in ['617', 'insufficient', 'maximum']:
            directory = ROOT / f'artifacts/f32_k_continue/stream-check-{label}'
            if not directory.exists():
                self.skipTest('Saved local canister evidence is unavailable')
            for path in sorted(directory.glob('*.bin')):
                raw = path.read_bytes()
                h, x = decode(raw)
                if h.get('encoding') != NAME:
                    continue
                header_size = int.from_bytes(raw[:4], 'little')
                self.assertEqual(encode_payload(h, x), raw[4 + header_size:-32], path)
                count += 1
        self.assertGreaterEqual(count, 250)
        print(f'Compared {count} saved MLP stream payloads byte for byte.')


if __name__ == '__main__':
    unittest.main()
