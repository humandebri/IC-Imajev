"""Inherited compilers and release profile variables cannot select the build configuration."""
from pathlib import Path
import unittest
from build_optimized_inference import build_environment, RELEASE_PROFILE


class OptimizedBuildTests(unittest.TestCase):
    def test_rejects_inherited_compiler_flags_and_profile_overrides(self):
        inherited = dict(PATH='kept', RUSTC='other-rustc', RUSTDOC='other-rustdoc',
                         RUSTUP_TOOLCHAIN='stable', RUSTC_WRAPPER='other-wrapper', CARGO_BUILD_TARGET='wasm32-unknown-unknown',
                         CARGO_PROFILE_RELEASE_LTO='false', CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_OPT_LEVEL='0',
                         RUSTFLAGS='other-flags', CARGO_ENCODED_RUSTFLAGS='other-flags')
        env, flags = build_environment(Path('/source-tree'), Path('/output'), inherited)
        self.assertNotIn('RUSTC', env)
        self.assertNotIn('RUSTDOC', env)
        self.assertNotIn('RUSTC_WRAPPER', env)
        self.assertNotIn('CARGO_BUILD_TARGET', env)
        self.assertFalse(any(k.startswith('CARGO_PROFILE_') for k in env))
        self.assertEqual(env['RUSTUP_TOOLCHAIN'], '1.97.1')
        self.assertEqual(env['PATH'], 'kept')
        self.assertEqual(env['CARGO_ENCODED_RUSTFLAGS'], '\x1f'.join(flags))
        self.assertEqual(RELEASE_PROFILE['lto'], '"thin"')
        self.assertEqual(inherited['RUSTC'], 'other-rustc')


if __name__ == '__main__':
    unittest.main()
