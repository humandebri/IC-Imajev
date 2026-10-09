import re
import tempfile
import unittest
from pathlib import Path

from canister_api_names import METHOD_NAMES, REMOVED_METHODS, DIAGNOSTIC_METHODS, PRODUCTION_NAMES, rename_api_methods, rename_directory_api_methods

ROOT = Path(__file__).resolve().parents[1]


class ApiNameTests(unittest.TestCase):
    def test_async_and_feature_guards_are_preserved_without_wrapper(self):
        source = '''#[cfg(feature="paid-update-inference")]
#[ic_cdk::update]
async fn infer(request: InferRequest) -> Result<InferenceResult, InferError> {
    owner_auth(); do_work().await
}
'''
        result = rename_api_methods(source)
        self.assertEqual(result, source.replace('#[ic_cdk::update]', '#[ic_cdk::update(name = "runPaidInference")]'))
        self.assertNotIn('fn api_', result)
        self.assertEqual(rename_api_methods(result), result)

    def test_owner_and_self_guards_and_worker_destinations_are_preserved(self):
        for old, guard in [('prepare', 'owner();'), ('inference_step', 'assert_self();')]:
            source = f'#[ic_cdk::update]\nfn {old}(value:u64) -> Result<(),String> {{ {guard} Ok(()) }}'
            result = rename_api_methods(source)
            self.assertIn(f'name = "{METHOD_NAMES[old]}"', result)
            self.assertIn(f'{{ {guard} Ok(()) }}', result)
            self.assertEqual(result.count('fn '), 1)
        source = 'Call::unbounded_wait(canister, "inference_step").await'
        self.assertIn('"runPaidInferenceWorker"', rename_api_methods(source))

    def test_current_sources_export_every_api_once_and_without_legacy_names(self):
        paths = [*(ROOT / 'canisters/inference/src').glob('*.rs'),
                 ROOT / 'scripts/query_attention_mlp_front.rs',
                 ROOT / 'scripts/prefix_state_cache_canister.rs']
        text = '\n'.join(p.read_text() for p in paths)
        for old, preferred in METHOD_NAMES.items():
            self.assertEqual(text.count(f'name = "{preferred}"'), 0 if old in REMOVED_METHODS else 1, preferred)
            if old in DIAGNOSTIC_METHODS:
                pattern = r'#\[cfg\(feature="paid-update-diagnostics"\)\]\s*#\[ic_cdk::(?:query|update)\(name = "' + preferred + r'"\)\]'
                self.assertRegex(text, pattern)
        self.assertNotIn('fn api_', text)
        self.assertIsNone(re.search(r'#\[ic_cdk::(?:query|update)\]', text))
        for p in paths:
            self.assertEqual(rename_api_methods(p.read_text()), p.read_text(), str(p))

    def test_paid_transport_schema_uses_canonical_methods(self):
        from paid_update_transport import PaidTransport
        with tempfile.TemporaryDirectory() as tmp:
            helper = Path(tmp) / 'mock-call-helper'
            helper.touch()
            transport = PaidTransport(tmp, "local-test", call_helper=helper)
            schema = transport.did.read_text()
            for old in ['quote', 'infer', 'configure_paid', 'inference_status',
                        'inference_step', 'retry_inference_refund', 'paid_debug',
                        'paid_config', 'paid_fault']:
                self.assertIn(METHOD_NAMES[old] + ':', schema)
                self.assertIsNone(re.search(r'(?<=[{;])' + old + ':', schema))

    def test_removed_handlers_and_diagnostic_gates_apply_to_frozen_sources(self):
        source = '#[ic_cdk::query]\nfn decision(state:Vec<u8>) -> Result<(),String> { let braces="}"; Ok(()) }\n'
        self.assertEqual(rename_api_methods(source), '')
        source = '#[ic_cdk::query]\nfn profile_step(state:Vec<u8>) -> Result<(),String> { owner(); Ok(()) }\n'
        updated = rename_api_methods(source)
        self.assertTrue(updated.startswith('#[cfg(feature="paid-update-diagnostics")]'))
        self.assertEqual(rename_api_methods(updated), updated)

    def test_frozen_release_sources_follow_the_same_api_policy(self):
        # A legacy export fixture covers the full policy without frozen local builds.
        frozen = '\n'.join(
            f'#[ic_cdk::{"query" if old in ("step", "profile_step", "decision", "decision_fast") else "update"}]\n'
            f'fn {old}() {{}}' for old in METHOD_NAMES
        ) + '\nic_cdk::export_candid!();\n'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'lib.rs').write_text(frozen)
            rename_directory_api_methods(directory)
            text = '\n'.join(p.read_text() for p in directory.glob('*.rs'))
            exports = set(re.findall(r'ic_cdk::(?:query|update)\(name = "(\w+)"', text))
            self.assertEqual(exports, PRODUCTION_NAMES | {METHOD_NAMES[x] for x in DIAGNOSTIC_METHODS})
            for x in REMOVED_METHODS:
                self.assertNotIn('name = "' + METHOD_NAMES[x] + '"', text)

    def test_unknown_export_requires_a_naming_decision(self):
        with self.assertRaisesRegex(ValueError, 'unnamed or unexpected public API handler'):
            rename_api_methods('#[ic_cdk::update]\nfn mystery() {}')

    def test_frozen_export_runs_after_appended_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / 'lib.rs'
            path.write_text('#[ic_cdk::query]\nfn step(state:StateBytes) -> Result<Measurement,String> { query_access(); run(state) }\nic_cdk::export_candid!();\nmod query_attention_mlp_front;\n')
            rename_directory_api_methods(directory)
            result = path.read_text()
            self.assertTrue(result.rstrip().endswith('ic_cdk::export_candid!();'))
            self.assertIn('name = "runInferenceStep"', result)
            self.assertNotIn('runPaidInference', result)
            rename_directory_api_methods(directory)
            self.assertEqual(result, path.read_text())


if __name__ == '__main__':
    unittest.main()
