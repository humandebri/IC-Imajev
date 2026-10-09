"""Canonical camelCase canister exports. No legacy entrypoints are exported."""
import re

# Keep the full noun when it distinguishes a query step from a paid job or
# an owner operation. Internal runtime operations and diagnostics are separate.
METHOD_NAMES = {
    'prepare': 'prepareModelUpload',
    'upload': 'uploadModelBytes',
    'upload_chunk': 'uploadModelChunk',
    'hash_pack': 'verifyModelUpload',
    'seal': 'finalizeModelUpload',
    'pack_status': 'getModelStatus',
    'status': 'getModelUploadProgress',
    'step': 'runInferenceStep',
    'profile_step': 'profileInferenceStep',
    'decision': 'getInferenceDecision',
    'decision_fast': 'getCandidateDecision',
    'terminal_step_decision': 'runFinalInferenceStep',
    'mlp_delta_front': 'runDeltaInferenceStep',
    'attention_mlp_front': 'runAttentionInferenceStep',
    'warm_weights': 'prepareWeightCache',
    'clear_weight_cache': 'clearWeightCache',
    'weight_cache_status': 'getWeightCacheStatus',
    'update_prefix': 'installInferencePrefix',
    'update_infer_start': 'startOwnerInference',
    'update_infer_continue': 'continueOwnerInference',
    'quote': 'getInferenceQuote',
    'infer': 'runPaidInference',
    'configure_paid': 'configurePaidInference',
    'paid_config': 'getPaidInferenceConfig',
    'inference_status': 'getInferenceReceipt',
    'inference_step': 'runPaidInferenceWorker',
    'retry_inference_refund': 'retryInferenceRefund',
    'paid_debug': 'getPaidInferenceDebug',
    'prepare_fixed_prefix_state': 'prepareFixedPrefixCache',
    'paid_fault': 'setPaidInferenceFault',
    'paid_probe_step': 'probePaidInferenceWorker',
    'paid_upgrade_probe': 'preparePaidInferenceUpgradeProbe',
    'paid_reference': 'setPaidInferenceReference',
}


REMOVED_METHODS = {'upload', 'seal', 'status', 'decision', 'decision_fast'}
DIAGNOSTIC_METHODS = {'profile_step', 'update_infer_start', 'update_infer_continue',
                      'paid_debug', 'paid_fault', 'paid_probe_step', 'paid_upgrade_probe', 'paid_reference'}
PRODUCTION_NAMES = {name for method, name in METHOD_NAMES.items()
                    if method not in REMOVED_METHODS | DIAGNOSTIC_METHODS}


def remove_api_handlers(text):
    """Drop obsolete entrypoints from live or frozen Rust source."""
    for method in REMOVED_METHODS:
        pattern = (r'(?m)^#\[ic_cdk::(?:query|update)(?:\(name = "\w+"\))?\]\s*'
                   r'(?:async\s+)?fn ' + re.escape(method) + r'\([^{}]*\)[^{}]*\{')
        match = re.search(pattern, text)
        if not match:
            continue
        depth = 1
        # Skip quoted strings and comments when locating the function end.
        tokens = re.finditer(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|[{}]', text[match.end():], re.S)
        for token in tokens:
            if token[0] == '{': depth += 1
            elif token[0] == '}': depth -= 1
            if depth == 0:
                end = match.end() + token.end()
                text = text[:match.start()] + text[end:].lstrip('\n')
                break
        else:
            raise ValueError(f'unterminated handler: {method}')
    return text


def rename_api_methods(text):
    """Apply the production/diagnostic export policy to live or frozen source."""
    text = remove_api_handlers(text)
    pattern = r'(?m)^#\[ic_cdk::(query|update)(?:\(name = "(\w+)"\))?\]([\s]*)(async\s+)?fn (\w+)\('

    def replace(match):
        kind, existing, space, asynchronous, old = match.groups()
        name = METHOD_NAMES.get(old)
        if name is None or (existing is not None and existing != name):
            raise ValueError(f'unnamed or unexpected public API handler: {old}')
        gate = '#[cfg(feature="paid-update-diagnostics")]'
        prefix = ''
        if old in DIAGNOSTIC_METHODS and not text[:match.start()].rstrip().endswith(gate):
            prefix = gate + '\n'
        return f'{prefix}#[ic_cdk::{kind}(name = "{name}")]{space}{asynchronous or ""}fn {old}('

    text = re.sub(pattern, replace, text)
    return text.replace('"inference_step"', '"runPaidInferenceWorker"')


def rename_directory_api_methods(directory):
    for path in directory.glob('*.rs'):
        text = path.read_text()
        updated = rename_api_methods(text)
        if path.name == 'lib.rs' and 'mod query_attention_mlp_front;' in updated and 'use query_attention_mlp_front::AttentionMlpFrontMeasurement;' not in updated:
            updated = updated.replace('mod query_attention_mlp_front;', 'mod query_attention_mlp_front;\nuse query_attention_mlp_front::AttentionMlpFrontMeasurement;', 1)
        export = 'ic_cdk::export_candid!();'
        if path.name == 'lib.rs' and export in updated and not updated.rstrip().endswith(export):
            updated = updated.replace(export, '', 1).rstrip() + '\n\n' + export + '\n'
        if text != updated:
            path.write_text(updated)
