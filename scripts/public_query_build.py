"""Apply the current public-query policy to the archived optimized runtime."""
import re

PUBLIC = ('step', 'terminal_step_decision', 'decision_fast', 'decision',
          'status', 'pack_status', 'weight_cache_status')
GUARD = '''// Ordinary queries are public. Replicated execution remains an owner operation.
fn query_access() {
    if ic_cdk::api::in_replicated_execution() { owner(); }
}
'''

def expose(text, methods):
    for method in methods:
        pattern = r'(\bfn ' + re.escape(method) + r'\([^{}]*\)[^{}]*\{\s*)owner\(\);'
        text, count = re.subn(pattern, r'\1query_access();', text)
        if count != 1:
            raise ValueError(f'expected one owner guard for {method}, found {count}')
    return text

def expose_public_queries(directory):
    path = directory / 'lib.rs'
    text = expose(path.read_text(), PUBLIC)
    if 'fn query_access(' in text:
        raise ValueError('query_access already exists in archive')
    text = text.replace('fn owner() {', GUARD + 'fn owner() {', 1)
    if GUARD not in text:
        raise ValueError('missing owner function in archive')
    path.write_text(text)
    for filename, method in [('query_mlp_delta.rs', 'mlp_delta_front'),
                             ('query_attention_mlp_front.rs', 'attention_mlp_front')]:
        path = directory / filename
        path.write_text(expose(path.read_text(), [method]))
