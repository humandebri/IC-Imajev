"""Bind frozen scheduler and token carry to the paid entrypoint's prefix."""
import re


def align_prefix_contract(directory):
    paid_path = directory / 'paid_inference.rs'
    paid = paid_path.read_text()
    pattern = r'(?:pub\(super\) )?const (COMMON_PREFIX|PREFIX27):\[u32;(\d+)\]=\[([^]]+)\];'
    match = re.search(pattern, paid)
    if not match:
        raise ValueError('missing paid prefix declaration')
    name, count, ids = match.groups()
    count = int(count)
    if count != len(ids.split(',')) or count == 0:
        raise ValueError('paid prefix length mismatch')
    paid = paid[:match.start()] + 'pub(super) ' + match[0].removeprefix('pub(super) ') + paid[match.end():]
    paid_path.write_text(paid)
    reference = f'super::paid_inference::{name}.len()'
    path = directory / 'update_inference.rs'
    text = path.read_text()
    # The paid admission bound and the frozen non-chunked scheduler must agree.
    guard = 'if !(1..=89).contains(&ids.len())'
    if text.count(guard) == 1:
        text = text.replace(guard, 'if !(1..=UNCHUNKED_SUFFIX_LIMIT).contains(&ids.len())', 1)
        text += '\npub(super) const UNCHUNKED_SUFFIX_LIMIT: usize = 89;\n'
    text, replaced = re.subn(r'if p!=\d+ \{return Err\("prefix token bounds"',
                            f'if p!={reference} {{return Err("prefix token bounds"', text)
    if replaced != 1:
        raise ValueError('missing frozen prefix registration guard')
    text = re.sub(r'if b.tokens!=\d+ \{return Err\("stream prefix identity"',
                  f'if b.tokens!={reference} {{return Err("stream prefix identity"', text)
    path.write_text(text)
    path = directory / 'chunked_update.rs'
    if path.exists():
        text = path.read_text()
        text, replaced = re.subn(r'const PREFIX:usize=(?:\d+|super::paid_inference::COMMON_PREFIX.len\(\));',
                                f'const PREFIX:usize={reference};', text)
        if replaced != 1:
            raise ValueError('missing token carry prefix constant')
        path.write_text(text)
    # The runtime is a separate crate, so bind its cache size while generating
    # the frozen source rather than referring to a canister crate constant.
    path = directory / 'runtime/prefix_state_cache.rs'
    if path.exists():
        text, replaced = re.subn(r'const TOKENS:usize=\d+;', f'const TOKENS:usize={count};', path.read_text())
        if replaced != 1:
            raise ValueError('missing fixed runtime prefix cache length')
        path.write_text(text)
    return count
