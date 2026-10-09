"""Apply current sequence-position bounds to the frozen optimized runtime.

Tile row limits and projection kernel bodies remain unchanged.
"""
def align_sequence_contract(directory, canonical, reference=False):
    runtime = directory / 'runtime'
    declaration = next(line for line in (canonical / 'lib.rs').read_text().splitlines()
                       if line.startswith('pub const MAX_SEQUENCE_TOKENS:'))
    path = runtime / 'lib.rs'
    text = path.read_text()
    if 'pub const MAX_SEQUENCE_TOKENS:' in text:
        raise ValueError('frozen runtime already declares sequence bound')
    text += '\n' + declaration + '\n'
    for old, new in [
        ('|| prefix > 512', '|| prefix > MAX_SEQUENCE_TOKENS'),
        ('|| n + prefix > 512', '|| n + prefix > MAX_SEQUENCE_TOKENS'),
        ('&& d.get(2).copied().unwrap_or(0) <= 512', '&& d.get(2).copied().unwrap_or(0) <= MAX_SEQUENCE_TOKENS'),
        ('&& d[0] + d.get(2).copied().unwrap_or(0) <= 512', '&& d[0] + d.get(2).copied().unwrap_or(0) <= MAX_SEQUENCE_TOKENS'),
    ]:
        if text.count(old) != 1:
            raise ValueError('missing frozen sequence guard: ' + old)
        text = text.replace(old, new, 1)
    if reference:
        import shutil
        shutil.copyfile(canonical / 'attention_reference.rs', runtime / 'attention_reference.rs')
        text += '\n#[cfg(feature="experimental-attention-reference")]\nmod attention_reference;\n#[cfg(feature="experimental-attention-reference")]\npub use attention_reference::{enabled as attention_reference_enabled,set as set_attention_reference};\n'
        live=(canonical / 'lib.rs').read_text()
        begin=live.index('                    #[cfg(feature="experimental-attention-reference")]')
        end=live.index('                    out.extend(crate::profile::measure("gqa_head_views"',begin)
        anchor='                    out.extend(crate::profile::measure("gqa_head_views"'
        if text.count(anchor)!=1:raise ValueError('missing frozen GQA reference insertion')
        text=text.replace(anchor,live[begin:end]+anchor,1)
    path.write_text(text)
    for name in ('attention_fusion.rs', 'attention_full.rs'):
        path = runtime / name
        text = path.read_text()
        old = 'offset>512 || n+offset>512'
        if text.count(old) != 1:
            raise ValueError('missing frozen Attention position guard: ' + name)
        text = text.replace(old, 'offset>crate::MAX_SEQUENCE_TOKENS || n+offset>crate::MAX_SEQUENCE_TOKENS', 1)
        if name == 'attention_fusion.rs':
            if text.count('total>512') != 1:
                raise ValueError('missing frozen Attention Q sequence guard')
            text = text.replace('total>512', 'total>crate::MAX_SEQUENCE_TOKENS', 1)
        path.write_text(text)
