"""Extract Candid from the compiled module and verify its method export contract."""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from check_canister_api_exports import check_exports

EXTRACT = r'''
const fs = require('node:fs');
const mod = new WebAssembly.Module(fs.readFileSync(process.argv[1]));
const imports = {};
for (const item of WebAssembly.Module.imports(mod)) {
  if (item.kind !== 'function') throw Error('unsupported import: ' + item.name);
  (imports[item.module] ??= {})[item.name] = () => { throw Error('Candid extraction called ' + item.name); };
}
const instance = new WebAssembly.Instance(mod, imports);
const get = instance.exports.get_candid_pointer ?? instance.exports.__get_candid_interface_tmp_hack;
if (!get) throw Error('module has no Candid getter');
const start = get();
const bytes = new Uint8Array(instance.exports.memory.buffer);
let end = start;
while (end < bytes.length && bytes[end] !== 0) end++;
if (end === bytes.length || end === start) throw Error('invalid Candid pointer');
process.stdout.write(new TextDecoder('utf-8', {fatal: true}).decode(bytes.subarray(start, end)) + '\n');
'''


def check_paid_contract(candid):
    """Reject a frozen paid ABI even when its exported method names match."""
    service = candid.rsplit('service :', 1)[1]
    if not re.search(r'^  runPaidInference\s*:', service, re.M):
        return
    if not re.search(r'^  runPaidInference\s*:\s*\(InferRequest, text\)\s*->', service, re.M):
        raise ValueError('runPaidInference must accept request and request_id only')
    for name in ('Config', 'Quote', 'InferenceResult', 'InferError'):
        declaration = re.search(rf'^type {name} = .*?^}};', candid, re.M | re.S)
        if not declaration:
            raise ValueError(f'missing paid Candid type: {name}')
        if re.search(r'\benabled\s*:|\bPaused\b', declaration[0]):
            raise ValueError(f'obsolete acceptance switch in paid Candid type: {name}')
        if re.search(r'\b(version|quote_version)\s*:|\bQuoteChanged\b', declaration[0]):
            raise ValueError(f'obsolete fee version in paid Candid type: {name}')


def embed_public_candid(wasm, candid_file):
    """Publish the extracted service, replacing metadata without changing code."""
    wasm = Path(wasm).resolve()
    data = wasm.read_bytes()

    def read_leb(position, limit):
        value = 0
        for shift in range(0, 35, 7):
            if position >= limit:
                raise ValueError('truncated Wasm section length')
            byte = data[position]; position += 1
            value |= (byte & 127) << shift
            if byte < 128:
                if value > 0xffffffff:
                    raise ValueError('invalid Wasm section length')
                return value, position
        raise ValueError('invalid Wasm section length')

    def encode_leb(value):
        result = bytearray()
        while value >= 128:
            result.append((value & 127) | 128)
            value >>= 7
        result.append(value)
        return bytes(result)

    if data[:8] != b'\0asm\x01\0\0\0':
        raise ValueError('expected a WebAssembly module')
    sections = [data[:8]]
    position = 8
    names = {b'icp:public candid:service', b'icp:private candid:service'}
    while position < len(data):
        start = position
        kind = data[position]
        size, body = read_leb(position + 1, len(data))
        end = body + size
        if end > len(data):
            raise ValueError('truncated Wasm section')
        replace = False
        if kind == 0:
            length, name_start = read_leb(body, end)
            if name_start + length > end:
                raise ValueError('truncated Wasm custom section name')
            replace = data[name_start:name_start + length] in names
        if not replace:
            sections.append(data[start:end])
        position = end
    name = b'icp:public candid:service'
    payload = encode_leb(len(name)) + name + Path(candid_file).read_bytes()
    sections.append(b'\0' + encode_leb(len(payload)) + payload)
    with tempfile.TemporaryDirectory(prefix='candid-', dir=wasm.parent) as directory:
        published = Path(directory) / 'published.wasm'
        published.write_bytes(b''.join(sections))
        # Inspect the actual section and reject private/duplicate publication.
        subprocess.run(['node', '-e', r'''
const fs = require('node:fs');
const mod = new WebAssembly.Module(fs.readFileSync(process.argv[1]));
const sections = WebAssembly.Module.customSections(mod, 'icp:public candid:service');
if (sections.length !== 1 ||
    WebAssembly.Module.customSections(mod, 'icp:private candid:service').length ||
    !Buffer.from(sections[0]).equals(fs.readFileSync(process.argv[2]))) {
  throw Error('Published Candid metadata differs from the extracted service');
}
''', str(published), str(Path(candid_file).resolve())], check=True)
        published.replace(wasm)


def generate_candid(wasm, output, *, diagnostics=False, profile='optimized', embed_public=False):
    api = check_exports(wasm, profile=profile, diagnostics=diagnostics)
    candid = subprocess.check_output(['node', '-e', EXTRACT, str(Path(wasm).resolve())], text=True)
    service = candid.rsplit('service :', 1)[1]
    methods = {name: 'query' if query else 'update' for name, query in
               re.findall(r'^  (\w+)\s*:[^;]+?( query)?;', service, re.M)}
    if methods != api['exports']:
        raise ValueError('Candid and WASM method exports disagree')
    check_paid_contract(candid)
    Path(output).write_text(candid)
    if embed_public:
        embed_public_candid(wasm, output)
    return api


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wasm', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--profile', choices=['optimized', 'cargo', 'base'], default='optimized')
    parser.add_argument('--embed-public', action='store_true',
                        help='Embed the full extracted service as public candid:service metadata')
    args = parser.parse_args()
    generate_candid(args.wasm, args.output, diagnostics=args.diagnostics,
                    profile=args.profile, embed_public=args.embed_public)
