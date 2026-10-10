"""Extract Candid from the compiled module and verify its method export contract."""
import argparse
import re
import subprocess
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


def generate_candid(wasm, output, *, diagnostics=False):
    api = check_exports(wasm, diagnostics=diagnostics)
    candid = subprocess.check_output(['node', '-e', EXTRACT, str(Path(wasm).resolve())], text=True)
    service = candid.rsplit('service :', 1)[1]
    methods = {name: 'query' if query else 'update' for name, query in
               re.findall(r'^  (\w+)\s*:[^;]+?( query)?;', service, re.M)}
    if methods != api['exports']:
        raise ValueError('Candid and WASM method exports disagree')
    check_paid_contract(candid)
    Path(output).write_text(candid)
    return api


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wasm', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    generate_candid(args.wasm, args.output, diagnostics=args.diagnostics)
