#!/usr/bin/env python3
"""Exercise chunk upload on an explicitly supplied, disposable LOCAL canister."""
import argparse
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1_800_000
# Use the same pinned Candid runtime as the browser, without installing packages.
CODEC = '''
import { readFileSync, writeFileSync } from "node:fs";
import { IDL } from "@icp-sdk/core/candid";
import { methods } from "./src/inference-agent.ts";
const j = JSON.parse(readFileSync(0, "utf8"));
const blob = IDL.Vec(IDL.Nat8);
const ok = t => IDL.Variant({ Ok:t, Err:IDL.Text });
const schemas = {
 prepareModelUpload: { args:[IDL.Text], reply:[IDL.Variant({Ok:IDL.Null,Err:IDL.Text})] },
 uploadModelChunk: { args:[IDL.Nat64,blob,blob], reply:[ok(IDL.Nat64)] },
 verifyModelUpload: { args:[IDL.Nat64], reply:[ok(IDL.Tuple(IDL.Nat64,IDL.Bool))] },
 getModelStatus: methods.getModelStatus,
};
const schema=schemas[j.method];
if(j.hex !== undefined) {
 console.log(JSON.stringify(IDL.decode(schema.reply,Uint8Array.from(Buffer.from(j.hex,"hex")).buffer)[0],(_,v)=>typeof v==="bigint"?v.toString():ArrayBuffer.isView(v)?Array.from(v):v));
} else {
 let args=j.args;
 if(j.method==="uploadModelChunk")args=[BigInt(args[0]),Buffer.from(args[1],"base64"),Buffer.from(args[2],"base64")];
 if(j.method==="verifyModelUpload")args=[BigInt(args[0])];
 writeFileSync(j.file,Buffer.from(IDL.encode(schema.args,args)));
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canister', required=True)
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--identity', default='imajev-local')
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    common = ['--network', 'local', '--identity', args.identity]
    rows = []

    def codec(value):
        return subprocess.check_output(['node', '--experimental-strip-types', '--input-type=module', '-e', CODEC],
            input=json.dumps(value), text=True, cwd=ROOT / 'frontend')

    def call(method, values=()):
        path = directory / f'{len(rows):02d}-{method}.bin'
        codec(dict(method=method, args=values, file=str(path)))
        raw = subprocess.check_output(['icp', 'canister', 'call', args.canister, method,
            '--args-file', str(path), '--args-format', 'bin', '--output', 'hex',
            '--candid', str(ROOT / 'canisters/inference/paid-inference.did'), *common], text=True, cwd=ROOT)
        hex_lines = [line.strip().removeprefix('0x') for line in raw.splitlines()
                     if re.fullmatch(r'(?:0x)?[0-9a-fA-F]+', line.strip())]
        if len(hex_lines) != 1:
            raise ValueError('expected one hex Candid reply')
        result = json.loads(codec(dict(method=method, hex=hex_lines[0])))
        rows.append(dict(method=method, result=result))
        (directory / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
        return result

    data = bytes(CHUNK * 2 + 16)
    manifest = dict(version=1, model='a' * 64, pack_hash='0' * 64, bytes=len(data),
                    tensors=[dict(name='test', offset=0, rows=1, cols=1, dtype='f32', bytes=4)])
    assert call('prepareModelUpload', [json.dumps(manifest)]) == {'Ok': None}

    def upload(offset, chunk, digest=None):
        digest = hashlib.sha256(chunk).digest() if digest is None else digest
        return call('uploadModelChunk', [offset, base64.b64encode(chunk).decode(), base64.b64encode(digest).decode()])

    assert 'checksum' in upload(0, data[:CHUNK], bytes(32))['Err']
    assert 'alignment' in upload(1, data[:CHUNK])['Err']
    assert upload(CHUNK * 2, data[CHUNK * 2:]) == {'Ok': '0'}
    before = call('getModelStatus')
    assert before['chunks'] == [str(CHUNK * 2)] and before['received'] == '0'
    assert upload(CHUNK * 2, data[CHUNK * 2:]) == {'Ok': '0'}
    assert 'retry mismatch' in upload(CHUNK * 2, bytes([1]) + data[CHUNK * 2 + 1:])['Err']
    subprocess.run(['icp', 'canister', 'install', args.canister, '--mode', 'upgrade',
        '--wasm', str(args.wasm), '--yes', *common], cwd=ROOT, check=True)
    # Existing stable metadata does not persist the heap chunk receipt map.
    # Preserve that format: resend chunks after upgrade and re-verify the pack.
    assert call('getModelStatus') == {**before, 'chunks': []}
    assert upload(CHUNK * 2, data[CHUNK * 2:]) == {'Ok': '0'}
    assert upload(0, data[:CHUNK]) == {'Ok': str(CHUNK)}
    assert upload(CHUNK, data[CHUNK:CHUNK * 2]) == {'Ok': str(len(data))}
    assert 'pack hash mismatch' in call('verifyModelUpload', [8_000_000])['Err']
    assert not call('getModelStatus')['ready']

    manifest['pack_hash'] = hashlib.sha256(data).hexdigest()
    assert call('prepareModelUpload', [json.dumps(manifest)]) == {'Ok': None}
    manifest_path = directory / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    pack_path = directory / 'pack.bin'; pack_path.write_bytes(data)
    # Pre-upload a chunk, then exercise the CLI's sequential path and resume logic.
    assert upload(CHUNK, data[CHUNK:CHUNK * 2]) == {'Ok': '0'}
    command = dict(op='upload', manifest=str(manifest_path), pack=str(pack_path), concurrency=16)
    completed = subprocess.check_output([str(args.client), 'http://localhost:8001/', args.canister,
        str(ROOT / 'artifacts/imajev-local.pem')], input=json.dumps(command) + '\n', text=True, cwd=ROOT)
    result = json.loads(completed)['ok']
    assert result['concurrency'] == 1 and result['uploaded'] == len(data), result
    status = call('getModelStatus')
    assert status['ready'] and status['hashed'] == str(len(data))
    assert call('verifyModelUpload', [8_000_000]) == {'Ok': [str(len(data)), True]}
    subprocess.run(['icp', 'canister', 'install', args.canister, '--mode', 'upgrade',
        '--wasm', str(args.wasm), '--yes', *common], cwd=ROOT, check=True)
    assert call('getModelStatus') == {**status, 'chunks': []}, 'ready model must survive upgrade'
    # Recover from a wrong whole-pack digest through the shared CLI path.
    def cli_upload(manifest_file, reset=False):
        command = dict(op='upload', manifest=str(manifest_file), pack=str(pack_path), reset=reset)
        raw = subprocess.check_output([str(args.client), 'http://localhost:8001/', args.canister,
            str(ROOT / 'artifacts/imajev-local.pem')], input=json.dumps(command) + '\n', text=True, cwd=ROOT)
        return json.loads(raw)
    bad_path = directory / 'wrong-hash.json'
    bad_path.write_text(json.dumps({**manifest, 'pack_hash': '0' * 64}))
    assert 'existing pack mismatch' in cli_upload(bad_path)['error']
    assert call('getModelStatus')['ready'], 'mismatch must preserve the current model'
    assert 'already prepared' in cli_upload(bad_path, reset=True)['error']
    owner = subprocess.check_output(['icp', 'identity', 'principal', '--identity', args.identity], text=True).strip()
    subprocess.run(['icp', 'canister', 'install', args.canister, '--mode', 'reinstall',
        '--wasm', str(args.wasm), '--args', '(principal "' + owner + '")', '--yes', *common], cwd=ROOT, check=True)
    assert 'pack hash mismatch' in cli_upload(bad_path, reset=True)['error']
    assert not call('getModelStatus')['ready']
    recovered = cli_upload(manifest_path, reset=True)['ok']
    assert recovered['concurrency'] == 1 and recovered['uploaded'] == len(data)
    assert call('getModelStatus')['ready']
    report = dict(complete=True, network='local', canister=args.canister, checks=rows,
                  sequential_cli=result, reset_recovery_cli=recovered, wasm_sha256=hashlib.sha256(args.wasm.read_bytes()).hexdigest())
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(complete=True, calls=len(rows), sequential_cli=result)))


if __name__ == '__main__':
    main()
