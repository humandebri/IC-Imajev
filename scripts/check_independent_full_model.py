#!/usr/bin/env python3
"""Run a pinned before/after query graph on a disposable, already uploaded IC canister.

The target must have a creation receipt in the evidence directory. Never uses the
shared reference canister. Upgrade only this disposable target; cleanup is recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from evaluate_prompt_accuracy import base_flags


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--records', default='1,2')
    args = parser.parse_args()
    directory = args.directory.resolve()
    build = args.build.resolve()
    receipt = json.loads((directory/'target.json').read_text())
    target = receipt['canister']
    if receipt['network'] != 'http://localhost:8001/' or receipt['created_for'] != 'independent-runtime-full-model-comparison':
        raise ValueError('requires dedicated local creation receipt')
    if target in ('4caro-hl777-77775-aaaba-cai', 'xis3j-paaaa-aaaai-axumq-cai'):
        raise ValueError('shared/mainnet target forbidden')
    network = json.loads(subprocess.check_output(['icp','network','status','--json'],cwd=ROOT,text=True))
    if network['api_url'] != receipt['network']:
        raise ValueError('selected local endpoint differs from the creation receipt')
    if (directory/'comparison.json').exists():
        raise ValueError('comparison evidence already exists')
    modules = dict(before=build/'baseline/full.wasm', after=build/'full.wasm')
    hashes = {key:sha(path) for key,path in modules.items()}
    build_report = json.loads((build/'report.json').read_text())
    if hashes != dict(before=build_report['baseline_wasm_sha256'],after=build_report['wasm_sha256']):
        raise ValueError('build provenance mismatch')
    sources = [Path(__file__),ROOT/'artifacts/query32-v1/run_queries.py',
               ROOT/'scripts/evaluate_prompt_accuracy.py',ROOT/'artifacts/text-short-v2/run_queries.py',
               ROOT/'artifacts/query32-v1/client-build/imajev-client',
               ROOT/'artifacts/text-short-v2/inputs.json',ROOT/'scripts/prepare_weight_cache.py',
               ROOT/'scripts/prepare_fixed_prefix_states.py'] + sorted((ROOT/'client').glob('*.py'))
    source_hashes = {str(path.relative_to(ROOT)):sha(path) for path in sources}
    report = dict(complete=False, local_only=True, canister=target, modules=hashes,
                  source_hashes=source_hashes,build_report_sha256=sha(build/'report.json'),
                  instruction_scope='Handler counters exclude CDK decode/encode; actual IC query success checked',
                  communication_scope='Candid request+reply', cases=[], operations=[], cleanup={})
    def save():
        (directory/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    def icp(*command):
        result = subprocess.run(['icp','canister',*command,'--network','local','--identity','imajev-local'],
                                cwd=ROOT, text=True, capture_output=True)
        report['operations'].append(dict(command=list(command), returncode=result.returncode,
                                         stdout=result.stdout, stderr=result.stderr))
        save()
        result.check_returncode()
        return result.stdout
    def run(command, log):
        with log.open('w') as stream:
            subprocess.run(command,cwd=ROOT,check=True,stdout=stream,stderr=subprocess.STDOUT)
    try:
        status = json.loads(icp('status',target,'--json'))
        if int(status['settings']['wasm_memory_limit'].replace('_','')) != 4*2**30:
            raise ValueError('full weight cache proof requires the reference 4GiB Wasm limit')
        if status['module_hash'].removeprefix('0x') != hashes['before']:
            raise ValueError('dedicated target must start with pinned baseline module')
        for variant,module in modules.items():
            stage = directory/variant
            stage.mkdir(exist_ok=False)
            if variant == 'after':
                icp('install',target,'--mode','upgrade','--wasm',str(module),'--yes')
            for script,name,flags in (
                ('prepare_weight_cache.py','weights',['--include-f32','--require-prepared-rope',
                 '--require-prepared-activation','--require-all-output-pairs']),
                ('prepare_fixed_prefix_states.py','prefix',[])):
                run([sys.executable,'-B',str(ROOT/'scripts'/script),'--canister',target,
                     '--wasm',str(module),'--directory',str(stage/name),*flags],stage/f'{name}.log')
                print(f'{variant} {name} ready',flush=True)
            for index in map(int,args.records.split(',')):
                output = stage/f'record-{index}'
                command = base_flags()
                command[1] = str(ROOT/'artifacts/query32-v1/run_queries.py')
                for flag,value in (('--canister',target),('--wasm',str(module)),
                    ('--reference','artifacts/text-short-v2/inputs.json'),
                    ('--cache','artifacts/query-packing-v3/prefix-v2/queries'),
                    ('--bridge-binary','artifacts/query32-v1/client-build/imajev-client')):
                    command[command.index(flag)+1] = value
                command += ['--hybrid-cache','artifacts/query-packing-v3/packets-v2',
                    '--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision',
                    '--fuse-terminal-tail','--fuse-prefix-start','--tail-start','--join-start',
                    '--roll-start','--packed-start','--record',str(index),'--directory',str(output)]
                run(command,stage/f'record-{index}.log')
                result = json.loads((output/'report.json').read_text())
                if result['executed_query_count'] != 32 or result['replayed_queries'] or result['wasm_sha256'] != hashes[variant]:
                    raise ValueError('requires fresh full 32-query inference')
                if result['max_query_instructions'] >= 5_000_000_000:
                    raise ValueError('query instruction ceiling reached')
                if variant == 'after':
                    reference = directory/'before'/f'record-{index}'
                    old = json.loads((reference/'report.json').read_text())
                    compared = []
                    paths = ['final-hidden.npy'] + [str(p.relative_to(reference)) for p in sorted((reference/'queries').glob('layer-*.npy'))]
                    paths += [str(p.relative_to(reference)) for p in sorted((reference/'queries/states').glob('layer-*.npz'))]
                    for name in paths:
                        equal = (output/name).read_bytes() == (reference/name).read_bytes()
                        if name.endswith('.npz'):
                            with np.load(output/name,allow_pickle=False) as a, np.load(reference/name,allow_pickle=False) as b:
                                equal = set(a.files)==set(b.files) and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a.files)
                        if not equal:
                            raise ValueError(f'bit mismatch: record {index}, {name}')
                        compared.append(dict(path=name,sha256=sha(output/name)))
                    semantic = lambda r:{key:value for key,value in r['decision_query']['ok']['decision'].items() if key!='instructions'}
                    if semantic(result) != semantic(old):
                        raise ValueError('decision/logit/probability mismatch')
                    for field in ('raw_logits','probabilities','unknown_probability'):
                        if np.asarray(semantic(result)[field],dtype='<f4').tobytes() != np.asarray(semantic(old)[field],dtype='<f4').tobytes():
                            raise ValueError('decision F32 bit mismatch: '+field)
                    metrics = ('total_instructions','max_query_instructions','total_candid_bytes','max_observed_heap_bytes','wall_seconds_this_run')
                    row = dict(record=index,tokens=result['tokens'],queries=32,bitwise_equal=True,
                        compared_files=compared,decision=semantic(result),
                        before={key:old[key] for key in metrics},after={key:result[key] for key in metrics},
                        instruction_ratio=result['total_instructions']/old['total_instructions'],
                        per_query_instruction_ratios=[a['ok']['instructions']/b['ok']['instructions'] for a,b in zip(result['queries'],old['queries'],strict=True)])
                    report['cases'].append(row)
                    save()
                print(f'{variant} record {index} complete',flush=True)
        if any(sha(modules[key]) != digest for key,digest in hashes.items()):
            raise ValueError('module changed during verification')
        if source_hashes != {str(path.relative_to(ROOT)):sha(path) for path in sources}:
            raise ValueError('verification sources changed during execution')
        report['complete'] = True
    finally:
        for action in ('stop','delete'):
            result = subprocess.run(['icp','canister',action,target,'--network','local','--identity','imajev-local'],cwd=ROOT,text=True,capture_output=True)
            report['cleanup'][action] = dict(returncode=result.returncode,stderr=result.stderr)
        if any(result['returncode'] for result in report['cleanup'].values()):
            report['complete'] = False
        save()
    if not report['complete']:
        raise RuntimeError('verification or dedicated-canister cleanup incomplete')


if __name__ == '__main__':
    main()
