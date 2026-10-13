#!/usr/bin/env python3
"""Check the local heap decoder against the source-built diagnostic Candid getter."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlparse
from paid_instruction_accounting import CheckpointMeter
from paid_state_evidence import capture, validate_layout, SCHEMA, digest_file
import paid_state_evidence
from paid_update_transport import PaidTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('url','canister','identity-pem'):
        parser.add_argument('--'+name, required=True)
    for name in ('directory','build-report','checkpoint-state-dir','helper','call-helper','bridge'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    url = urlparse(args.url)
    if url.scheme != 'http' or url.hostname not in ('localhost','127.0.0.1','::1'):
        parser.error('Explicit loopback endpoint required')
    build = json.loads(args.build_report.read_text())
    if build['verification_only'] is not True or 'paid-update-diagnostics' not in build['features']:
        parser.error('A source-built diagnostic module is required')
    def bridge(op):
        raw = subprocess.check_output([str(args.bridge.resolve()),args.url,args.canister,args.identity_pem],
            input=json.dumps(dict(op=op))+'\n',text=True)
        return json.loads(raw)['ok']
    assert bridge('module_hash')['module_hash'] == build['module']
    assert bridge('pack_status')['ready'] is False, 'Use a dedicated empty diagnostic canister'
    args.directory.mkdir(parents=True, exist_ok=False)
    wire = PaidTransport(args.directory/'wire',args.canister,args.helper.resolve(),args.call_helper.resolve(),
        url=args.url,identity_pem=args.identity_pem)
    wire.call('paid_upgrade_probe',dict(active=False,refund_state=0))
    getter = wire.call('paid_debug')
    assert getter['result']['job_id'] == 2**64-1
    meter = CheckpointMeter(args.checkpoint_state_dir,args.canister,build['module'])
    deadline, seen, heap = time.monotonic()+180, set(), None
    while time.monotonic()<deadline:
        try:
            _, snapshot = meter.latest()
        except ValueError as error:
            if str(error) not in ('No immutable checkpoint for selected canister','Replica instruction counter unavailable'):
                raise
            time.sleep(0.25);continue
        if snapshot['height'] not in seen:
            seen.add(snapshot['height'])
            try:
                heap = capture(meter,snapshot,dict(job_id=2**64-1),args.directory/f'heap-{snapshot["height"]:016x}',args.directory)
                break
            except (FileNotFoundError,ValueError) as error:
                if not isinstance(error,FileNotFoundError) and 'found 0' not in str(error):
                    raise
        time.sleep(0.25)
    if heap is None:
        raise TimeoutError('Diagnostic fixture checkpoint unavailable')
    build_file = args.directory/'build-report.json'
    shutil.copyfile(args.build_report,build_file)
    getter_file = args.directory/'wire'/f'{wire.counter:04d}-paid_debug.json'
    report = dict(complete=True,schema=SCHEMA,module=build['module'],parser_sha256=digest_file(Path(paid_state_evidence.__file__)),
        heap=heap,build_evidence=dict(file='build-report.json',sha256=digest_file(build_file)),
        getter_evidence=dict(file=str(getter_file.relative_to(args.directory)),sha256=digest_file(getter_file)))
    report_path = args.directory/'report.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    validate_layout(report_path)
    assert bridge('module_hash')['module_hash'] == build['module']
    print('Paid state decoder matches the typed diagnostic getter, including signed zero.')


if __name__ == '__main__':
    main()
