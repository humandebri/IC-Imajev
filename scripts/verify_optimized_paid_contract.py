#!/usr/bin/env python3
"""Exercise production refund/upgrade behavior on an explicit dedicated local IC.

Stops/starts and upgrades only the selected loopback validation canister. Run
after numerical measurements; a successful upgrade clears its heap caches.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse
from uuid import uuid4
from paid_update_transport import PaidTransport, LocalProcessRunner


def validate_contract(path, module):
    """Recheck retained call results and lifecycle evidence before proposal output."""
    if not __debug__:
        raise ValueError('Contract validation requires Python assertions enabled')
    report = json.loads(path.read_text())
    assert report['complete'] and report['live_fault_refund_passed'] and report['upgrade_gate_passed']
    assert not report['failures'] and report['candidate_module'] == module
    url = urlparse(report['url'])
    assert url.scheme == 'http' and url.hostname in ('localhost', '127.0.0.1', '::1')
    for name, digest in report['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        current = path.parent
        for part in relative.parts:
            current /= part
            assert not current.is_symlink()
        assert hashlib.sha256(current.read_bytes()).hexdigest() == digest
    def row(name):
        file = report['observations'][name]
        assert file in report['files']
        result = json.loads((path.parent/file).read_text())
        assert result['relay'] == (None if name.endswith('_owner') or name=='quote' else report['relay'])
        return result
    failed = row('failed')['result']['Err']['Failed']
    assert any(word in failed['reason'].lower() for word in ('stopped','stopping'))
    fee = row('quote')['result']['Ok']['fee']
    forward = row('failed')['forward']
    assert failed['refund'] == 'Done' and forward['refunded'] == 1_000_000
    assert 0 <= forward['balance_before'] - forward['balance_after'] < fee//100
    assert row('failed')['attached_cycles'] == fee+1_000_000
    assert row('active')['result']['state'] == 'Running'
    receipt = row('failed_receipt')['result']
    assert receipt['refund'] == 'Done' and receipt['job_id'] == failed['job_id'] and receipt['state'] == {'Failed':failed['reason']}
    assert row('scope_owner')['result'] is None and row('post_scope_owner')['result'] is None
    assert row('retry_owner')['result'] == {'Err':'missing receipt'}
    retry = row('retry')
    assert retry['result'] == {'Ok':'Done'} and retry['forward']['balance_after'] <= retry['forward']['balance_before']
    assert 'Completed' in row('completed')['result']['state']
    for before, after in [('completed','post_completed'),('failed_receipt','post_failed_receipt'),('config_owner','post_config_owner')]:
        assert row(before)['result'] == row(after)['result']
    replay = row('replay')
    assert replay['result']['Err']['Failed'] == failed and replay['forward']['refunded'] == fee
    assert replay['request'] == row('failed')['request'] and replay['attached_cycles'] == fee
    for name in ('active','failed_receipt','scope_owner','retry','retry_owner','post_failed_receipt','post_scope_owner'):
        assert row(name)['request'] == report['request_id']
    assert row('failed')['request']['request_id'] == report['request_id']
    assert len(row('failed')['request']['request']['token_ids']) == 1024
    assert row('quote')['request'] == row('failed')['request']['request']
    assert row('completed')['request'] == row('post_completed')['request'] == report['completed_request_id']
    assert [a['action'] for a in report['actions']] == ['install','stop','start','install']
    first, *others = report['actions']
    assert first['returncode'] != 0 and 'paid inference active' in first['stderr']
    assert all(a['returncode'] == 0 for a in others)
    assert report['module_observations'] == [module]*4
    return report


def main():
    if not __debug__:
        raise SystemExit('Local contract verification requires Python assertions enabled')
    from check_paid_instruction_evidence import validate
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('url','canister','relay','identity','identity-pem'):
        p.add_argument('--'+name,required=True)
    for name in ('directory','numerical-report','fixture','wasm','project-root','helper','call-helper','bridge'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    url=urlparse(args.url)
    if url.scheme!='http' or url.hostname not in ('localhost','127.0.0.1','::1'):
        p.error('Explicit loopback targets required')
    network=json.loads(subprocess.check_output(['icp','network','status','--json','--project-root-override',str(args.project_root.resolve())]))
    if urlparse(network['api_url']).port!=url.port:
        p.error('Dedicated project network differs from selected endpoint')
    numerical=validate(args.numerical_report)
    assert numerical['intermediate_state_gate_passed'] is True
    module=numerical['candidate_module']
    assert hashlib.sha256(args.wasm.read_bytes()).hexdigest()==module
    module_observations=[]
    def module_hash():
        raw=subprocess.check_output([str(args.bridge.resolve()),args.url,args.canister,args.identity_pem],input='{"op":"module_hash"}\n',text=True)
        assert json.loads(raw)['ok']['module_hash']==module
        module_observations.append(module)
    module_hash()
    args.directory.mkdir(parents=True,exist_ok=False)
    options=dict(url=args.url,identity_pem=args.identity_pem)
    def wire(name, **extra):
        return PaidTransport(args.directory/name,args.canister,args.helper.resolve(),args.call_helper.resolve(),**options,**extra)
    inference_processes=LocalProcessRunner()
    infer,control=wire('inference',process_runner=inference_processes),wire('control')
    request=next(c['request'] for c in json.loads(args.fixture.read_text())['cases'] if c['total']==1024)
    quote=control.call('quote',request)['result']['Ok']
    request_id='local-contract-'+uuid4().hex
    assert control.call('inference_status',request_id,relay=args.relay)['result'] is None
    payload=dict(request=request,request_id=request_id)
    summary=dict(complete=False,candidate_module=module,request_id=request_id,
                 url=args.url,canister=args.canister,relay=args.relay,
                 numerical_report_sha256=hashlib.sha256(args.numerical_report.read_bytes()).hexdigest(),
                 live_fault_refund_passed=False,upgrade_gate_passed=False,failures=[],actions=[],observations={},
                 module_observations=module_observations)
    def observe(name, transport, kind, value=None, **kwargs):
        result=transport.call(kind,value,**kwargs)
        summary['observations'][name]=str((transport.d/f'{transport.counter:04d}-{kind}.json').relative_to(args.directory.resolve()))
        return result
    # Quote was measured before the evidence map was initialized.
    summary['observations']['quote']='control/0001-quote.json'
    # icp install stops the canister before upgrading, which would finish the
    # worker chain before testing pre_upgrade. Submit install_chunked_code
    # directly instead; prepare chunks before starting the active inference.
    management=args.directory/'management'
    management.mkdir()
    raw_options=[str(args.call_helper.resolve()),'--url',args.url,'--identity-pem',args.identity_pem,
                 '--effective-canister-id',args.canister,'aaaaa-aa']
    def management_args(name,kind,value):
        source=management/(name+'.json');binary=management/(name+'.bin')
        source.write_text(json.dumps(value)+'\n')
        subprocess.run([str(args.helper.resolve()),'args',kind,str(source.resolve()),str(binary.resolve())],check=True)
        return binary.resolve()
    clear=management_args('clear','local_clear_chunks',dict(canister=args.canister))
    subprocess.run([*raw_options,'clear_chunk_store',str(clear)],check=True,capture_output=True,text=True,timeout=180)
    hashes=[]
    wasm=args.wasm.read_bytes()
    for index,offset in enumerate(range(0,len(wasm),1024*1024)):
        chunk=wasm[offset:offset+1024*1024];file=management/f'chunk-{index}.wasm'
        file.write_bytes(chunk);hashes.append(list(hashlib.sha256(chunk).digest()))
        binary=management_args(f'upload-{index}','local_upload_chunk',dict(canister=args.canister,file=str(file.resolve())))
        response=subprocess.check_output([*raw_options,'upload_chunk',str(binary)],text=True,timeout=180)
        (management/f'upload-{index}.reply.hex').write_text(response)
    upgrade=management_args('upgrade','local_chunked_upgrade',dict(canister=args.canister,hashes=hashes,module_hash=list(bytes.fromhex(module))))
    def manage(action,more=(),*,expected_success=True):
        command=([*raw_options,'install_chunked_code',str(upgrade)] if action=='install' else
                 ['icp','canister',action,args.canister,*more,'--network','local','--identity',args.identity,
                  '--project-root-override',str(args.project_root.resolve())])
        result=subprocess.run(command,text=True,capture_output=True,timeout=180)
        evidence=dict(action=action,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        summary['actions'].append(evidence)
        if expected_success:
            assert result.returncode==0, result.stderr
        return result
    pool=ThreadPoolExecutor(max_workers=1)
    stopped=False
    future=None
    try:
        future=pool.submit(observe,'failed',infer,'infer',payload,relay=args.relay,cycles=quote['fee']+1_000_000)
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            receipt=observe('active',control,'inference_status',request_id,relay=args.relay)['result']
            if receipt and receipt['state']=='Running':
                break
            if future.done():
                raise ValueError('Request completed before observing active upgrade guard')
            time.sleep(0.25)
        else:
            raise TimeoutError('Active inference was not observed')
        upgrade_args=['--mode','upgrade','--wasm',str(args.wasm.resolve()),'--args','()']
        rejected=manage('install',upgrade_args,expected_success=False)
        assert rejected.returncode!=0 and 'paid inference active' in rejected.stderr
        module_hash()
        print('Active inference upgrade rejected; stopping local worker chain.',flush=True)
        stopped=True
        manage('stop')
        response=future.result(timeout=180)
        failed=response['result']['Err']['Failed']
        assert any(word in failed['reason'].lower() for word in ('stopped','stopping'))
        assert failed['refund']=='Done'
        assert response['forward']['refunded']==1_000_000
        # Refund::Done is written only after deposit_cycles(fee) succeeds. The
        # relay balance provides a separate live transfer check; network costs
        # must be much smaller than the returned full fee.
        forward=response['forward']
        assert 0<=forward['balance_before']-forward['balance_after']<quote['fee']//100
        manage('start');stopped=False
        receipt=observe('failed_receipt',control,'inference_status',request_id,relay=args.relay)['result']
        assert receipt['refund']=='Done' and 'Failed' in receipt['state']
        assert receipt['job_id']==failed['job_id']
        assert observe('scope_owner',control,'inference_status',request_id)['result'] is None
        retry=observe('retry',control,'retry_inference_refund',request_id,relay=args.relay)
        assert retry['result']=={'Ok':'Done'} and retry['forward']['balance_after']<=retry['forward']['balance_before']
        assert observe('retry_owner',control,'retry_inference_refund',request_id)['result']=={'Err':'missing receipt'}
        summary['live_fault_refund_passed']=True
        completed_id=numerical['inputs'][0]['request_id']
        summary['completed_request_id']=completed_id
        completed=observe('completed',control,'inference_status',completed_id,relay=args.relay)['result']
        assert 'Completed' in completed['state']
        config=observe('config_owner',control,'paid_config')['result']
        manage('install',upgrade_args)
        module_hash()
        assert observe('post_config_owner',control,'paid_config')['result']==config
        assert observe('post_completed',control,'inference_status',completed_id,relay=args.relay)['result']==completed
        assert observe('post_failed_receipt',control,'inference_status',request_id,relay=args.relay)['result']==receipt
        replay=observe('replay',control,'infer',payload,relay=args.relay,cycles=quote['fee'])
        assert replay['result']['Err']['Failed']==failed and replay['forward']['refunded']==quote['fee']
        assert observe('post_scope_owner',control,'inference_status',request_id)['result'] is None
        module_hash()
        summary.update(complete=True,upgrade_gate_passed=True,heap_caches_require_rewarm=True)
        print('Live failure/refund and completed-receipt upgrade checks passed.',flush=True)
    except Exception as error:
        summary['failures'].append(str(error))
        if future is not None and not future.done():
            # Do not leave a paid worker chain running after a failed local
            # validation assertion. The callback can finish and refund while
            # the selected canister stops; finally starts it again.
            try:
                stopped=True
                manage('stop')
                future.result(timeout=180)
            except Exception as cleanup_error:
                summary['failures'].append('Local worker cleanup: '+str(cleanup_error))
        raise
    finally:
        try:
            # Future cancellation cannot stop an already running subprocess.
            # Terminate and reap owned helpers before restarting or inspecting
            # evidence, then join the thread to prevent subsequent file writes.
            inference_processes.cancel()
            pool.shutdown(wait=True,cancel_futures=True)
            if stopped:
                manage('start')
            subprocess.run([*raw_options,'clear_chunk_store',str(clear)],check=True,capture_output=True,text=True,timeout=180)
        finally:
            summary['files']={str(file.relative_to(args.directory)):hashlib.sha256(file.read_bytes()).hexdigest()
                              for file in sorted(args.directory.rglob('*')) if file.is_file() and file.name!='report.json'}
            path=args.directory/'report.json'
            path.write_text(json.dumps(summary,indent=2)+'\n')
            if summary['complete']:
                validate_contract(path,module)


if __name__=='__main__':
    if sys.argv[1:2]==['check']:
        parser=argparse.ArgumentParser(description='Recheck original local contract evidence before migration approval.')
        parser.add_argument('--report',type=Path,required=True)
        parser.add_argument('--paid-report',type=Path,required=True)
        args=parser.parse_args(sys.argv[2:])
        paid=json.loads(args.paid_report.read_text())
        contract=validate_contract(args.report,paid['candidate_module'])
        assert contract['numerical_report_sha256']==hashlib.sha256(args.paid_report.read_bytes()).hexdigest()
        assert contract['canister']==paid['inputs'][0]['whole_message_accounting']['candidate']['before']['canister']
        assert contract['completed_request_id']==paid['inputs'][0]['request_id']
        print('Verified local live refund, authorization, replay and upgrade evidence.')
    else:
        main()
