"""Restart bounded experimental inference in an isolated standard query journal."""
import hashlib,json,pathlib,subprocess,sys,time
from transport import atomic,is_instruction_limit

FRAME_ERRORS={'pair continuation frame bounds','pair frame bounds','MLP attention frame bounds','attention MLP frame bounds','Delta MLP start frame bounds','Delta MLP ids frame bounds','carry frame size'}
def eligible(error):
    return (type(error)is ValueError and str(error)in FRAME_ERRORS) or (type(error)is RuntimeError and is_instruction_limit(error))

def standard_command(argv,directory):
    flags={'--adaptive-start','--packed-start','--tail-start','--join-start','--roll-start'}
    values={'--directory','--tail-heads28','--tail-front28','--tail-heads29','--tail-front30','--tail-down29','--roll-begin','--roll-heads','--roll-down'}
    result=[];i=0
    while i<len(argv):
        key=argv[i].split('=',1)[0]
        if key in flags:i+=1;continue
        if key in values:
            if '='not in argv[i]:i+=1
            i+=1;continue
        result.append(argv[i]);i+=1
    return [sys.executable,str(pathlib.Path(__file__).resolve().parents[1]/'scripts/run_prefix_canister.py'),*result,'--directory',str(directory)]

def restart(directory,transport,argv,error=None,primary_seconds=0.):
    directory=pathlib.Path(directory);marker=directory/'fallback.json';session_bytes=(directory/'session.json').read_bytes();session=json.loads(session_bytes);session_hash=hashlib.sha256(session_bytes).hexdigest()
    child=directory/'standard-fallback';command=standard_command(argv,child.resolve())
    if marker.exists():
        event=json.loads(marker.read_text())
        if event['session_sha256']!=session_hash or event['command']!=command:raise ValueError('fallback identity mismatch')
        primary_executed=[];failed_this_run=0
    else:
        if error is None or not eligible(error):raise ValueError('unsupported fallback reason')
        sent=type(error)is RuntimeError
        request=transport.directory/f'{transport.index:06d}.request.bin'
        event=dict(version=1,decision_options=list(transport.decision_options),session_sha256=session_hash,command=command,reason=str(error),stage='query'if sent else'frame',primary_queries=transport.measurements,failed_query_count=int(sent),failed_request_file=str(request)if sent else None,failed_request_frame_bytes=request.stat().st_size if sent and request.exists()else None)
        atomic(marker,(json.dumps(event,indent=2)+'\n').encode())
        primary_executed=[q for q in transport.measurements if not q.get('replayed')];failed_this_run=int(sent)
    def child_failures():
        path=child/'queries/failures.jsonl'
        return [json.loads(line)for line in path.read_text().splitlines()if line]if path.exists()else []
    failures_before=len(child_failures());started=time.perf_counter()
    with (directory/'fallback.log').open('a')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((child/'report.json').read_text())
    for key in ['model','pack_hash','input_hash','wasm_sha256','prefix_identity']:
        if report[key]!=session[key]:raise ValueError('fallback result identity mismatch: '+key)
    from decision_validation import validate_decision
    validate_decision(report['decision_query']['ok']['decision'],event['decision_options'])
    if report['queries'][-1].get('decision_options')!=event['decision_options']:raise ValueError('fallback decision options mismatch')
    failures=child_failures();new_failures=len(failures)-failures_before
    primary=event['primary_queries'];failed=event['failed_query_count']+len(failures)
    known_instructions=sum(q['ok']['instructions']for q in primary)+report['total_instructions']
    known_bytes=sum(q['ok']['request_bytes']+q['ok']['reply_bytes']for q in primary)+report['total_candid_bytes']
    report.update(wall_seconds_this_run=primary_seconds+time.perf_counter()-started,scope='Experimental limit fallback; standard restart from token IDs',fallback=event,standard_failed_queries=failures,successful_run_directory=str(child.resolve()),query_count=len(primary)+failed+report['query_count'],completed_query_count=len(primary)+report['query_count'],successful_query_instructions=known_instructions,successful_query_candid_bytes=known_bytes,total_instructions=None if failed else known_instructions,total_candid_bytes=None if failed else known_bytes,unmeasured_failed_queries=failed,executed_query_count=len(primary_executed)+failed_this_run+new_failures+report['executed_query_count'],executed_instructions=None if failed_this_run or new_failures else sum(q['ok']['instructions']for q in primary_executed)+report['executed_instructions'],executed_candid_bytes=None if failed_this_run or new_failures else sum(q['ok']['request_bytes']+q['ok']['reply_bytes']for q in primary_executed)+report['executed_candid_bytes'],fallback_seconds_this_run=time.perf_counter()-started,max_successful_query_instructions=max([report['max_query_instructions']]+[q['ok']['instructions']for q in primary]),max_query_instructions=None if failed else max([report['max_query_instructions']]+[q['ok']['instructions']for q in primary]),max_observed_heap_bytes=max([report['max_observed_heap_bytes']]+[q['ok']['heap_pages']*65536 for q in primary]))
    report.pop('end_to_end_seconds_excluding_process_startup',None)
    atomic(directory/'report.json',(json.dumps(report,indent=2)+'\n').encode());return report
