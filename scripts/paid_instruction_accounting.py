"""Read full replicated-execution counters from immutable local IC checkpoints.

Schema: DFINITY CanisterStateBits fields 58..63 and ExecutionStateBits field 6.
These are replica counters, not checkpoints inside a canister function.
https://github.com/dfinity/ic/blob/master/rs/protobuf/def/state/canister_state_bits/v1/canister_state_bits.proto
"""
import base64
import hashlib
from pathlib import Path
import time
import zlib
import struct
import threading

# WORKER_BUDGET (30B) is checked before starting another stage, so a worker
# can exceed it while finishing that stage. The existing IC update message
# limit is 40B: https://docs.internetcomputer.org/references/resource-limits/
MESSAGE_INSTRUCTION_LIMIT = 40_000_000_000


def protobuf_fields(data):
    fields, offset = {}, 0

    def varint():
        nonlocal offset
        value = 0
        for shift in range(0, 70, 7):
            if offset >= len(data):
                raise ValueError('Truncated protobuf varint')
            byte = data[offset]; offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                if value >= 1 << 64:
                    raise ValueError('Protobuf integer overflow')
                return value
        raise ValueError('Invalid protobuf varint')

    while offset < len(data):
        tag = varint(); number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError('Invalid protobuf field')
        if wire == 0:
            value = varint()
        elif wire in (1, 2, 5):
            size = varint() if wire == 2 else (8 if wire == 1 else 4)
            end = offset + size
            if end > len(data):
                raise ValueError('Truncated protobuf field')
            value, offset = data[offset:end], end
        else:
            raise ValueError('Unsupported protobuf wire type')
        fields.setdefault(number, []).append(value)
    return fields


def single(fields, number, default=None):
    values = fields.get(number, [default])
    if len(values) != 1:
        raise ValueError('Duplicate singular protobuf field')
    return values[0]


def principal_hex(text):
    encoded = text.replace('-', '').upper()
    raw = base64.b32decode(encoded + '=' * (-len(encoded) % 8))
    if len(raw) < 5 or raw[:4] != zlib.crc32(raw[4:]).to_bytes(4, 'big'):
        raise ValueError('Invalid principal checksum')
    return raw[4:].hex()


def decode_snapshot(raw, height, canister, module):
    fields = protobuf_fields(raw)
    execution = protobuf_fields(single(fields, 7, b''))
    if single(execution, 6, b'').hex() != module:
        raise ValueError('Checkpoint module mismatch')
    result = dict(height=height, canister=canister, module=module,
                  canister_version=single(fields, 34, 0), sha256=hashlib.sha256(raw).hexdigest())
    for number, name in [(58, 'ingress'), (59, 'remote'), (60, 'local'), (61, 'http'), (62, 'timers'), (63, 'instructions')]:
        value = single(fields, number, 0)
        if type(value) is not int:
            raise ValueError('Unexpected replica counter wire type')
        result[name] = value
    if result['instructions'] == 0:
        raise ValueError('Replica instruction counter unavailable')
    tasks = protobuf_fields(single(fields, 54, b''))
    task = single(tasks, 1)
    if tasks.get(3):
        raise ValueError('Unexpected queued execution tasks')
    if task is not None:
        task = protobuf_fields(task)
        if set(task) != {1}:
            raise ValueError('Unexpected checkpoint execution task')
        aborted = protobuf_fields(single(task, 1))
        if set(aborted) - {1, 7} or 1 not in aborted:
            raise ValueError('Only aborted self-worker requests are supported')
        request = protobuf_fields(single(aborted, 1))
        def canister_id(number):
            wrapped = protobuf_fields(single(request, number, b''))
            principal = protobuf_fields(single(wrapped, 1, b''))
            return single(principal, 1, b'').hex()
        result['aborted_request'] = dict(receiver=canister_id(1), sender=canister_id(2),
            callback=single(request, 3, 0), method=single(request, 5, b'').decode(),
            payload=single(request, 6, b'').hex())
    return result


class CheckpointTrace:
    """Keep original pending retry tasks while the measured call is running."""
    def __init__(self, meter, before, directory, evidence_root):
        self.meter, self.before = meter, before
        self.directory, self.evidence_root = Path(directory), Path(evidence_root).resolve()
        self.records, self.heights, self.error = [], set(), None
        self.stop = threading.Event()

    def __enter__(self):
        self.directory.mkdir()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        return self

    def monitor(self):
        try:
            while not self.stop.is_set():
                raw, snapshot = self.meter.latest()
                height = snapshot['height']
                if height > self.before['height'] and height not in self.heights:
                    self.heights.add(height)
                    if 'aborted_request' in snapshot:
                        path = self.directory / f'{height:016x}.pbuf'
                        path.write_bytes(raw)
                        snapshot['evidence_file'] = str(path.resolve().relative_to(self.evidence_root))
                        self.records.append(snapshot)
                self.stop.wait(0.1)
        except Exception as error:
            self.error = error

    def __exit__(self, *args):
        self.stop.set(); self.thread.join()
        if self.error:
            raise self.error


class CheckpointMeter:
    def __init__(self, state_directory, canister, module, timeout=60):
        self.root = Path(state_directory).resolve() / 'checkpoints'
        self.canister, self.module, self.timeout = canister, module, timeout
        self.hex = principal_hex(canister)

    def latest(self):
        heights = sorted((p for p in self.root.iterdir() if p.is_dir() and len(p.name) == 16), reverse=True)
        for directory in heights:
            path = directory / 'canister_states' / self.hex / 'canister.pbuf'
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                continue
            return raw, decode_snapshot(raw, int(directory.name, 16), self.canister, self.module)
        raise ValueError('No immutable checkpoint for selected canister')

    def fresh(self, destination, *, after=None, worker_count=None):
        # Waiting past the current height prevents a completed call from being
        # attributed to a stale disk snapshot. No calls to this canister may run
        # between the before snapshot and the measured inference except that call.
        previous = after['height'] if after else self.latest()[1]['height']
        expected_version = after['canister_version'] + 1 + 2 * worker_count if after else None
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            raw, snapshot = self.latest()
            if snapshot['height'] > previous:
                if expected_version is not None:
                    if snapshot['canister_version'] > expected_version:
                        raise ValueError('Unexpected canister mutation before completed checkpoint')
                    if snapshot['canister_version'] < expected_version:
                        time.sleep(0.25)
                        continue
                Path(destination).write_bytes(raw)
                return snapshot
            time.sleep(0.25)
        raise TimeoutError('Fresh IC checkpoint not available')


def account(before, after, workers, *, job_id=None, retries=()):
    if 'aborted_request' in before or 'aborted_request' in after:
        raise ValueError('Measurement boundary contains unfinished worker')
    for key in ('canister', 'module'):
        if before[key] != after[key]:
            raise ValueError('Validation target changed during measurement')
    if after['height'] <= before['height']:
        raise ValueError('Stale checkpoint measurement')
    delta = {key: after[key] - before[key] for key in ('ingress', 'remote', 'local', 'http', 'timers', 'instructions')}
    if any(value < 0 for value in delta.values()) or any(delta[key] for key in ('ingress', 'remote', 'http', 'timers')):
        raise ValueError('Unexpected execution or counter reset during measurement')
    # One relay request, one self request and one callback per worker. Replica
    # LoadMetrics count both local request and response messages.
    retry_keys = set()
    for retry in retries:
        if retry['module'] != before['module'] or retry['canister'] != before['canister'] or not before['height'] < retry['height'] <= after['height']:
            raise ValueError('Retry checkpoint outside measured target/window')
        for key in ('ingress', 'remote', 'http', 'timers'):
            if retry[key] != before[key]:
                raise ValueError('Unexpected execution in retry checkpoint')
        if not all(before[key] <= retry[key] <= after[key] for key in ('instructions', 'local', 'canister_version')):
            raise ValueError('Retry counters outside measured window')
        request = retry['aborted_request']
        target = principal_hex(before['canister'])
        if request['sender'] != target or request['receiver'] != target or request['method'] != 'runPaidInferenceWorker':
            raise ValueError('Retry was not a self-worker request')
        payload = bytes.fromhex(request['payload'])
        if len(payload) != 24 or payload[:8] != b'DIDL\x00\x02\x78\x78':
            raise ValueError('Unsupported retry worker Candid')
        job, stage = struct.unpack('<QQ', payload[8:])
        if job != job_id or stage not in {0, *[worker['stage'] for worker in workers[:-1]]}:
            raise ValueError('Retry belongs to another job/stage')
        retry_keys.add((request['callback'], request['payload'], retry['local']))
    expected = 1 + 2 * len(workers)
    if delta['local'] != expected + len(retry_keys):
        raise ValueError('Unexpected local message count during measurement')
    if after['canister_version'] - before['canister_version'] != expected:
        raise ValueError('Unexpected canister mutation during measurement')
    checkpoints = [worker['instructions'] for worker in workers]
    if not checkpoints or any(type(n) is not int or n <= 0 for n in checkpoints):
        raise ValueError('Invalid worker checkpoint counters')
    overhead = delta['instructions'] - sum(checkpoints)
    if overhead < 0:
        raise ValueError('Worker checkpoints exceed replica instruction total')
    # Every worker checkpoint is a lower bound for that complete message.
    # All unmeasured instructions across *all* messages provide a conservative
    # upper bound for the omitted part of any one worker or callback/admission.
    bound = max(checkpoints) + overhead
    result = dict(total_instructions=delta['instructions'], local_messages=delta['local'],
                unmeasured_instructions=overhead, max_message_instructions_upper_bound=bound,
                message_budget_passed=bound <= MESSAGE_INSTRUCTION_LIMIT, before=before, after=after)
    if retries:
        result.update(retry_evidence=list(retries), aborted_worker_attempts=len(retry_keys))
    return result
