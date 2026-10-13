"""Read retained paid DebugResult from immutable local IC heap checkpoints.

Verification only; captured overlays contain model data and must remain local.
Overlay V0 specification: DFINITY rs/replicated_state/src/page_map/storage.rs.
The supported Rust wasm32 layout is checked separately against the diagnostic
Candid getter; unsupported or ambiguous records are rejected.
"""
import bisect
import errno
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
from paid_instruction_accounting import decode_snapshot

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'rust-wasm32-paid-debug-v1'
HASHES, WIDTH = 32, 2560


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Heap:
    def __init__(self, directory, page_size):
        if page_size not in (4096, 16384):
            raise ValueError('Unsupported IC host page size')
        self.page_size, self.layers, self.base = page_size, [], None
        try:
            for path in sorted(directory.glob('*_vmemory_0.overlay'), reverse=True):
                if not re.fullmatch(r'[0-9a-f]{16}_[0-9a-f]{4}_vmemory_0\.overlay', path.name) or path.is_symlink():
                    raise ValueError('Unexpected heap overlay path')
                with path.open('rb') as stream:
                    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                self.layers.append((mapping, [], []))
                if len(mapping) < 12:
                    raise ValueError('Truncated heap overlay')
                pages, version = struct.unpack_from('<QI', mapping, len(mapping)-12)
                end = pages * page_size
                if version != 0 or end >= len(mapping)-12 or (len(mapping)-12-end) % 24:
                    raise ValueError('Unsupported heap overlay format')
                ranges = list(struct.iter_unpack('<QQQ', mapping[end:-12]))
                offset, previous = 0, 0
                for start, stop, physical in ranges:
                    if not previous <= start < stop <= 2**32//page_size or physical != offset:
                        raise ValueError('Invalid heap overlay ranges')
                    previous, offset = stop, offset + stop-start
                if offset != pages:
                    raise ValueError('Heap overlay page count mismatch')
                self.layers[-1] = mapping, ranges, [r[0] for r in ranges]
            base = directory/'vmemory_0.bin'
            if base.exists():
                if base.is_symlink() or base.stat().st_size % page_size or base.stat().st_size > 2**32:
                    raise ValueError('Invalid heap base')
                with base.open('rb') as stream:
                    self.base = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            if not self.layers and self.base is None:
                raise ValueError('Missing checkpoint heap')
        except Exception:
            self.close()
            raise

    def close(self):
        for mapping, _, _ in self.layers:
            mapping.close()
        if self.base is not None:
            self.base.close()

    def read(self, pointer, size, *, required=False):
        if not 0 <= pointer <= pointer+size <= 2**32:
            raise ValueError('Heap pointer outside wasm32 memory')
        result = bytearray()
        while size:
            page, offset = divmod(pointer, self.page_size)
            count = min(size, self.page_size-offset)
            data = None
            for mapping, ranges, starts in self.layers:
                index = bisect.bisect_right(starts, page)-1
                if index >= 0 and page < ranges[index][1]:
                    start, _, physical = ranges[index]
                    at = (physical+page-start)*self.page_size+offset
                    data = mapping[at:at+count]
                    break
            if data is None and self.base is not None and pointer+count <= len(self.base):
                data = self.base[pointer:pointer+count]
            if data is None:
                if required:
                    raise ValueError('Trace data missing from heap')
                data = bytes(count)
            result.extend(data); pointer += count; size -= count
        return bytes(result)


def read_state(directory, job_id, page_size):
    heap = Heap(directory, page_size)
    try:
        # Rust's pinned wasm32 linker uses a 1 MiB stack. Search static storage
        # only, avoiding stale temporary DebugResult copies on the stack.
        start = 1024*1024
        static = heap.read(start, 3*1024*1024)
        marker = struct.pack('<Q', job_id)
        offset, records, addresses = 0, {}, []
        while True:
            at = static.find(marker, offset)
            if at < 0:
                break
            offset = at+1
            if at % 8 or at+44 > len(static):
                continue
            try:
                vectors = [struct.unpack_from('<III', static, at+n) for n in (8,20,32)]
                strings = []
                for capacity, pointer, length in vectors[:2]:
                    if not HASHES == length <= capacity <= HASHES*2 or pointer < start or pointer % 4:
                        raise ValueError('Unsupported hash vector')
                    words = []
                    for cap, ptr, size in struct.iter_unpack('<III', heap.read(pointer, HASHES*12, required=True)):
                        if not size == 64 <= cap <= 128 or ptr < start:
                            raise ValueError('Unsupported hash string')
                        text = heap.read(ptr, size, required=True)
                        if any(c not in b'0123456789abcdef' for c in text):
                            raise ValueError('Invalid state hash')
                        words.append(text.decode())
                    strings.append(words)
                capacity, pointer, length = vectors[2]
                if not WIDTH == length <= capacity <= 2**30 or pointer < start or pointer % 4:
                    raise ValueError('Unsupported final hidden vector')
                raw = heap.read(pointer, WIDTH*4, required=True)
                if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)):
                    raise ValueError('Non-finite final hidden')
                value = dict(hidden_hashes=strings[0], state_hashes=strings[1], final_hidden_bits=raw.hex())
                records[json.dumps(value, sort_keys=True)] = value
                addresses.append(start+at)
            except (ValueError, struct.error):
                continue
        if len(records) != 1:
            raise ValueError(f'Expected one unambiguous paid state, found {len(records)}')
        return dict(schema=SCHEMA, job_id=job_id, addresses=addresses, **next(iter(records.values())))
    finally:
        heap.close()


def capture(meter, snapshot, result, directory, evidence_root):
    directory, evidence_root = Path(directory).resolve(), Path(evidence_root).resolve()
    try:
        relative = directory.relative_to(ROOT)
        if relative.parts[0] not in ('.cache', '.local-only'):
            raise ValueError('Model-bearing heap evidence must use an ignored local directory')
    except ValueError:
        if directory.is_relative_to(ROOT):
            raise
    source = meter.root / f'{snapshot["height"]:016x}' / 'canister_states' / meter.hex
    if hashlib.sha256((source/'canister.pbuf').read_bytes()).hexdigest() != snapshot['sha256']:
        raise ValueError('Heap checkpoint metadata differs from measured snapshot')
    directory.mkdir()
    paths = [source/'canister.pbuf', *source.glob('*_vmemory_0.overlay')]
    if (source/'vmemory_0.bin').exists():
        paths.append(source/'vmemory_0.bin')
    # Checkpoint files are immutable. Hard links retain them after replica GC
    # without copying multi-gigabyte weights for every input.
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Regular immutable checkpoint files required')
        destination = directory/path.name
        try:
            os.link(path, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            # Cross-volume verification retains identical immutable bytes.
            # Exclusive creation also prevents overwriting prior evidence.
            with path.open('rb') as original, destination.open('xb') as retained:
                shutil.copyfileobj(original, retained, length=1024*1024)
    files = {path.name: dict(bytes=(directory/path.name).stat().st_size, sha256=digest_file(directory/path.name)) for path in paths}
    page_size = os.sysconf('SC_PAGE_SIZE')
    state = read_state(directory, result['job_id'], page_size)
    return dict(directory=str(directory.relative_to(evidence_root)), page_size=page_size,
                snapshot=snapshot, files=files, state=state)


def validate(evidence_root, evidence, snapshot, job_id):
    relative = Path(evidence['directory'])
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Unsafe heap evidence directory')
    directory = Path(evidence_root)
    for part in relative.parts:
        directory = directory/part
        if directory.is_symlink():
            raise ValueError('Symlink in heap evidence path')
    files = evidence['files']
    if set(p.name for p in directory.iterdir()) != set(files) or 'canister.pbuf' not in files:
        raise ValueError('Heap evidence file coverage mismatch')
    for name, identity in files.items():
        if name not in ('canister.pbuf', 'vmemory_0.bin') and not re.fullmatch(r'[0-9a-f]{16}_[0-9a-f]{4}_vmemory_0\.overlay',name):
            raise ValueError('Unexpected heap evidence file')
        path = directory/name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != identity['bytes'] or digest_file(path) != identity['sha256']:
            raise ValueError('Heap evidence checksum/type mismatch')
    original = decode_snapshot((directory/'canister.pbuf').read_bytes(), snapshot['height'], snapshot['canister'], snapshot['module'])
    if original != snapshot or evidence['snapshot'] != snapshot:
        raise ValueError('Heap metadata differs from measured snapshot')
    state = read_state(directory, job_id, evidence['page_size'])
    if state != evidence['state']:
        raise ValueError('Paid state report differs from original heap')
    return state


def layout_files(report_path):
    report = json.loads(report_path.read_text())
    return {report_path.name, report['build_evidence']['file'], report['getter_evidence']['file'],
            *[str(Path(report['heap']['directory'])/name) for name in report['heap']['files']]}


def checked_json(root, evidence):
    name = Path(evidence['file'])
    if name.is_absolute() or not name.parts or '..' in name.parts:
        raise ValueError('Unsafe layout evidence path')
    path = root
    for part in name.parts:
        path = path/part
        if path.is_symlink():
            raise ValueError('Symlink in layout evidence')
    if digest_file(path) != evidence['sha256']:
        raise ValueError('Layout evidence checksum mismatch')
    return json.loads(path.read_text())


def validate_layout(report_path):
    root = report_path.resolve().parent
    report = json.loads(report_path.read_text())
    if report.get('complete') is not True or report.get('schema') != SCHEMA or report.get('parser_sha256') != digest_file(Path(__file__)):
        raise ValueError('Incomplete or stale state layout conformance')
    build = checked_json(root, report['build_evidence'])
    if not build.get('build_complete') or build.get('verification_only') is not True or 'paid-update-diagnostics' not in build['features']:
        raise ValueError('State layout requires source-built diagnostic module')
    snapshot = report['heap']['snapshot']
    if snapshot['module'] != build['module'] or report['module'] != build['module']:
        raise ValueError('Layout module identity mismatch')
    getter = checked_json(root, report['getter_evidence'])
    if getter['kind'] != 'paid_debug' or getter['result']['job_id'] != 2**64-1:
        raise ValueError('Missing typed diagnostic getter evidence')
    actual = getter['result']
    expected = dict(hidden_hashes=[f'{i:064x}' for i in range(32)], state_hashes=[f'{i:064x}' for i in range(32,64)],
        final_hidden_bits=b''.join(struct.pack('<f',-0.0 if i==0 else (i-128)/8) for i in range(WIDTH)).hex())
    decoded = validate(root, report['heap'], snapshot, 2**64-1)
    typed = dict(hidden_hashes=actual['hidden_hashes'],state_hashes=actual['state_hashes'],
                 final_hidden_bits=b''.join(struct.pack('<f',f) for f in actual['final_hidden']).hex())
    if typed != expected or any(decoded[k] != typed[k] for k in typed):
        raise ValueError('Heap decoder differs from typed Candid getter')
    return build
