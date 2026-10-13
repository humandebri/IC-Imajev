"""Check the actual query/update exports, not just the Candid declarations."""
import argparse
import json
from pathlib import Path
from canister_api_names import METHOD_NAMES, PRODUCTION_NAMES, DIAGNOSTIC_METHODS

QUERY_NAMES = {'runInferenceStep', 'runDeltaInferenceStep', 'runAttentionInferenceStep',
               'runFinalInferenceStep', 'getModelStatus', 'getWeightCacheStatus',
               'getInferenceQuote', 'getInferenceReceipt', 'getPaidInferenceConfig',
               'profileInferenceStep', 'getPaidInferenceDebug'}
BASE_NAMES = {'prepareModelUpload', 'uploadModelChunk', 'verifyModelUpload',
              'getModelStatus', 'runInferenceStep', 'runFinalInferenceStep',
              'prepareWeightCache', 'clearWeightCache', 'getWeightCacheStatus'}


def exported_methods(path):
    data = Path(path).read_bytes()
    if data[:8] != b'\0asm\x01\0\0\0':
        raise ValueError('expected a WebAssembly module')

    def leb(pos):
        value = shift = 0
        while True:
            byte = data[pos]; pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, pos
            shift += 7
            if shift > 35:
                raise ValueError('invalid wasm length')

    result = {}
    pos = 8
    while pos < len(data):
        section = data[pos]; size, start = leb(pos + 1)
        if section == 7:
            count, p = leb(start)
            for _ in range(count):
                length, p = leb(p)
                name = data[p:p + length].decode(); p += length
                kind = data[p]; p += 1
                _, p = leb(p)
                for mode in ['query', 'update', 'composite_query']:
                    prefix = 'canister_' + mode + ' '
                    if name.startswith(prefix):
                        if kind != 0:
                            raise ValueError('canister entrypoint is not a function')
                        result[name[len(prefix):]] = mode
        pos = start + size
    return result


def check_exports(path, profile='optimized', diagnostics=False):
    expected = set(BASE_NAMES if profile == 'base' else PRODUCTION_NAMES)
    if profile == 'cargo':
        expected -= {'runAttentionInferenceStep', 'prepareFixedPrefixCache'}
    if diagnostics:
        expected |= {METHOD_NAMES[x] for x in DIAGNOSTIC_METHODS}
    actual = exported_methods(path)
    if set(actual) != expected:
        raise AssertionError({'missing': sorted(expected - actual.keys()),
                              'unexpected': sorted(actual.keys() - expected)})
    for name, mode in actual.items():
        assert mode == ('query' if name in QUERY_NAMES else 'update'), (name, mode)
    return {'methods': len(actual), 'profile': profile, 'diagnostics': diagnostics,
            'exports': actual}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wasm', type=Path)
    parser.add_argument('--profile', choices=['optimized', 'cargo', 'base'], default='optimized')
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_exports(args.wasm, args.profile, args.diagnostics), sort_keys=True))
