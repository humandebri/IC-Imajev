"""Client-selected exact query schedule for shorter text suffixes.

No host numerical inference: the client selects chunk dimensions and frames
canister-produced carry/state. Larger inputs retain the proven fixed route.
"""
import json, time
from dataclasses import dataclass
import numpy as np
from tail_inference import TailPrefixGraph
from joined_inference import C, H, KV
from delta_mlp_start_codec import NAME as START, CONV, encode_ids
from mlp_delta_carry import HUFFMAN_NAME, encode_request as finish
from mlp_attention_finish_codec import NAME as BRIDGE, FRONT, STREAM_COMPLETE, TERMINAL, encode_request as bridge

MEASURED_MODULE = '88507a95ea9e6e2111843c5139025d8ce2b3b8e21d9a9957beca4f6f74e103be'

@dataclass(frozen=True)
class ShortPlan:
    suffix: int
    prefix: int
    front: int
    heads: int
    plain_heads: int
    down: int = 1280
    version: str = 'short-query-v2'


def short_plan(n, p, module=MEASURED_MODULE):
    if type(n) is not int or type(p) is not int or n < 1 or not 1 <= p <= 132:
        raise ValueError('adaptive token bounds')
    # Measured with the pinned single-quad module. The full attention bridge
    # requires more budget for longer suffix/prefix; preserve the fixed route.
    if n > 69 or p > 27 or module != MEASURED_MODULE:
        return None
    front = min(7936, (4352 * 69 // n // 256) * 256)
    heads = min(30, max(20, (20 * 69 // n // 2) * 2))
    # Full MLP plus Delta projection costs more than completing a partial MLP.
    # Keep that query's head count lower; v1's 20 heads exceeded5B at n=69.
    plain_heads = min(30, max(8, (16 * 67 // n // 2) * 2))
    return ShortPlan(n, p, front, heads, plain_heads)


class AdaptivePrefixGraph(TailPrefixGraph):
    def forward(self, token_ids, layers=32):
        p = self.position_offset
        if layers != 32 or list(token_ids[:p]) != self.cache['metadata']['token_ids'] or not p < len(token_ids) <= 512:
            raise ValueError('adaptive prefix identity')
        ids = list(token_ids[p:]); plan = short_plan(len(ids), p, self.cache['metadata']['wasm_sha256'])
        self.adaptive_effective = plan is not None
        if plan is None:
            return super().forward(token_ids, layers)
        self.tail_effective = self.join_effective = self.roll_effective = True
        self.roll_blocks = False
        n, b, k, rows = plan.suffix, plan.front, plan.heads, plan.down
        path = self.t.directory / 'adaptive-plan.json'
        path.write_text(json.dumps(plan.__dict__, indent=2) + '\n')
        start, clock = len(self.t.measurements), time.perf_counter()
        z = self.cache['states'][0]
        x = self.send(0, 'delta_mlp_stream_start_ids', START, [n, b, p], encode_ids, ids, z['conv'], z['delta_log'])
        self.save_conv(0, x[-CONV:]); carry = x[:-CONV]
        layer, kind = 0, 'generation'
        while layer <= 30:
            if kind in ('generation', 'attention_pair'):
                if kind == 'generation' and layer % 4 == 2:
                    op = TERMINAL if layer == 30 else STREAM_COMPLETE
                    x = self.send(layer, op, BRIDGE, [n, p, b], bridge, carry, self.prefix_kv(layer+1))
                    if layer == 30:
                        if x.size != 2*C+n*KV:
                            raise ValueError('adaptive terminal shape')
                        self.save_kv(31, x[2*C:], n)
                        np.save(self.t.directory/'layer-31.npy', x[:C].reshape(1,C))
                        q = self.t.measurements[start:]
                        self.layers.extend([dict(layer=30,kind='delta',queries=len(q),instructions=sum(v['ok']['instructions'] for v in q),candid_bytes=sum(v['ok']['request_bytes']+v['ok']['reply_bytes'] for v in q),wall_seconds=time.perf_counter()-clock,replayed=sum(bool(v.get('replayed')) for v in q),hidden_exported=False),dict(layer=31,kind='full_attention',queries=0,instructions=0,candid_bytes=0,wall_seconds=0.,replayed=0,included_in_layer=30)])
                        (self.t.directory/'layers.json').write_text(json.dumps(self.layers,indent=2)+'\n')
                        return x[C:2*C].reshape(1,C)
                    self.mark(layer,x[:n*C].reshape(n,C),start,clock,next_attention_included=layer+1)
                    self.save_kv(layer+1,x[2*n*C:],n)
                    carry=x[:2*n*C]; layer+=1; kind='attention_pair'
                    start,clock=len(self.t.measurements),time.perf_counter()
                    continue
                plain = kind == 'attention_pair'
                pair_heads = plan.plain_heads if plain else k
                x,conv,cols,cv,log = self.complete_pair(layer,n,0 if plain else b,pair_heads,carry,start,clock,plain=plain)
                start,clock=len(self.t.measurements),time.perf_counter()
                terminal_front = layer + 1 == 30
                carry=self.follow_pair(layer,n,0 if plain else b,pair_heads,'delta_partial_mlp_front' if terminal_front else 'delta_partial_mlp_prepare_down',b if terminal_front else rows,x,conv,cols,cv,log)
                layer+=1;kind='generation' if terminal_front else 'down'
            else:
                if layer % 4 == 2:
                    x=self.send(layer,FRONT,BRIDGE,[n,p,rows,b],bridge,carry,self.prefix_kv(layer+1))
                    self.mark(layer,x[:n*C].reshape(n,C),start,clock,next_attention_included=layer+1)
                    self.save_kv(layer+1,x[-n*KV:],n)
                    carry=x[n*C:-n*KV]
                else:
                    z=self.cache['states'][layer+1]
                    x=self.send(layer,'mlp_finish_delta_log_mlp_front',HUFFMAN_NAME,[n,p,rows,b],finish,carry,z['conv'],z['delta_log'])
                    self.mark(layer,x[:n*C].reshape(n,C),start,clock,next_delta_included=layer+1)
                    self.save_conv(layer+1,x[-CONV:]);carry=x[n*C:-CONV]
                layer+=1;kind='generation'
                start,clock=len(self.t.measurements),time.perf_counter()
        raise AssertionError('adaptive graph did not reach terminal')
