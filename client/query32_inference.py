"""Experimental four-query blocks. Numerical work stays in ordinary queries."""
import hashlib,json,time
import numpy as np
from packed_inference import PackedPrefixGraph
from joined_inference import C,H,KV
from delta_mlp_start_codec import NAME as START,CONV,encode_ids
from mlp_stream_codec import NAME as STREAM
from mlp_attention_finish_codec import NAME as BRIDGE,STREAM_COMPLETE,TERMINAL,encode_request as bridge
from transport import encode,decode,atomic
MODULE='f6b399b6adb4e30871fef9833b89c766f099dc6cf97f736bd4e8f98464d9cf0b'

class Query32PrefixGraph(PackedPrefixGraph):
    front_rows=4096
    def attention_front(self,layer,n,begin,next_front,carry):
        header=self.header(STREAM_COMPLETE,BRIDGE,[n,self.position_offset,begin],layer)
        data=bridge(header,carry,self.prefix_kv(layer+1));index=self.t.index;d=self.t.directory
        request=d/f'{index:06d}.request.bin';expected=d/f'{index:06d}.expected.bin';output=d/f'{index:06d}.response.bin';hidden=d/f'{index:06d}.previous.bf16';kv=d/f'{index:06d}.kv.bf16'
        atomic(request,data)
        target=self.header('mlp_stream_prepare',STREAM,[n,0,next_front],layer+1)
        atomic(expected,encode(target,np.zeros(2*n*C,dtype=np.float32)))
        started=time.perf_counter()
        try:
            result=self.t.command(dict(op='attention_mlp_front',input=str(request),expected=str(expected),output=str(output),hidden=str(hidden),kv=str(kv),front=next_front))
        except Exception as e:
            with (d/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(index=index,op='attention_mlp_front',tensor=header['tensor'],error=str(e),wall_seconds=time.perf_counter()-started))+'\n')
            raise
        returned,values=decode(output.read_bytes())
        assert returned['step']==index+1 and all(returned[k]==v for k,v in target.items() if k not in ('step','scalars'))
        assert np.array(returned['scalars'],dtype='<f4').tobytes()==np.array(target['scalars'],dtype='<f4').tobytes()
        result.update(index=index,op='attention_mlp_front',tensor=header['tensor'],outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [output,hidden,kv]})
        self.t.measurements.append(result);self.t.index+=1;atomic(d/f'{index:06d}.metric.json',(json.dumps(result,indent=2)+'\n').encode())
        def bf16(p):return (np.frombuffer(p.read_bytes(),dtype='<u2').astype('<u4')<<16).view('<f4')
        previous,keys=bf16(hidden),bf16(kv)
        if previous.size!=n*C or keys.size!=n*KV:raise ValueError('attention front output shape')
        return values,previous,keys
    def forward(self,token_ids,layers=32):
        p=self.position_offset
        if layers!=32 or list(token_ids[:p])!=self.cache['metadata']['token_ids'] or not p<len(token_ids)<=512:raise ValueError('query32 prefix identity')
        n=len(token_ids)-p;b=self.front_rows
        if not 1<=n<=59 or p!=27 or b<=0 or b>=H or b%256:raise ValueError('query32 experimental input bounds')
        self.adaptive_effective=self.tail_effective=self.join_effective=self.roll_effective=True;self.roll_blocks=False
        atomic(self.t.directory/'adaptive-plan.json',(json.dumps(dict(version='query32-v1',suffix=n,prefix=p,front=b,expected_queries=32,module=MODULE),indent=2)+'\n').encode())
        query_start=len(self.t.measurements)
        start,clock=query_start,time.perf_counter();z=self.cache['states'][0]
        x=self.send(0,'delta_mlp_stream_start_ids',START,[n,b,p],encode_ids,list(token_ids[p:]),z['conv'],z['delta_log'])
        self.save_conv(0,x[-CONV:]);carry=x[:-CONV]
        for layer in range(31):
            if layer==30:
                x=self.send(layer,TERMINAL,BRIDGE,[n,p,b],bridge,carry,self.prefix_kv(31))
                if x.size!=2*C+n*KV:raise ValueError('query32 terminal shape')
                self.save_kv(31,x[2*C:],n);np.save(self.t.directory/'layer-31.npy',x[:C].reshape(1,C))
                self.layers.extend([dict(layer=30,kind='delta',hidden_exported=False),dict(layer=31,kind='full_attention',included_in_layer=30)])
                atomic(self.t.directory/'layers.json',(json.dumps(self.layers,indent=2)+'\n').encode())
                assert len(self.t.measurements)-query_start==32
                return x[C:2*C].reshape(1,C)
            if layer%4==2:
                carry,previous,keys=self.attention_front(layer,n,b,b,carry)
                self.mark(layer,previous.reshape(n,C),start,clock,next_attention_included=layer+1)
                self.save_kv(layer+1,keys,n)
            else:
                carry,previous,conv=self.chain(layer,n,b,b,carry)
                self.mark(layer,previous.reshape(n,C),start,clock,next_delta_included=layer+1)
                self.save_conv(layer+1,conv)
            start,clock=len(self.t.measurements),time.perf_counter()
        raise AssertionError('query32 terminal missing')
