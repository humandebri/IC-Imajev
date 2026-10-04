#!/usr/bin/env python3
"""New tail request directions, exact byte planes and terminal checkpoint identity."""
import json,pathlib,struct,sys,tempfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'client'),str(ROOT/'scripts')]
from mlp_attention_finish_codec import NAME as BRIDGE,STREAM_COMPLETE,TERMINAL,encode_request as bridge,decode_reply
from mlp_delta_stream_codec import NAME as PAIR,encode_request as pair,encode_continue_request as follow
from mlp_stream_codec import NAME as STREAM,C,H,decode_payload
from test_mlp_stream_codec import carry
from test_mlp_delta_stream_codec import reply
from full_inference import JournalTransport
from transport import encode,decode,frame_digest

def header(op,codec,n,layer,dims):
    return dict(version=3,model='a'*64,pack_hash='b'*64,input_hash='c'*64,step=0,op=op,encoding=codec,
        tensor=f'model.language_model.layers.{layer}.post_attention_layernorm.weight',
        aux=[f'model.language_model.layers.{layer+1}.input_layernorm.weight'],dims=dims,scalars=[2.,1e-6])
def payload(frame):return frame[4+int.from_bytes(frame[:4],'little'):-32]
def plane(data,count):
    # Independent canonical decoder: verifies actual hidden bytes, including -0.
    tag,size=struct.unpack('<BI',data[:5]);offset=5
    if tag==0:return data[offset:offset+size],offset+size
    if tag==2:return data[offset:offset+1]*count,offset+1
    if tag!=1:raise AssertionError('unexpected plane direction')
    lengths=data[offset:offset+256];offset+=256;code=0;previous=0;lookup={}
    for symbol,length in sorted(enumerate(lengths),key=lambda x:(x[1],x[0])):
        if not length:continue
        code <<= length-previous;lookup[length,code]=symbol;code+=1;previous=length
    result=bytearray();current=0;length=0
    for byte in data[offset:offset+size]:
        for bit in range(7,-1,-1):
            current=(current<<1)|((byte>>bit)&1);length+=1
            if (length,current)in lookup:
                result.append(lookup[length,current]);current=length=0
                if len(result)==count:return bytes(result),offset+size
    raise AssertionError('short plane')

class TailCodecTests(unittest.TestCase):
    def test_hidden_planes_reconstruct_old_packet_without_changing_other_bytes(self):
        for n in [1,87,89]:
            k=10;h=header('delta_partial_mlp_front',PAIR,n,25,[n,1024,H-1024,k,45,6912]);_,v=reply(n,k)
            v[:n*C:3]=0.;v[1:n*C:3]=1.;v[2:n*C:3]=-0.
            history=np.zeros(3*(32-k)*256,np.float32);log=np.zeros(45*((32-k)//2*128+(32-k)*128+32-k),np.float32)
            old=payload(follow(h,v,history,log,compress_base=True,compress_prefix=True))
            frame=follow(h,v,history,log,compress_base=True,compress_prefix=True,compress_hidden=True);new=payload(frame)
            self.assertEqual(new[0],12);size=int.from_bytes(new[1:5],'little');packed=new[5:5+size]
            low,end=plane(packed,n*C);high,end2=plane(packed[end:],n*C);self.assertEqual(end+end2,size)
            raw=np.stack([np.frombuffer(low,dtype=np.uint8),np.frombuffer(high,dtype=np.uint8)],axis=1).tobytes()
            self.assertEqual(b'\12'+new[5+size:9+size]+raw+new[9+size:],old)
            self.assertLess(len(frame),2_000_000)
            for kwargs in [{},{'compress_base':True},{'compress_prefix':True}]:
                with self.assertRaises(ValueError):follow(h,v,history,log,compress_hidden=True,**kwargs)

    def test_plain_full_mlp_uses_bf16_input_and_rejects_bad_lanes(self):
        for n in [1,87,89]:
            h=header('mlp_full_delta_partial',PAIR,n,27,[n,0,H,8,45]);v=np.full(2*n*C,-0.,np.float32)
            cv=np.zeros(3*8*256,np.float32);log=np.zeros(45*(4*128+8*128+8),np.float32)
            packet=pair(h,v,cv,log);p=payload(packet);self.assertEqual(p[0],13);self.assertEqual(int.from_bytes(p[1:5],'little'),4*n*C)
            self.assertEqual((np.frombuffer(p[5:5+4*n*C],dtype='<u2').astype('<u4')<<16).view('<f4').tobytes(),v.tobytes())
            for value in [.1234567,np.inf,np.nan]:
                bad=v.copy();bad[0]=value
                with self.assertRaises(ValueError):pair(h,bad,cv,log)
            with self.assertRaises(ValueError):pair(dict(h,dims=[n,128,H-128,8,45]),v,cv,log)
            with self.assertRaises(ValueError):pair(h,v,cv,log,compress_residual=True)

    def test_stream_request_progress_and_distinct_terminal_reply_shape(self):
        for n in [1,87,89]:
            for op,layer in [(STREAM_COMPLETE,26),(TERMINAL,30)]:
                h=header(op,BRIDGE,n,layer,[n,45,6912]);v=carry(n,6912);kv=np.full(45*2048,-0.,np.float32)
                p=payload(bridge(h,v,kv));self.assertEqual(p[0],3);end=5+int.from_bytes(p[1:5],'little')
                inner=dict(h,op='mlp_stream_complete',encoding=STREAM,dims=[n,6912,H-6912])
                self.assertEqual(decode_payload(inner,p[5:end]).tobytes(),v.tobytes());self.assertEqual(p[end:],(kv.view('<u4')>>16).astype('<u2').tobytes())
                count=2*C+n*2048 if op==TERMINAL else n*(2*C+2048);y=np.full(count,-0.,np.float32);raw=b'\0'+(y.view('<u4')>>16).astype('<u2').tobytes()
                self.assertEqual(decode_reply(h,raw).tobytes(),y.tobytes())
                for bad in [raw[:-1],raw+b'\0',b'\3'+raw[1:]]:
                    with self.assertRaises(ValueError):decode_reply(h,bad)
                for dims in [[n,45,0],[n,45,H],[n,45,6913],[n,45,6656]]:
                    with self.assertRaises(ValueError):bridge(dict(h,dims=dims),v,kv)
                with self.assertRaises(ValueError):bridge(h,carry(n,H),kv)

    def test_terminal_stream_replay_binds_option_order_and_mode(self):
        with tempfile.TemporaryDirectory()as name:
            directory=pathlib.Path(name);calls=[];h=header(TERMINAL,BRIDGE,1,30,[1,45,2560]);packet=bridge(h,carry(1,2560),np.zeros(45*2048,np.float32))
            def transport(options=('yes','no'),fused=True):
                t=JournalTransport.__new__(JournalTransport);t.directory=directory;t.index=0;t.measurements=[];t.replayed=0;t.model=h['model'];t.pack_hash=h['pack_hash'];t.input_hash=h['input_hash'];t.fuse_terminal_decision=fused;t.decision_options=list(options)
                def command(cmd):
                    calls.append(cmd);self.assertEqual(cmd['op'],'terminal_step_decision');self.assertEqual(cmd['options'],list(options))
                    rh=dict(h,step=1);encoded=json.dumps(rh,separators=(',',':')).encode();y=np.full(2*C+2048,-0.,np.float32)
                    body=struct.pack('<I',len(encoded))+encoded+b'\0'+(y.view('<u4')>>16).astype('<u2').tobytes()
                    pathlib.Path(cmd['output']).write_bytes(body+frame_digest(rh,body))
                    return dict(ok=dict(instructions=100,stable_read_bytes=0,request_bytes=10,reply_bytes=20,decision=dict(value='yes')),wall_seconds=.01)
                t.command=command;return t
            first=transport();y=first.run_encoded(h,packet);replay=transport();self.assertEqual(replay.run_encoded(h,packet).tobytes(),y.tobytes());self.assertEqual(replay.terminal_decision,first.terminal_decision);self.assertEqual(len(calls),1)
            for options,fused in [(('no','yes'),True),(('yes','no'),False)]:
                with self.assertRaisesRegex(ValueError,'checkpoint'):transport(options,fused).run_encoded(h,packet)
            metric=directory/'000000.metric.json';saved=json.loads(metric.read_text());del saved['ok']['decision'];metric.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,'checkpoint'):transport().run_encoded(h,packet)
            self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
