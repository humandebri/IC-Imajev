"""Lossless byte framing of canister-produced partial MLP carry; no host inference."""
import json,struct,zlib
import numpy as np
NAME='mlp-delta-log-carry-exact-v1'
HUFFMAN_NAME='mlp-delta-huffman-carry-exact-v1'
def planar(raw,width):
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,width).T.copy().tobytes()
def encode_request(header,state,conv=None,log=None,raw_threshold=0.):
    if not 0<=raw_threshold<=1:raise ValueError("carry raw threshold")
    from transport import frame_digest
    dims=header['dims']
    front=header['op']=='mlp_finish_delta_log_mlp_front'
    if front and (len(dims)!=4 or type(dims[3])is not int or not 0<dims[3]<9216 or dims[3]%256):raise ValueError('carry front rows')
    if len(dims)not in ((4,)if front else(2,3))or any(type(v)is not int for v in dims)or len(dims)>=3 and not (0<dims[2]<2560 and dims[2]%32==0):raise ValueError('carry partial rows')
    n,p=dims[:2];fused=header['op']in ('mlp_finish_delta_log_integer','mlp_finish_delta_log_mlp_front')
    if header['encoding'] not in (NAME,HUFFMAN_NAME) or not 1<=n<=89 or not 0<=p<=132 or (fused and p==0) or (not fused and (header['op']!='mlp_finish_partial_integer' or p!=0)):raise ValueError('carry metadata')
    if fused and n*2560+24576+p*6176>900000:raise ValueError('carry Delta input bounds')
    values=np.asarray(state,dtype='<f4').ravel();c=n*2560;q=n*9216
    if values.size!=c+q+n*100 or not np.isfinite(values).all() or np.any(values[:c].view('<u4')&65535):raise ValueError('carry state shape/finite')
    ints=values[c:c+q]
    if np.any(ints<-127) or np.any(ints>127) or np.any(ints!=np.trunc(ints)) or np.any(ints.view('<u4')==0x80000000) or np.any(values[c+q:c+q+n*36]<=0):raise ValueError('carry integer/scales')
    streams=[planar((values[:c].view('<u4')>>16).astype('<u2').tobytes(),2),ints.astype(np.int8).tobytes(),planar(values[c+q:].tobytes(),4)]
    if fused:
        cv=np.asarray(conv,dtype='<f4').ravel();lv=np.asarray(log,dtype='<f4').ravel()
        if cv.size!=24576 or lv.size!=p*6176 or not np.isfinite(cv).all() or not np.isfinite(lv).all() or np.any(cv.view('<u4')&65535) or np.any(lv[:p*2048].view('<u4')&65535) or np.any(lv[p*6144:]<0) or np.any(lv[p*6144:]>1):raise ValueError('carry prefix state')
        streams += [planar((cv.view('<u4')>>16).astype('<u2').tobytes(),2),planar((lv[:p*2048].view('<u4')>>16).astype('<u2').tobytes(),2),planar(lv[p*2048:].tobytes(),4)]
    else:streams += [b'',b'',b'']
    packed=[zlib.compress(s,1)if s else b''for s in streams] if header['encoding']==NAME else []
    h=json.dumps(header,separators=(',',':'),allow_nan=False).encode()
    if header['encoding']==HUFFMAN_NAME:
        parts=[]
        for stream,width in zip(streams,[2,1,4,2,2,4]):
            count=len(stream)//width
            parts.extend(encode_plane(stream[b*count:(b+1)*count],raw_threshold) for b in range(width))
        payload=b'\1'+b''.join(parts)
    else:payload=b'\1'+struct.pack('<6I',*(len(s)for s in packed))+b''.join(packed)
    body=struct.pack('<I',len(h))+h+payload
    if len(h)>16384 or len(body)+32>2000000:raise ValueError('carry frame size')
    return body+frame_digest(header,body)


def encode_dictionary_plane(raw):
    import collections
    if not raw:return b'\0'+struct.pack('<I',0)
    table=bytes(s for s,_ in collections.Counter(raw).most_common(15));lookup={s:i for i,s in enumerate(table)}
    codes=bytearray((len(raw)+1)//2);escape=bytearray()
    for i,s in enumerate(raw):
        code=lookup.get(s,15);codes[i//2]|=code<<(4*(i%2))
        if code==15:escape.append(s)
    payload=bytes([len(table)])+table+codes+escape
    return b'\3'+struct.pack('<I',len(payload))+payload

def encode_plane(raw,raw_threshold=0.):
    import collections,heapq
    freq=collections.Counter(raw)
    if not raw:return b'\0'+struct.pack('<I',0)
    if len(freq)==1:return b'\2'+struct.pack('<I',1)+raw[:1]
    heap=[(n,s,s)for s,n in freq.items()];heapq.heapify(heap);serial=256
    while len(heap)>1:
        a,b=heapq.heappop(heap),heapq.heappop(heap)
        heapq.heappush(heap,(a[0]+b[0],serial,(a[2],b[2])));serial+=1
    lengths={};stack=[(heap[0][2],0)]
    while stack:
        node,depth=stack.pop()
        if isinstance(node,int):lengths[node]=depth
        else:stack.extend((child,depth+1)for child in node)
    bits=sum(freq[s]*l for s,l in lengths.items())
    if max(lengths.values())>24 or (bits+7)//8+256>=len(raw)*(1-raw_threshold):return b'\0'+struct.pack('<I',len(raw))+raw
    code=0;last=0;codes={}
    for symbol,length in sorted(lengths.items(),key=lambda item:(item[1],item[0])):
        code <<= length-last;codes[symbol]=(code,length);code+=1;last=length
    data=bytearray();reservoir=0;count=0
    for symbol in raw:
        code,length=codes[symbol];reservoir=(reservoir<<length)|code;count+=length
        while count>=8:
            count-=8;data.append((reservoir>>count)&255)
        reservoir &= (1<<count)-1
    if count:data.append(reservoir<<(8-count))
    return b'\1'+struct.pack('<I',len(data))+bytes(lengths.get(s,0)for s in range(256))+data


def decode_front_reply(header,payload):
    from mlp_stream_codec import NAME as STREAM_NAME,decode_payload,length,layout as stream_layout
    dims=header['dims']
    if header.get('op')!='mlp_finish_delta_log_mlp_front' or header.get('encoding')not in (NAME,HUFFMAN_NAME)or len(dims)!=4 or any(type(d)is not int for d in dims):raise ValueError('carry front metadata')
    n,p,rows,front=dims;C=2560;history=24576
    if not 1<=n<=89 or not 1<=p<=132 or not 0<rows<C or rows%32 or not 0<front<9216 or front%256:raise ValueError('carry front bounds')
    stream_layout(dict(header,encoding=STREAM_NAME,op='mlp_stream_prepare',dims=[n,0,256]))
    layer=int(header['tensor'].split('.')[3])
    if (layer+2)%4==0 or n*C+history+p*6176>900000:raise ValueError('carry front layer/input bounds')
    inner=dict(header,encoding=STREAM_NAME,op='mlp_stream_prepare',dims=[n,0,front],tensor=f'model.language_model.layers.{layer+1}.post_attention_layernorm.weight',aux=[f'model.language_model.layers.{layer+2}.input_layernorm.weight'])
    if payload[:1]!=b'\2'or len(payload)<1+2*(n*C+history):raise ValueError('carry front reply direction/length')
    end=len(payload)-2*history;hidden=(np.frombuffer(payload[1:1+2*n*C],dtype='<u2').astype('<u4')<<16).view('<f4');carry=decode_payload(inner,payload[1+2*n*C:end]);conv=(np.frombuffer(payload[end:],dtype='<u2').astype('<u4')<<16).view('<f4')
    if carry.size!=length(n,front)or not np.isfinite(hidden).all()or not np.isfinite(conv).all():raise ValueError('carry front reply shape/finite')
    return np.concatenate([hidden,carry,conv])
