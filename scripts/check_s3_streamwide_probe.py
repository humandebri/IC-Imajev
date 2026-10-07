#!/usr/bin/env python3
"""Check Winograd's modulo-I32 reconstruction against independent native dots."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/check_s3_stream_probe.py';s=p.read_text().replace('artifacts/s3-stream-v1','artifacts/s3-streamwide-v1').replace('s3_stream','s3_streamwide').replace('for n in [57,59,67]:','for n in [48,56,57,59,67]:')
 d=ROOT/'artifacts/s3-streamwide-v1';(d/'frozen-check.py').write_text(s);(d/'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
