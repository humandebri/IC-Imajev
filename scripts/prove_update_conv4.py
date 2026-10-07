#!/usr/bin/env python3
"""Compare the conv4 candidate on all frozen proposals and restore its snapshot."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/prove_update_lora_finish.py';s=p.read_text().replace('artifacts/update-lora-finish-v1','artifacts/update-conv4-v1')
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
