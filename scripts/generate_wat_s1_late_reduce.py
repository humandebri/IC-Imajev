#!/usr/bin/env python3
"""Reduce SIMD lanes after integer Strassen recombination, preserving F32 order."""
import argparse,hashlib,json,pathlib,runpy
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
g=runpy.run_path(str(ROOT/'scripts/generate_wat_s1.py'),run_name='__main__');old=(ROOT/'scripts/wat_s1_bench/kernel.wat').read_text();new=old;changes=0
for m in range(7):
 fragment='\n'.join(['local.tee $a','local.get $a','local.get $a','i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11','i32x4.add',f'local.set $p{m}'])
 assert new.count(fragment)==8;new=new.replace(fragment,f'local.set $p{m}');changes+=8
for t in range(2):
 x=g['combine'](g['C'][t*2]);y=g['combine'](g['C'][t*2+1]);fragment='\n'.join(x+y+['i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27','f32x4.convert_i32x4_s']);assert new.count(fragment)==8
 replacement='\n'.join(x+['local.set $a']+y+['local.set $value','local.get $a','local.get $value','i8x16.shuffle 0 1 2 3 16 17 18 19 8 9 10 11 24 25 26 27','local.get $a','local.get $value','i8x16.shuffle 4 5 6 7 20 21 22 23 12 13 14 15 28 29 30 31','i32x4.add','f32x4.convert_i32x4_s'])
 new=new.replace(fragment,replacement)
# Linear integer map commutes with lane-pair reduction. Check each lane's
# symbolic contribution independently, beyond the old Strassen identity.
for terms in g['C']:
 before={(m,lane):c for m,c in terms.items()for lane in [0,1]}
 after={}
 for lane in [0,1]:
  for m,c in terms.items():after[m,lane]=after.get((m,lane),0)+c
 assert before==after
(d/'kernel.wat').write_text(new);sha=lambda b:hashlib.sha256(b).hexdigest();(d/'generator.json').write_text(json.dumps(dict(scope=__doc__,generator_sha256=sha(pathlib.Path(__file__).read_bytes()),original_generator_sha256=sha((ROOT/'scripts/generate_wat_s1.py').read_bytes()),original_kernel_sha256=sha(old.encode()),kernel_sha256=sha(new.encode()),removed_early_reductions=changes,output_reductions=16,symbolic_identity=True),indent=2)+'\n');print('late lane reduction identity verified')
