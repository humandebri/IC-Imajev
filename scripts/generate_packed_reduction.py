#!/usr/bin/env python3
"""Pack four integer dot reductions into SIMD lanes, preserving exact sums."""
import argparse
import pathlib
import re


def generate(text):
    start = text.index('unsafe fn dot_tile_immediate')
    begin = text.index('        macro_rules! columns', start)
    end = text.index('        macro_rules! rows', begin)
    old = text[begin:end]
    expr = re.search(r'let v=(.*?);\n', old).group(1)
    replacement = '''        macro_rules! dot {($input:ident,$j:literal)=>{EXPRESSION};}
        macro_rules! group {($i:literal,$input:ident,$j:literal)=>{if C >= $j+4 {
            let a0=dot!($input,$j);
            let a1=dot!($input,{$j+1});
            let a2=dot!($input,{$j+2});
            let a3=dot!($input,{$j+3});
            let a=i32x4_add(i32x4_shuffle::<0,1,4,5>(a0,a1),i32x4_shuffle::<2,3,6,7>(a0,a1));
            let b=i32x4_add(i32x4_shuffle::<0,1,4,5>(a2,a3),i32x4_shuffle::<2,3,6,7>(a2,a3));
            let total=i32x4_add(i32x4_shuffle::<0,2,4,6>(a,b),i32x4_shuffle::<1,3,5,7>(a,b));
            v128_store(out[$i].as_mut_ptr().add($j).cast(),total);
        }};}
        macro_rules! columns {($i:literal,$input:ident;$($unused:literal),*)=>{
            group!($i,$input,0);group!($i,$input,4);group!($i,$input,8);group!($i,$input,12);
        };}
'''.replace('EXPRESSION', expr)
    # Expressions such as {4+1} remain compile-time indices.
    replacement = replacement.replace('dot {($input:ident,$j:literal)', 'dot {($input:ident,$j:expr)')
    return text[:begin] + replacement + text[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='crates/imajev-runtime/src/int8_kernel.rs')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate(pathlib.Path(args.source).read_text()))


if __name__ == '__main__':
    main()
