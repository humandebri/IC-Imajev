#!/usr/bin/env python3
"""Report progress/errors using saved results only; never feed references to graph."""
import argparse,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/full-int8-canister');ap.add_argument('--output',default='artifacts/full-int8-canister/layer-comparison.json');args=ap.parse_args();directory=ROOT/args.directory;rows=[]
for path in sorted((directory/'queries').glob('layer-*.npy')):
 i=int(path.stem.split('-')[1]);actual=np.load(path);expected=np.load(ROOT/f'artifacts/full-reference/layer-{i:02d}.npz')['output'][0];error=abs(actual-expected);rows.append(dict(layer=i,max_error=float(error.max()),mean_error=float(error.mean()),last_token_max_error=float(error[-1].max()),rmse=float(np.sqrt(np.mean(error**2))),finite=bool(np.isfinite(actual).all())))
(ROOT/args.output).write_text(json.dumps(dict(scope='Actual full graph versus official unquantized hidden tensors; not judgment accuracy',layers=rows),indent=2)+'\n');print(json.dumps(rows[-3:],indent=2))
