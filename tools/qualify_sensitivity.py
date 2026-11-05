#!/usr/bin/env python3
"""Check the advisory screen on the previously observed FP16-sensitive gene.

The fixture supplies already-normalized values.  A fixed FP32 graph is used
for both inputs; only S/U are narrowed to binary16 then widened to FP32.
This establishes a concrete regression case, not broad sensitivity recall.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.sparse import csr_matrix
import anndata as ad


def graph(connectivities):
    w = csr_matrix(connectivities, dtype=np.float32, copy=True)
    w.sum_duplicates()
    w.data = (w.data > 0).astype(np.float32)
    w.eliminate_zeros()
    w.setdiag(1)
    w.eliminate_zeros()
    w.sort_indices()
    sums = np.asarray(w.sum(axis=1)).ravel()
    return (w.multiply(1.0 / sums[:, None])).tocsr().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=Path('/home/tumlinson/accelerated-ports/scVelo/tests/_data/dentategyrus_100obs_preprocessed.h5ad'))
    parser.add_argument('--fixture-sha256', default='46033b23d673e2ff7e8fd6e75b619706861ad8418b6723989300b91ee9e870d8')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    fixture_hash = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    if fixture_hash != args.fixture_sha256:
        raise RuntimeError(f'fixture hash mismatch: {fixture_hash}')
    data = ad.read_h5ad(args.fixture)
    w = graph(data.obsp['connectivities'])
    s = np.asarray(data.layers['spliced'][:, 21].toarray(), dtype=np.float32).reshape(-1)
    u = np.asarray(data.layers['unspliced'][:, 21].toarray(), dtype=np.float32).reshape(-1)
    inputs = np.column_stack((w @ s, w @ u, w @ s.astype(np.float16).astype(np.float32),
                              w @ u.astype(np.float16).astype(np.float32))).astype(np.float32)
    payload = str(len(inputs)) + '\n' + '\n'.join(' '.join(format(float(x), '.9g') for x in row) for row in inputs) + '\n'
    run = subprocess.run([str(args.driver.resolve())], input=payload, text=True,
                         capture_output=True, timeout=30, check=True)
    result = json.loads(run.stdout)
    result.update(fixture=str(args.fixture.resolve()), fixture_sha256=fixture_hash,
                  driver_sha256=hashlib.sha256(args.driver.read_bytes()).hexdigest(),
                  rows=len(inputs), source_gene=21, weight_dtype='float32', storage_dtype='float16', compute_dtype='float32')
    result['pass'] = (result['original_status'] == 0 and result['half_status'] == 0 and
                      not result['original_selected'] and bool(result['half_selected']) and result['report_state'] == 3)
    result['limit'] = 'One fixture and one trimming/intercept setting; no claim about general recall or biological validity.'
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
