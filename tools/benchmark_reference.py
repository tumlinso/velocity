#!/usr/bin/env python3
"""Isolated, phase-accounted scVelo comparison worker for milestone 1.

Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS to 1 before
launch. Use an external 30 s timeout and 2 GiB process memory limit. The
binary input is shared with the native worker, and is not timed as setup.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import struct
import time

import numpy as np
from scipy.sparse import csr_matrix
from qualify import source_namespace, oracle


def rss():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value, *_ = line.split()
            values[key[:-1]] = int(value) * 1024
    return values


def generate(path, n, g, k, density, seed):
    rng = np.random.default_rng(seed)
    def expr():
        # Exact number of independently sampled positions; duplicates are summed.
        size = int(n * g * density)
        row = rng.integers(0, n, size, dtype=np.int32)
        col = rng.integers(0, g, size, dtype=np.int32)
        val = rng.uniform(.1, 10., size).astype(np.float32)
        x = csr_matrix((val, (row, col)), shape=(n, g))
        x.sort_indices()
        return x
    s, u = expr(), expr()
    row = np.repeat(np.arange(n, dtype=np.int32), k)
    col = rng.integers(0, n, n*k, dtype=np.int32)
    w = csr_matrix((np.ones(n*k, np.float32), (row, col)), shape=(n, n))
    w.data[:] = 1
    w.setdiag(1)
    w = w.multiply(1. / np.asarray(w.sum(axis=1)).ravel()[:, None]).tocsr().astype(np.float32)
    w.sort_indices()
    with open(path, 'wb') as f:
        f.write(struct.pack('<5Q', n, g, s.nnz, u.nnz, w.nnz))
        for x in (s, u, w):
            f.write(x.indptr.astype('<i8').tobytes())
            f.write(x.indices.astype('<i4').tobytes())
            f.write(x.data.astype('<f4').tobytes())
    print(json.dumps(dict(file=str(path), n=n, g=g, k=k, density=density, seed=seed,
                          nnz=[s.nnz, u.nnz, w.nnz], bytes=os.path.getsize(path))))


def load(path):
    with open(path, 'rb') as f:
        n, g, sn, un, wn = struct.unpack('<5Q', f.read(40))
        matrices = []
        for columns, nnz in ((g, sn), (g, un), (n, wn)):
            indptr = np.fromfile(f, dtype='<i8', count=n+1)
            indices = np.fromfile(f, dtype='<i4', count=nnz)
            data = np.fromfile(f, dtype='<f4', count=nnz)
            matrices.append(csr_matrix((data, indices, indptr), shape=(n, columns)))
        if f.read(1):
            raise ValueError('trailing input bytes')
    return matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('generate', 'run'))
    parser.add_argument('file', type=Path)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--g', type=int, default=2048)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--density', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=20261103)
    args = parser.parse_args()
    if args.mode == 'generate':
        generate(args.file, args.n, args.g, args.k, args.density, args.seed)
        return
    ns = source_namespace()
    before = rss()
    start = time.perf_counter()
    s, u, w = load(args.file)
    prepared = rss()
    ready = time.perf_counter()
    out = oracle(ns, s, u, w)
    # The sink examines the complete velocity, then returns only scalar diagnostics.
    checksum = float(np.sum(out['velocity'], dtype=np.float64))
    done = time.perf_counter()
    end = rss()
    print(json.dumps(dict(mode='scvelo_source', input=str(args.file),
                          load_s=ready-start, execute_sink_s=done-ready,
                          cold_total_s=done-start, rss_before=before,
                          rss_prepared=prepared, rss_after=end,
                          incremental_execution_peak_bytes=max(0, end['VmHWM']-prepared['VmRSS']),
                          total_process_peak_bytes=end['VmHWM'],
                          expression_bytes=sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in (s,u)),
                          graph_bytes=w.data.nbytes+w.indices.nbytes+w.indptr.nbytes,
                          materialized_velocity_bytes=out['velocity'].nbytes,
                          retained_output='complete velocity plus moments and diagnostics', checksum=checksum,
                          maxrss_resource=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))

if __name__ == '__main__':
    main()
