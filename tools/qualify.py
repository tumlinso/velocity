#!/usr/bin/env python3
"""Independent pinned scVelo production-source numerical qualification.

Runs the original Velocity.compute_deterministic and its original numerical
helpers extracted from source, avoiding unrelated Scanpy import-time machinery.
Inputs are explicit S/U and positive-support/self-loop row-normalized W; no
normalization or neighborhood discovery is performed here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import warnings
from typing import List, Optional, Tuple, Union

import anndata as ad
import numpy as np
from numpy import ndarray
import scipy
from scipy.sparse import csr_matrix, issparse, spmatrix

ROOT = Path(__file__).resolve().parents[1]
SCVELO = Path(os.environ.get('SCVELO_SOURCE', '/home/tumlinson/accelerated-ports/scVelo'))
SOURCE_COMMIT = '4706c4115963b7cae68883d110d351a29904bc24'
SOURCE_HASHES = {
    'scvelo/tools/velocity.py': '13fdc0d785ec835e8f63dbd4562312cf5aa6d69bb457b96c34a6c3943eb656e8',
    'scvelo/core/_linear_models.py': '1c6eea7ce5277e9b44a3188e6a3004b1df5adb5326e21373f07c38a0a26d738a',
    'scvelo/core/_arithmetic.py': '18f281e8016f4bf9916b57742e57c089c0904ecdce7e2cda6b74cd1cdfaa61f5',
    'scvelo/tools/utils.py': '78ad0be1dc6121e9ea716c173209037e21ff79ee4db9a73a04b6057e83265715',
}


def source_namespace():
    ns = dict(np=np, ndarray=ndarray, csr_matrix=csr_matrix, issparse=issparse,
              spmatrix=spmatrix, warnings=warnings, List=List, Optional=Optional,
              Tuple=Tuple, Union=Union)
    class Log:
        @staticmethod
        def warn(*args):
            pass
    ns['logg'] = Log
    for path, names in [
        ('scvelo/core/_arithmetic.py', {'prod_sum', 'sum'}),
        ('scvelo/core/_linear_models.py', {'LinearRegression'}),
        ('scvelo/tools/utils.py', {'make_dense', 'R_squared', 'groups_to_bool'}),
        ('scvelo/tools/velocity.py', {'Velocity'}),
    ]:
        data = (SCVELO / path).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != SOURCE_HASHES[path]:
            raise RuntimeError(f'{path} source hash mismatch: {actual}')
        tree = ast.parse(data, filename=str(SCVELO / path))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if {node.name for node in nodes} != names:
            raise RuntimeError(f'{path} missing source definitions')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCVELO / path), 'exec'), ns)
    return ns


def graph(connectivities):
    """scVelo support recipe: positive edges, self, then row normalize."""
    w = connectivities.tocsr(copy=True)
    w.data = (w.data > 0).astype(np.float64)
    w.eliminate_zeros()
    w.setdiag(1)
    w.eliminate_zeros()
    w.sort_indices()
    return w.multiply(1.0 / np.asarray(w.sum(axis=1)).ravel()[:, None]).tocsr().astype(np.float32)


def oracle(ns, s, u, w, fit_mask=None, hvg=None):
    # Match scVelo moments.py: product is converted to float32 before fitting.
    ms = np.asarray((w @ s).toarray() if issparse(s) else w @ s, dtype=np.float32)
    mu = np.asarray((w @ u).toarray() if issparse(u) else w @ u, dtype=np.float32)
    velocity = ns['Velocity'](Ms=ms, Mu=mu, use_highly_variable=None, r2_adjusted=False)
    # Exercise production subgroup branch with an already resolved observation mask.
    velocity._groups_for_fit = fit_mask
    velocity._highly_variable = hvg
    with np.errstate(all='ignore'):
        velocity.compute_deterministic(fit_offset=False, perc=(5, 95))
    gamma, r2 = velocity._gamma, velocity._r2
    # Native default intentionally does not use scVelo's low-signal recovery.
    selected = (r2 > .01) & (gamma > .01) & (ms.max(axis=0) > 0) & (mu.max(axis=0) > 0)
    if hvg is not None:
        selected &= hvg
    return dict(ms=ms, mu=mu, gamma=gamma, offset=velocity._offset,
                r2=r2, velocity=velocity._residual, selected=selected,
                source_selected=velocity._velocity_genes)


def fixture(path):
    a = ad.read_h5ad(path)
    return a.layers['spliced'], a.layers['unspliced'], graph(a.obsp['connectivities']), np.asarray(a.var['highly_variable'], dtype=bool)



def write_input(path, s, u, w):
    """Serialize the common benchmark/interop CSR input as little-endian bytes."""
    s, u, w = (x.tocsr().astype(np.float32) for x in (s, u, w))
    for x in (s, u, w):
        x.sum_duplicates()
        x.sort_indices()
    n, g = s.shape
    if u.shape != (n, g) or w.shape != (n, n):
        raise ValueError('incompatible input shapes')
    with open(path, 'wb') as f:
        f.write(struct.pack('<5Q', n, g, s.nnz, u.nnz, w.nnz))
        for x in (s, u, w):
            f.write(x.indptr.astype('<i8').tobytes())
            f.write(x.indices.astype('<i4').tobytes())
            f.write(x.data.astype('<f4').tobytes())


def read_output(path, n, g):
    with open(path, 'rb') as f:
        def get(dtype, shape):
            count = int(np.prod(shape))
            a = np.fromfile(f, dtype=dtype, count=count)
            if a.size != count:
                raise RuntimeError('truncated native output')
            return a.reshape(shape)
        out = dict(ms=get('<f8', (n, g)), mu=get('<f8', (n, g)),
                   velocity=get('<f4', (n, g)), gamma=get('<f8', (g,)),
                   offset=get('<f8', (g,)), r2=get('<f8', (g,)),
                   selected=get('u1', (g,)).astype(bool), status=get('u1', (g,)))
        if f.read(1):
            raise RuntimeError('trailing native output')
        return out


def compare(ref, got):
    n, g = ref['ms'].shape
    ok = got['status'] == 0
    variance = np.var(ref['mu'].astype(np.float64), axis=0)
    stable = ok & np.isfinite(ref['r2']) & (variance > 1e-8 * np.maximum(1., np.max(np.abs(ref['mu']), axis=0)**2))
    signal = np.maximum(1., np.maximum(np.max(np.abs(ref['mu']), axis=0),
                                      np.abs(ref['gamma']) * np.max(np.abs(ref['ms']), axis=0)))
    moments_pass = all(np.all(np.isclose(got[k], ref[k], atol=1e-6, rtol=1e-5)) for k in ('ms', 'mu'))
    gamma_pass = bool(np.all(np.isclose(got['gamma'][ok], ref['gamma'][ok], atol=1e-5, rtol=1e-4)))
    offset_pass = bool(np.all(np.abs(got['offset'][ok] - ref['offset'][ok]) <= 1e-5*signal[ok] + 1e-4*np.abs(ref['offset'][ok])))
    velocity_pass = bool(np.all(np.abs(got['velocity'] - ref['velocity']) <= 1e-5*signal[None,:] + 1e-4*np.abs(ref['velocity'])))
    r2_pass = bool(np.all(np.abs(got['r2'][stable] - ref['r2'][stable]) <= 1e-4))
    flips = got['selected'] != ref['selected']
    near = (np.abs(ref['r2'] - .01) <= 1e-4) | (np.abs(ref['gamma'] - .01) <= 1.1e-5)
    checks = dict(moments=moments_pass, gamma=gamma_pass, offset=offset_pass,
                  velocity=velocity_pass, r2=r2_pass, selected_away_from_margin=not np.any(flips & ~near),
                  finite=all(np.isfinite(got[k]).all() for k in ('ms','mu','velocity','gamma','offset','r2')))
    return dict(pass_all=all(checks.values()), checks=checks, status_ok=int(ok.sum()),
                status_counts={str(int(i)): int((got['status']==i).sum()) for i in np.unique(got['status'])},
                r2_well_conditioned=int(stable.sum()), mask_flips=int(flips.sum()),
                mask_flips_away_from_margin=int((flips & ~near).sum()),
                scvelo_fallback_mask_differences=int(np.count_nonzero(ref['selected'] != ref['source_selected'])),
                max_abs={k: float(np.max(np.abs(got[k]-ref[k]))) for k in ('ms','mu','gamma','offset','r2','velocity')})


def qualify(driver, output=None, artifacts=None):
    ns = source_namespace()
    scratch = artifacts or ROOT / 'tests/artifacts'
    scratch.mkdir(parents=True, exist_ok=True)
    report = dict(source_commit=SOURCE_COMMIT, source_hashes=SOURCE_HASHES,
                  native_driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
                  native_sources_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted((ROOT / 'src').rglob('*')) if p.is_file()},
                  source_loading='original AST definitions with numerical helpers; logging suppressed; Scanpy wrapper not loaded',
                  versions=dict(numpy=np.__version__, scipy=scipy.__version__, anndata=__import__('importlib.metadata', fromlist=['version']).version('anndata')),
                  recipe='finalized spliced/unspliced; positive graph support + self + row normalization, float32 W and scVelo moments; no normalization or neighbor construction',
                  cases=[])
    for name in ('pancreas', 'dentategyrus'):
        path = SCVELO / 'tests/_data' / f'{name}_100obs_preprocessed.h5ad'
        s, u, w, hvg = fixture(path)
        n, g = s.shape
        native_input, masks, native_output = (scratch / f'{name}.{suffix}' for suffix in ('input', 'masks', 'output'))
        write_input(native_input, s, u, w)
        for variant in ('all', 'fit_subset_hvg'):
            fit = np.ones(n, dtype=np.uint8)
            genes = np.ones(g, dtype=np.uint8)
            if variant == 'fit_subset_hvg':
                fit[::3] = 0
                genes = hvg.astype(np.uint8)
            masks.write_bytes(fit.tobytes() + genes.tobytes())
            ref = oracle(ns, s, u, w, fit_mask=fit.astype(bool), hvg=genes.astype(bool))
            subprocess.run([str(driver), str(native_input), str(masks), str(native_output)], check=True, timeout=30)
            case = compare(ref, read_output(native_output, n, g))
            case.update(fixture=name, fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        input_sha256=hashlib.sha256(native_input.read_bytes()).hexdigest(),
                        variant=variant, observations=n, genes=g, graph_nnz=w.nnz)
            report['cases'].append(case)
    # Undefined fits have an explicit status and finite neutral diagnostics.
    n, g = 32, 4
    x = np.arange(n, dtype=np.float32) + 1
    s = np.column_stack((np.zeros(n), np.zeros(n), x, x)).astype(np.float32)
    u = np.column_stack((np.zeros(n), x, np.ones(n), x * 1.5)).astype(np.float32)
    w = scipy.sparse.eye(n, format='csr', dtype=np.float32)
    native_input, masks, native_output = (scratch / f'degenerate.{suffix}' for suffix in ('input', 'masks', 'output'))
    write_input(native_input, csr_matrix(s), csr_matrix(u), w)
    masks.write_bytes(bytes([1]) * (n+g))
    subprocess.run([str(driver), str(native_input), str(masks), str(native_output)], check=True, timeout=30)
    degenerate = read_output(native_output, n, g)
    finite = all(np.isfinite(degenerate[k]).all() for k in ('ms','mu','velocity','gamma','offset','r2'))
    degenerate_pass = bool(finite and np.all(degenerate['status'][:3] != 0)
                           and np.all(degenerate['gamma'][:2] == 0)
                           and np.all(degenerate['r2'][:3] == 0)
                           and not np.any(degenerate['selected'][:3]))
    report['degenerate'] = dict(pass_all=degenerate_pass, status=degenerate['status'].tolist(),
                                gamma=degenerate['gamma'].tolist(), r2=degenerate['r2'].tolist(),
                                selected=degenerate['selected'].tolist(), finite=finite)
    report['pass_all'] = all(x['pass_all'] for x in report['cases']) and degenerate_pass
    output = output or ROOT / 'tests/results/milestone-1.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['pass_all']:
        raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-smoke', action='store_true')
    parser.add_argument('--driver', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--artifacts-dir', type=Path)
    args = parser.parse_args()
    if args.driver:
        qualify(args.driver.resolve(), args.output, args.artifacts_dir)
        return
    ns = source_namespace()
    if args.reference_smoke:
        for name in ('pancreas', 'dentategyrus'):
            path = SCVELO / 'tests/_data' / f'{name}_100obs_preprocessed.h5ad'
            s, u, w, hvg = fixture(path)
            out = oracle(ns, s, u, w, hvg=hvg)
            print(json.dumps(dict(fixture=name, shape=s.shape, w_nnz=w.nnz,
                                  selected=int(out['selected'].sum()), finite_gamma=bool(np.isfinite(out['gamma']).all()))))
        return
    parser.error('select a qualification mode')


if __name__ == '__main__':
    main()
