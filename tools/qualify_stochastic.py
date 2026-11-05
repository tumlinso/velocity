#!/usr/bin/env python3
"""Compare the native stochastic path with pinned scVelo source.

Only original scVelo numerical definitions are loaded from the inspected
checkout. The source oracle intentionally preserves its all-cell stochastic
fit; a second, independent implementation applies the declared row mask to all
fitting phases. No neighbor discovery or expression normalization is run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib.metadata import version
import numbers
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
    'scvelo/tools/optimization.py': '2194b2dd59e00d8693d6e079bcb255dac0738569372695089afc86dc0f73231b',
    'scvelo/preprocessing/moments.py': '9189d13d1257d86f49b6bb8d1816d7bbdcd647cb330cca922d738f36bf98714e',
}


def source_namespace():
    ns = dict(np=np, ndarray=ndarray, csr_matrix=csr_matrix, issparse=issparse,
              spmatrix=spmatrix, warnings=warnings, numbers=numbers, List=List, Optional=Optional,
              Tuple=Tuple, Union=Union)

    class Log:
        @staticmethod
        def warn(*args):
            pass

    ns['logg'] = Log
    groups = [
        ('scvelo/core/_arithmetic.py', {'prod_sum', 'sum'}),
        ('scvelo/core/_linear_models.py', {'LinearRegression'}),
        ('scvelo/tools/utils.py', {'make_dense', 'R_squared', 'groups_to_bool'}),
        ('scvelo/tools/optimization.py', {'get_weight', 'leastsq_generalized'}),
        ('scvelo/tools/velocity.py', {'Velocity'}),
    ]
    for path, names in groups:
        data = (SCVELO / path).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != SOURCE_HASHES[path]:
            raise RuntimeError(f'{path} source hash mismatch: {actual}')
        tree = ast.parse(data, filename=str(SCVELO / path))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if {node.name for node in nodes} != names:
            raise RuntimeError(f'{path} missing source definitions')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCVELO / path), 'exec'), ns)
    # Load the actual second-order function; only the graph lookup is injected.
    path = 'scvelo/preprocessing/moments.py'
    data = (SCVELO / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[path]:
        raise RuntimeError(f'{path} source hash mismatch')
    tree = ast.parse(data, filename=str(SCVELO / path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'second_order_moments']
    if len(nodes) != 1:
        raise RuntimeError('pinned second_order_moments definition missing')
    ns['get_connectivities'] = lambda a: a.obsp['qualification_W']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCVELO / path), 'exec'), ns)
    return ns


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


def _csr(x):
    if issparse(x):
        result = csr_matrix(x, dtype=np.float32, copy=True)
    else:
        result = csr_matrix(np.asarray(x, dtype=np.float32))
    result.sum_duplicates()
    result.sort_indices()
    return result


def moments(s, u, w):
    s, u, w = _csr(s), _csr(u), _csr(w)
    ms = np.asarray((w @ s).toarray(), dtype=np.float32)
    mu = np.asarray((w @ u).toarray(), dtype=np.float32)
    mss = np.asarray((w @ s.multiply(s)).toarray(), dtype=np.float32)
    mus = np.asarray((w @ s.multiply(u)).toarray(), dtype=np.float32)
    return ms, mu, mss, mus


def _source_initial(ns, ms, mu, fit_mask, gene_mask, fit_offset, percentile):
    velocity = ns['Velocity'](Ms=ms, Mu=mu, use_highly_variable=None, r2_adjusted=False)
    velocity._groups_for_fit = fit_mask
    velocity._highly_variable = gene_mask
    with np.errstate(all='ignore'):
        velocity.compute_deterministic(fit_offset=fit_offset, perc=percentile)
    return velocity


def _source_gls(ns, ms, mu, mss, mus, initial_residual, fit_offset,
                fit_offset2, percentile):
    # Keep source dtype/order: scVelo makes the corrected moments in float32,
    # estimates residual scales, then calls its original pseudoinverse GLS.
    var_ss = 2 * mss - ms
    cov_us = 2 * mus + mu
    lr = ns['LinearRegression'](fit_intercept=fit_offset2)
    with np.errstate(all='ignore'):
        lr.fit(var_ss, cov_us)
        offset2_init, gamma2_init = lr.intercept_, lr.coef_
        residual_scale = np.asarray(initial_residual).std(0)
        residual2_scale = (cov_us - gamma2_init * var_ss - offset2_init).std(0)
        offset, offset2, gamma = ns['leastsq_generalized'](
            ms, mu, var_ss, cov_us, residual_scale, residual2_scale,
            fit_offset, fit_offset2, percentile)
    return dict(gamma=np.asarray(gamma), offset=np.asarray(offset),
                offset2=np.asarray(offset2), residual_scale=np.asarray(residual_scale),
                residual2_scale=np.asarray(residual2_scale), var_ss=var_ss,
                cov_us=cov_us, offset2_initial=np.asarray(offset2_init),
                gamma2_initial=np.asarray(gamma2_init))


def _native_selection(initial, ms, mu, gene_mask, min_r2=.01, min_ratio=.01):
    selected = ((initial._r2 > min_r2) & (initial._gamma > min_ratio) &
                (np.max(ms > 0, axis=0) > 0) & (np.max(mu > 0, axis=0) > 0) & gene_mask)
    return np.asarray(selected, dtype=bool)


def scvelo_oracle(ns, s, u, w, fit_mask=None, gene_mask=None,
                  fit_offset=False, fit_offset2=False, percentile=None,
                  min_r2=.01, min_ratio=.01, refit_all_if_none=False):
    """Pinned source GLS on the native selection; optionally refit all on empty selection."""
    ms, mu, mss, mus = moments(s, u, w)
    n, g = ms.shape
    fit_mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    gene_mask = np.ones(g, dtype=bool) if gene_mask is None else np.asarray(gene_mask, dtype=bool)
    init = _source_initial(ns, ms, mu, fit_mask, gene_mask, fit_offset, percentile)
    source_selected = np.asarray(init._velocity_genes, dtype=bool) & gene_mask
    selected = _native_selection(init, ms, mu, gene_mask, min_r2, min_ratio)
    refine = selected if np.any(selected) else (gene_mask.copy() if refit_all_if_none else selected)
    result = dict(gamma=init._gamma.copy(), offset=init._offset.copy(),
                  offset2=np.zeros(g), residual_scale=np.zeros(g),
                  residual2_scale=np.zeros(g), velocity=init._residual.copy(),
                  residual2=np.zeros((n, g), dtype=np.float32),
                  selected=selected, source_selected=source_selected,
                  refined=refine, ms=ms, mu=mu, mss=mss, mus=mus,
                  r2=init._r2.copy(), initial_gamma=init._gamma.copy(),
                  initial_offset=init._offset.copy())
    if not np.any(refine):
        return result
    cols = np.flatnonzero(refine)
    with np.errstate(all='ignore'):
        fitted = _source_gls(ns, ms[:, cols], mu[:, cols], mss[:, cols], mus[:, cols],
                            init._residual[:, cols], fit_offset, fit_offset2, percentile)
    result['gamma'][cols] = fitted['gamma']
    result['offset'][cols] = fitted['offset']
    result['offset2'][cols] = fitted['offset2']
    result['residual_scale'][cols] = fitted['residual_scale']
    result['residual2_scale'][cols] = fitted['residual2_scale']
    result['var_ss'], result['cov_us'] = fitted['var_ss'], fitted['cov_us']
    result['gamma2_initial'] = fitted['gamma2_initial']
    result['offset2_initial'] = fitted['offset2_initial']
    result['velocity'] = mu - result['gamma'][None, :] * ms
    if fit_offset:
        result['velocity'] -= result['offset'][None, :]
    r2 = fitted['cov_us'] - 2 * ms[:, cols] * mu[:, cols]
    r2 -= result['gamma'][None, cols] * (fitted['var_ss'] - 2 * ms[:, cols] ** 2)
    if fit_offset:
        r2 += 2 * result['offset'][None, cols] * ms[:, cols]
    if fit_offset2:
        r2 -= result['offset2'][None, cols]
    result['residual2'][:, cols] = r2
    result['refined_count'] = int(len(cols))
    result['objective'] = weighted_objective(ms[:, cols], mu[:, cols], fitted, cols, fit_offset,
                                             fit_offset2, percentile)
    return result


def production_class_check(ns, s, u, w, fit_mask=None, gene_mask=None,
                           fit_offset=False, fit_offset2=False, percentile=None):
    """Run the source class and source second_order_moments AST end to end."""
    ms, mu, mss, mus = moments(s, u, w)
    n, g = ms.shape
    fit_mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    gene_mask = np.ones(g, dtype=bool) if gene_mask is None else np.asarray(gene_mask, dtype=bool)
    data = ad.AnnData(X=_csr(s))
    data.layers['spliced'] = _csr(s)
    data.layers['unspliced'] = _csr(u)
    data.obsp['qualification_W'] = _csr(w)
    data.uns['neighbors'] = {'connectivities_key': 'qualification_W'}
    velocity = ns['Velocity'](adata=data, Ms=ms, Mu=mu,
                             use_highly_variable=None, r2_adjusted=False)
    velocity._groups_for_fit = fit_mask
    velocity._highly_variable = gene_mask
    with np.errstate(all='ignore'):
        velocity.compute_stochastic(fit_offset=fit_offset, fit_offset2=fit_offset2,
                                    perc=percentile)
    if not (np.allclose(velocity._residual, mu - velocity._gamma[None, :] * ms -
                        (velocity._offset[None, :] if fit_offset else 0),
                        atol=1e-6, rtol=1e-5)):
        raise AssertionError('source class first residual disagrees with direct formula')
    return dict(gamma=velocity._gamma.copy(), offset=velocity._offset.copy(),
                offset2=velocity._offset2.copy(), velocity=velocity._residual.copy(),
                residual2=velocity._residual2.copy(), r2=velocity._r2.copy(),
                selected=velocity._velocity_genes.copy(), ms=ms, mu=mu,
                mss=mss, mus=mus)


def masked_oracle(ns, s, u, w, fit_mask, gene_mask=None, fit_offset=False,
                  fit_offset2=False, percentile=None):
    """Independent oracle implementing the declared consistent row-mask policy."""
    ms, mu, mss, mus = moments(s, u, w)
    n, g = ms.shape
    fit_mask = np.asarray(fit_mask, dtype=bool)
    gene_mask = np.ones(g, dtype=bool) if gene_mask is None else np.asarray(gene_mask, dtype=bool)
    init = _source_initial(ns, ms, mu, fit_mask, gene_mask, fit_offset, percentile)
    source_selected = np.asarray(init._velocity_genes, dtype=bool) & gene_mask
    selected = _native_selection(init, ms, mu, gene_mask)
    cols = np.flatnonzero(selected)
    result = dict(ms=ms, mu=mu, mss=mss, mus=mus, selected=selected,
                  refined=selected, gamma=init._gamma.copy(), offset=init._offset.copy(),
                  offset2=np.zeros(g), residual_scale=np.zeros(g),
                  residual2_scale=np.zeros(g), velocity=init._residual.copy(),
                  residual2=np.zeros((n, g), dtype=np.float32), r2=init._r2.copy(),
                  initial_gamma=init._gamma.copy(), initial_offset=init._offset.copy(),
                  source_selected=source_selected)
    if not len(cols):
        result['refined_count'] = 0
        return result
    take = np.flatnonzero(fit_mask)
    with np.errstate(all='ignore'):
        fitted = _source_gls(ns, ms[np.ix_(take, cols)], mu[np.ix_(take, cols)],
                            mss[np.ix_(take, cols)], mus[np.ix_(take, cols)],
                            init._residual[np.ix_(take, cols)], fit_offset, fit_offset2,
                            percentile)
    result['gamma'][cols] = fitted['gamma']
    result['offset'][cols] = fitted['offset']
    result['offset2'][cols] = fitted['offset2']
    result['residual_scale'][cols] = fitted['residual_scale']
    result['residual2_scale'][cols] = fitted['residual2_scale']
    result['velocity'] = mu - result['gamma'][None, :] * ms
    if fit_offset:
        result['velocity'] -= result['offset'][None, :]
    # ScVelo's requested second residual is defined for every row after fit.
    all_var = 2 * mss - ms
    all_cov = 2 * mus + mu
    r2 = all_cov[:, cols] - 2 * ms[:, cols] * mu[:, cols]
    r2 -= result['gamma'][None, cols] * (all_var[:, cols] - 2 * ms[:, cols] ** 2)
    if fit_offset:
        r2 += 2 * result['offset'][None, cols] * ms[:, cols]
    if fit_offset2:
        r2 -= result['offset2'][None, cols]
    result['residual2'][:, cols] = r2
    result['refined_count'] = int(len(cols))
    result['objective'] = weighted_objective(ms[np.ix_(take, cols)], mu[np.ix_(take, cols)],
                                             fitted, np.arange(len(cols)), fit_offset,
                                             fit_offset2, percentile)
    return result


def fp64_initial_oracle(s, u, w, fit_mask, fit_offset, percentile):
    """Independent centered FP64 regression on FP64 sparse W@S/W@U values."""
    sd, ud, wd = _csr(s).astype(np.float64), _csr(u).astype(np.float64), _csr(w).astype(np.float64)
    ms = np.asarray((wd @ sd).toarray(), dtype=np.float64)
    mu = np.asarray((wd @ ud).toarray(), dtype=np.float64)
    rows = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
    gamma, offset, condition = np.zeros(ms.shape[1]), np.zeros(ms.shape[1]), np.ones(ms.shape[1])
    for j in range(ms.shape[1]):
        x, y = ms[rows, j], mu[rows, j]
        keep = np.ones(len(rows), dtype=bool)
        if percentile is not None and len(rows):
            score = x/max(float(np.max(x)), 1e-3) + y/max(float(np.max(y)), 1e-3)
            pct = percentile
            if not fit_offset and isinstance(pct, (list, tuple)):
                pct = pct[1]
            if isinstance(pct, (tuple, list)):
                lo, hi = np.percentile(score, pct)
                keep = (score <= lo) | (score >= hi)
            else:
                keep = score >= np.percentile(score, pct)
            x, y = x[keep], y[keep]
        if not len(x):
            continue
        xx, xy = float(np.dot(x, x)), float(np.dot(x, y))
        if fit_offset:
            design = np.column_stack((x, np.ones(len(x))))
            condition[j] = float(np.linalg.cond(design))
            solution, *_ = np.linalg.lstsq(design, y, rcond=None)
            gamma[j], offset[j] = solution
            if offset[j] < 0:
                gamma[j], offset[j] = (xy/xx if xx > 0 else 0.), 0.
        else:
            gamma[j] = xy/xx if xx > 0 else 0.
    return dict(gamma=gamma, offset=offset, condition=condition, ms=ms, mu=mu)


def initial_precision_diagnostic(s, u, w, fit_mask, fit_offset, percentile, got, ref):
    truth = fp64_initial_oracle(s, u, w, fit_mask, fit_offset, percentile)
    scale = np.maximum(1., np.max(np.abs(truth['mu']), axis=0))
    gamma_tol = 1e-5 + 1e-4*np.abs(truth['gamma'])
    offset_tol = 1e-5*scale + 1e-4*np.abs(truth['offset'])
    native_gamma_error = np.abs(got['initial_gamma'] - truth['gamma'])
    source_gamma_error = np.abs(ref['initial_gamma'] - truth['gamma'])
    native_offset_error = np.abs(got['initial_offset'] - truth['offset'])
    source_offset_error = np.abs(ref['initial_offset'] - truth['offset'])
    source_fail = (source_gamma_error > gamma_tol) | (source_offset_error > offset_tol)
    native_pass = (native_gamma_error <= gamma_tol) & (native_offset_error <= offset_tol)
    ranked = np.flatnonzero(source_fail)
    ranked = ranked[np.argsort((source_gamma_error[ranked]/gamma_tol[ranked] +
                                source_offset_error[ranked]/offset_tol[ranked]))[::-1]]
    examples = []
    for j in ranked[:5]:
        examples.append(dict(gene=int(j), condition=float(truth['condition'][j]),
            native_gamma=float(got['initial_gamma'][j]), source_gamma=float(ref['initial_gamma'][j]),
            fp64_gamma=float(truth['gamma'][j]), native_offset=float(got['initial_offset'][j]),
            source_offset=float(ref['initial_offset'][j]), fp64_offset=float(truth['offset'][j]),
            native_gamma_error=float(native_gamma_error[j]), source_gamma_error=float(source_gamma_error[j]),
            native_offset_error=float(native_offset_error[j]), source_offset_error=float(source_offset_error[j]),
            gamma_tolerance=float(gamma_tol[j]), offset_tolerance=float(offset_tol[j])))
    return dict(source_diagnostic_failures=int(np.count_nonzero(source_fail)),
                native_within_fp64_tolerance=int(np.count_nonzero(native_pass)),
                selected_source_diagnostic_failures=int(np.count_nonzero(source_fail & ref['selected'])),
                selected_native_within_fp64_tolerance=int(np.count_nonzero(native_pass & ref['selected'])),
                selected_gene_count=int(np.count_nonzero(ref['selected'])),
                independent_fp64_selected_pass=bool(np.all(native_pass[ref['selected']])),
                max_condition=float(np.max(truth['condition'])), examples=examples)


def weighted_objective(ms, mu, fitted, columns, fit_offset, fit_offset2, percentile):
    """Per-gene scVelo weighted SSE divided by 2*n fitting rows."""
    ms, mu = np.asarray(ms), np.asarray(mu)
    var, cov = np.asarray(fitted['var_ss']), np.asarray(fitted['cov_us'])
    r1 = np.asarray(fitted['residual_scale'])
    r2 = np.asarray(fitted['residual2_scale'])
    gamma = np.asarray(fitted['gamma'])
    off1, off2 = np.asarray(fitted['offset']), np.asarray(fitted['offset2'])
    values = []
    for j in range(ms.shape[1]):
        keep = np.ones(ms.shape[0], dtype=bool)
        if percentile is not None:
            x, y = ms[:, j], mu[:, j]
            # Faithfully reproduce optimization.get_weight union and only zero
            # the first-order observations; second-order observations stay whole.
            xnorm = x / max(float(np.max(x)), 1e-3) + y / max(float(np.max(y)), 1e-3)
            only_x = x / max(float(np.max(x)), 1e-3)
            pct = percentile
            if not fit_offset and isinstance(pct, (tuple, list)):
                pct = pct[1]
            if isinstance(pct, (tuple, list)):
                lo, hi = np.percentile(xnorm, pct)
                lo_x, hi_x = np.percentile(only_x, pct)
                keep = ((xnorm <= lo) | (xnorm >= hi) |
                        (only_x <= lo_x) | (only_x >= hi_x))
            else:
                keep = ((xnorm >= np.percentile(xnorm, pct)) |
                        (only_x >= np.percentile(only_x, pct)))
        x = np.where(keep, ms[:, j], 0.)
        y = np.where(keep, mu[:, j], 0.)
        pred1 = gamma[j] * x + (off1[j] if fit_offset else 0.)
        pred2 = gamma[j] * var[:, j] + (off2[j] if fit_offset2 else 0.)
        value = 0.
        if r1[j] > 0 and np.isfinite(r1[j]):
            value += float(np.sum(((y - pred1) / r1[j]) ** 2))
        if r2[j] > 0 and np.isfinite(r2[j]):
            value += float(np.sum(((cov[:, j] - pred2) / r2[j]) ** 2))
        values.append(value / (2. * len(ms)))
    return np.asarray(values)


def fixture(path):
    a = ad.read_h5ad(path)
    return a.layers['spliced'], a.layers['unspliced'], graph(a.obsp['connectivities']), np.asarray(a.var['highly_variable'], dtype=bool)


def write_input(path, s, u, w):
    s, u, w = (_csr(x) for x in (s, u, w))
    for x in (s, u, w):
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
                   mss=get('<f8', (n, g)), mus=get('<f8', (n, g)),
                   velocity=get('<f4', (n, g)), residual2=get('<f4', (n, g)),
                   gamma=get('<f8', (g,)), offset=get('<f8', (g,)),
                   offset2=get('<f8', (g,)), residual_scale=get('<f8', (g,)),
                   residual2_scale=get('<f8', (g,)), r2=get('<f8', (g,)),
                   objective=get('<f8', (g,)),
                   initial_gamma=get('<f8', (g,)), initial_offset=get('<f8', (g,)),
                   selected=get('u1', (g,)).astype(bool),
                   refined=get('u1', (g,)).astype(bool), status=get('u1', (g,)))
        if f.read(1):
            raise RuntimeError('trailing native output')
        return out


def compare(ref, got):
    status = got['status'] == 0
    refined = np.asarray(ref['refined'], dtype=bool)
    scale = np.maximum(1., np.max(np.abs(ref['mu']), axis=0))
    checks = {}
    for field in ('ms', 'mu', 'mss', 'mus'):
        checks[field] = bool(np.allclose(got[field], ref[field], atol=1e-6, rtol=1e-5))
    for field in ('gamma', 'residual_scale', 'residual2_scale'):
        checks[field] = bool(np.allclose(got[field][refined], ref[field][refined], atol=1e-5, rtol=1e-4))
    for field in ('offset', 'offset2'):
        checks[field] = bool(np.all(np.abs(got[field][refined] - ref[field][refined]) <=
                                    1e-5*scale[refined] + 1e-4*np.abs(ref[field][refined])))
    checks['velocity'] = bool(np.all(np.abs(got['velocity'] - ref['velocity']) <=
                                     1e-5 * scale[None, :] + 1e-4 * np.abs(ref['velocity'])))
    second_scale = np.maximum.reduce((
        np.ones(ref['ms'].shape[1]),
        np.max(np.abs(2*ref['mus'] + ref['mu']), axis=0),
        np.max(np.abs(ref['gamma'][None, :] * (2*ref['mss'] - ref['ms'])), axis=0),
        np.max(np.abs(2*ref['ms']*ref['mu']), axis=0),
        np.max(np.abs(2*ref['gamma'][None, :]*ref['ms']**2), axis=0),
        np.max(np.abs(2*ref['offset'][None, :]*ref['ms']), axis=0),
        np.abs(ref['offset2'])))
    checks['second_residual'] = bool(np.all(np.abs(got['residual2'] - ref['residual2']) <=
                                            1e-5 * second_scale[None, :] + 1e-4 * np.abs(ref['residual2'])))
    checks['r2'] = bool(np.allclose(got['r2'], ref['r2'], atol=1e-4, rtol=1e-4))
    checks['initial_gamma'] = bool(np.allclose(got['initial_gamma'], ref['initial_gamma'], atol=1e-5, rtol=1e-4))
    checks['initial_offset'] = bool(np.all(np.abs(got['initial_offset'] - ref['initial_offset']) <=
                                           1e-5*scale + 1e-4*np.abs(ref['initial_offset'])))
    checks['finite'] = all(np.isfinite(got[k]).all() for k in ('ms','mu','mss','mus','velocity','residual2','gamma','offset','offset2','residual_scale','residual2_scale','r2','objective'))
    checks['successful_refinement'] = bool(np.all(status[refined]) and np.any(refined))
    mask_flips = got['selected'] != ref['selected']
    margin = ((np.abs(ref['r2'] - .01) <= 1e-4) |
              (np.abs(ref['initial_gamma'] - .01) <= 1.1e-5))
    checks['selected_away_from_threshold_margin'] = not bool(np.any(mask_flips & ~margin))
    refined_flips = got['refined'] != ref['refined']
    checks['refined_away_from_threshold_margin'] = not bool(np.any(refined_flips & ~margin))
    common = got['refined'] & refined
    common_idx = np.flatnonzero(common)
    ref_idx = np.flatnonzero(refined)
    ref_common = np.searchsorted(ref_idx, common_idx)
    objective_error = (np.abs(got['objective'][common_idx] - ref['objective'][ref_common])
                       if common_idx.size else np.array([np.inf]))
    checks['normalized_objective'] = bool(np.all(objective_error <=
        1e-4 * np.maximum(1., np.abs(ref['objective'][ref_common])))) if common_idx.size else False
    diagnostic_keys = ('initial_gamma', 'initial_offset', 'r2')
    estimator_keys = ('ms', 'mu', 'mss', 'mus', 'gamma', 'offset', 'offset2',
                      'residual_scale', 'residual2_scale', 'velocity', 'second_residual',
                      'finite', 'successful_refinement', 'selected_away_from_threshold_margin',
                      'refined_away_from_threshold_margin', 'normalized_objective')
    max_abs = {k: float(np.max(np.abs(got[k] - ref[k]))) for k in
               ('ms','mu','mss','mus','gamma','offset','offset2','residual_scale',
                'residual2_scale','velocity','residual2','r2','initial_gamma','initial_offset')}
    max_abs['objective'] = float(np.max(objective_error)) if objective_error.size else 0.
    return dict(pass_all=all(checks.values()), estimator_pass=all(checks[k] for k in estimator_keys),
                diagnostics_pass=all(checks[k] for k in diagnostic_keys), checks=checks,
                refined_count=int(np.count_nonzero(got['refined'])),
                source_refined_count=int(np.count_nonzero(refined)),
                selected_exact_match=not bool(np.any(mask_flips)),
                refined_exact_match=not bool(np.any(refined_flips)),
                selection_flips=int(np.count_nonzero(mask_flips)),
                selection_flips_away_from_margin=int(np.count_nonzero(mask_flips & ~margin)),
                status_counts={str(int(i)): int((got['status'] == i).sum()) for i in np.unique(got['status'])},
                max_abs=max_abs)


def compare_source_class(manual, source):
    checks = {}
    for key in ('gamma', 'offset', 'offset2'):
        checks[key] = bool(np.allclose(manual[key], source[key], atol=1e-5, rtol=1e-4))
    for key in ('velocity', 'residual2'):
        scale = np.maximum(1., np.max(np.abs(manual['mu']), axis=0))
        checks[key] = bool(np.all(np.abs(manual[key] - source[key]) <=
                                   1e-5*scale[None, :] + 1e-4*np.abs(source[key])))
    checks['selected'] = bool(np.array_equal(manual['source_selected'], source['selected']))
    checks['moments'] = bool(np.allclose(manual['mss'], source['mss'], atol=1e-6, rtol=1e-5)
                             and np.allclose(manual['mus'], source['mus'], atol=1e-6, rtol=1e-5))
    return dict(pass_all=all(checks.values()), checks=checks,
                max_abs={k: float(np.max(np.abs(manual[k] - source[k])))
                         for k in ('gamma','offset','offset2','velocity','residual2','mss','mus')})


def qualify(driver):
    ns = source_namespace()
    scratch = ROOT / 'tests/artifacts'
    scratch.mkdir(parents=True, exist_ok=True)
    report = dict(source_commit=SOURCE_COMMIT, source_hashes=SOURCE_HASHES,
                  native_driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
                  native_sources_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted((ROOT / 'src').rglob('*')) if p.is_file()},
                  source_loading='original AST Velocity, second_order_moments, LinearRegression, arithmetic and GLS definitions; only graph lookup injected',
                  versions=dict(numpy=np.__version__, scipy=scipy.__version__, anndata=version('anndata')),
                  method_contract=dict(selection='deterministic R2 and ratio filters; no low-signal relaxation',
                                       empty_selection='retain deterministic result unless refit_all_if_none is explicitly set',
                                       row_mask='consistent across deterministic and stochastic fitting; predict every row',
                                       source_difference='pinned scVelo applies row mask only to deterministic initialization; source refit-all-on-empty is disabled by default'),
                  recipe='finalized S/U and supplied W; no normalization or neighbor discovery; raw second moments cast to FP32 before fitting',
                  cases=[])
    for name in ('pancreas', 'dentategyrus'):
        path = SCVELO / 'tests/_data' / f'{name}_100obs_preprocessed.h5ad'
        s, u, w, hvg = fixture(path)
        n, g = s.shape
        native_input, masks, native_output = (scratch / f'{name}.stochastic.{suffix}' for suffix in ('input', 'masks', 'output'))
        write_input(native_input, s, u, w)
        variants = [('all', np.ones(n, dtype=np.uint8), np.ones(g, dtype=np.uint8))]
        fit = np.ones(n, dtype=np.uint8); fit[::3] = 0
        variants.append(('fit_subset_hvg', fit, hvg.astype(np.uint8)))
        for variant, fit_mask, genes in variants:
            masks.write_bytes(fit_mask.tobytes() + genes.tobytes())
            for fit_offset in (False, True):
                for fit_offset2 in (False, True):
                    for percentile in (None, (5, 95)):
                        if variant == 'all':
                            ref = scvelo_oracle(ns, s, u, w, fit_mask=fit_mask.astype(bool),
                                                gene_mask=genes.astype(bool), fit_offset=fit_offset,
                                                fit_offset2=fit_offset2, percentile=percentile)
                            source_ref = ref
                        else:
                            ref = masked_oracle(ns, s, u, w, fit_mask.astype(bool),
                                                genes.astype(bool), fit_offset, fit_offset2,
                                                percentile)
                            source_ref = scvelo_oracle(ns, s, u, w,
                                fit_mask=fit_mask.astype(bool), gene_mask=genes.astype(bool),
                                fit_offset=fit_offset, fit_offset2=fit_offset2,
                                percentile=percentile)
                        if not ref.get('refined_count', len(ref.get('refined', []))):
                            raise RuntimeError(f'{name}/{variant} has no stochastic genes to qualify')
                        class_ref = production_class_check(ns, s, u, w,
                            fit_mask=fit_mask.astype(bool), gene_mask=genes.astype(bool),
                            fit_offset=fit_offset, fit_offset2=fit_offset2,
                            percentile=percentile)
                        class_case = compare_source_class(source_ref, class_ref)
                        subprocess.run([str(driver), str(native_input), str(masks), str(native_output),
                                        str(int(fit_offset)), str(int(fit_offset2)),
                                        'none' if percentile is None else 'tails', '.01', '.01'],
                                       check=True, timeout=30)
                        got = read_output(native_output, n, g)
                        case = compare(ref, got)
                        case['pass_all'] = case['pass_all'] and class_case['pass_all']
                        case['estimator_pass'] = case['estimator_pass'] and class_case['pass_all']
                        case['fp64_initial_diagnostic'] = initial_precision_diagnostic(
                            s, u, w, fit_mask.astype(bool), fit_offset, percentile, got, source_ref)
                        case.update(fixture=name, fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    input_sha256=hashlib.sha256(native_input.read_bytes()).hexdigest(),
                                    variant=variant, fit_offset=fit_offset,
                                    fit_offset2=fit_offset2, percentile=percentile,
                                    observations=n, genes=g, graph_nnz=w.nnz,
                                    oracle=('pinned_scvelo_source_all_cell_stochastic' if variant == 'all'
                                            else 'independent_consistent_fit_mask'),
                                    pinned_production_class=class_case)
                        if variant == 'fit_subset_hvg':
                            overlap = ref['refined'] & source_ref['refined']
                            case['source_mask_deviation'] = dict(
                                overlapping_refined_genes=int(np.count_nonzero(overlap)),
                                source_refined_count=int(np.count_nonzero(source_ref['refined'])),
                                masked_refined_count=int(np.count_nonzero(ref['refined'])),
                                max_abs_gamma=float(np.max(np.abs(ref['gamma'][overlap] - source_ref['gamma'][overlap]))) if np.any(overlap) else None,
                                max_abs_offset=float(np.max(np.abs(ref['offset'][overlap] - source_ref['offset'][overlap]))) if np.any(overlap) else None)
                        report['cases'].append(case)
    report['pass_all'] = all(x['pass_all'] for x in report['cases'])
    report['estimator_pass_all'] = all(x['estimator_pass'] for x in report['cases'])
    report['diagnostics_pass_all'] = all(x['diagnostics_pass'] for x in report['cases'])
    report['independent_fp64_initial_pass_all'] = all(
        x['fp64_initial_diagnostic']['independent_fp64_selected_pass'] for x in report['cases'])
    report['production_class_pass_all'] = all(x['pinned_production_class']['pass_all'] for x in report['cases'])
    output = ROOT / 'tests/results/milestone-2.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['pass_all']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path)
    parser.add_argument('--reference-smoke', action='store_true')
    parser.add_argument('--source-smoke', action='store_true')
    args = parser.parse_args()
    if args.driver:
        qualify(args.driver.resolve())
        return
    ns = source_namespace()
    if args.source_smoke:
        for name in ('pancreas', 'dentategyrus'):
            path = SCVELO / 'tests/_data' / f'{name}_100obs_preprocessed.h5ad'
            s, u, w, hvg = fixture(path)
            result = scvelo_oracle(ns, s, u, w, gene_mask=hvg,
                                   fit_offset=False, fit_offset2=False, percentile=None)
            print(json.dumps(dict(fixture=name, shape=s.shape, selected=int(result['selected'].sum()),
                                  refined=int(np.count_nonzero(result['refined'])),
                                  finite=bool(np.isfinite(result['gamma']).all()))))
        return
    parser.error('select --driver or --source-smoke')


if __name__ == '__main__':
    main()
