#!/usr/bin/env python3
"""Run one pinned scVelo stochastic comparison in an isolated worker.

The runner applies the external 30-second and 2-GiB limits. Input loading is
timed as part of the common case; source extraction and imports happen first.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import numbers
import os
from pathlib import Path
import resource
import struct
import time
import warnings

import numpy as np
from scipy.sparse import csr_matrix

from benchmark_reference import load, rss
from qualify import ROOT, SCVELO, SOURCE_COMMIT, SOURCE_HASHES, source_namespace

OPTIMIZATION_HASH = "2194b2dd59e00d8693d6e079bcb255dac0738569372695089afc86dc0f73231b"


class Selection:
    def __init__(self, genes):
        self.genes = np.asarray(genes, dtype=np.int64)

    def __getitem__(self, key):
        _, genes = key
        return Selection(self.genes[np.asarray(genes, dtype=bool)])


def source_stochastic_namespace():
    ns = source_namespace()
    path = SCVELO / "scvelo/tools/optimization.py"
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != OPTIMIZATION_HASH:
        raise RuntimeError(f"optimization.py source hash mismatch: {actual}")
    tree = ast.parse(data, filename=str(path))
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in
             {"get_weight", "leastsq_generalized"}]
    if {x.name for x in nodes} != {"get_weight", "leastsq_generalized"}:
        raise RuntimeError("pinned optimization source lacks stochastic fitting helpers")
    ns.update(dict(csr_matrix=csr_matrix, np=np, warnings=warnings, numbers=numbers))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    ns["maximum_likelihood"] = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Bayesian mode is outside this benchmark")
    )
    return ns


def run(path: Path, min_r2: float, min_ratio: float) -> dict:
    ns = source_stochastic_namespace()
    before = rss()
    start = time.perf_counter()
    s, u, w = load(path)
    loaded = rss()
    load_end = time.perf_counter()
    n, g = s.shape

    # Match scVelo's float32 first and raw second moments. SciPy products
    # materialize full fields here, matching the reference's allocation shape.
    ms = np.asarray((w @ s).toarray(), dtype=np.float32)
    mu = np.asarray((w @ u).toarray(), dtype=np.float32)
    mss = np.asarray((w @ s.multiply(s)).astype(np.float32).toarray(), dtype=np.float32)
    mus = np.asarray((w @ s.multiply(u)).astype(np.float32).toarray(), dtype=np.float32)

    class SelectedInput:
        def __init__(self, genes=None):
            self.genes = np.arange(g) if genes is None else np.asarray(genes, dtype=np.int64)

        def __getitem__(self, key):
            return SelectedInput(self.genes[np.asarray(key[1], dtype=bool)])

    def selected_source(adata):
        return (mss if adata.genes.size == g else mss[:, adata.genes],
                mus if adata.genes.size == g else mus[:, adata.genes])
    ns["second_order_moments"] = selected_source
    velocity = ns["Velocity"](
        adata=SelectedInput(), Ms=ms, Mu=mu, use_highly_variable=None,
        r2_adjusted=False, min_r2=min_r2, min_ratio=min_ratio,
    )
    captured = {}
    source_solver = ns["leastsq_generalized"]
    def capture_scales(*args, **kwargs):
        captured["first"] = np.asarray(args[4]).copy()
        captured["second"] = np.asarray(args[5]).copy()
        return source_solver(*args, **kwargs)
    ns["leastsq_generalized"] = capture_scales
    with np.errstate(all="ignore"):
        velocity.compute_deterministic(fit_offset=False, perc=(5, 95))
        selected = np.asarray(velocity._velocity_genes, dtype=bool).copy()
        velocity.compute_stochastic(fit_offset=False, fit_offset2=False, mode=None, perc=(5, 95))
        velocity_checksum = float(np.sum(velocity._residual, dtype=np.float64))
        second_residual_checksum = float(np.sum(velocity._residual2, dtype=np.float64))
    done = time.perf_counter()
    after = rss()
    refined_indices = np.flatnonzero(selected) if np.any(selected) else np.arange(g)
    refined_ok = np.isfinite(velocity._gamma[refined_indices]) & np.isfinite(velocity._offset[refined_indices])
    refined_ok &= np.isfinite(captured["first"]) & (captured["first"] > 0)
    refined_ok &= np.isfinite(captured["second"]) & (captured["second"] > 0)
    successful = np.zeros(g, dtype=bool)
    successful[refined_indices] = refined_ok
    mask_bytes = selected.astype(np.uint8).tobytes()
    return dict(
        mode="scvelo_production_source",
        source_commit=SOURCE_COMMIT,
        source_hashes={**SOURCE_HASHES, "scvelo/tools/optimization.py": OPTIMIZATION_HASH},
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        shape=[n, g],
        options=dict(min_r2=min_r2, min_ratio=min_ratio, fit_intercept=False,
                     fit_second_intercept=False, percentile=[5, 95], refit_all_if_none=True),
        selected_genes=int(selected.sum()),
        attempted_refinements=int(refined_indices.size),
        successful_refinements=int(successful.sum()),
        selected_mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        velocity_checksum=velocity_checksum,
        second_residual_checksum=second_residual_checksum,
        velocity_absolute_checksum=float(np.sum(np.abs(velocity._residual), dtype=np.float64)),
        second_residual_absolute_checksum=float(np.sum(np.abs(velocity._residual2), dtype=np.float64)),
        gamma_checksum=float(np.sum(velocity._gamma, dtype=np.float64)),
        offset_checksum=float(np.sum(velocity._offset, dtype=np.float64)),
        offset2_checksum=float(np.sum(velocity._offset2, dtype=np.float64)),
        finite=bool(all(np.isfinite(x).all() for x in
                        (velocity._residual, velocity._gamma, velocity._offset, velocity._offset2))),
        load_s=load_end-start,
        execute_sink_s=done-load_end,
        cold_total_s=done-start,
        rss_before_bytes=before["VmRSS"], rss_loaded_bytes=loaded["VmRSS"],
        process_peak_bytes=after["VmHWM"],
        incremental_peak_including_preparation_bytes=max(0, after["VmHWM"]-loaded["VmRSS"]),
        input_storage_bytes=sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in (s,u,w)),
        logical_velocity_bytes=n*g*4,
        source_materialized_fields=dict(first_moments_bytes=2*n*g*4,
            raw_second_moments_bytes=2*n*g*4,
            first_residual_bytes=n*g*4,
            gsl_doubled_design_and_response_bytes=2*2*n*g*4,
            final_velocity_bytes=n*g*4),
        retained_output="full source moments, fit intermediates and velocity until scalar checksums finish",
        python_allocation_accounting="NumPy/SciPy allocation bytes are not intercepted; HWM captures peak process memory",
        maxrss_resource_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--min-r2", type=float, default=-100.0)
    p.add_argument("--min-ratio", type=float, default=-100.0)
    args = p.parse_args()
    print(json.dumps(run(args.input, args.min_r2, args.min_ratio), allow_nan=False))


if __name__ == "__main__":
    main()
