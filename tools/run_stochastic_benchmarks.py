#!/usr/bin/env python3
"""Run bounded paired stochastic velocity and workspace scaling probes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MIN_R2 = -100.0
MIN_RATIO = -100.0
BLOCK = 64


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_digest(bits: str) -> str:
    if any(c not in "01" for c in bits):
        raise RuntimeError("native selection mask is malformed")
    return hashlib.sha256(bytes(c == "1" for c in bits)).hexdigest()


def compact_scaling(report: dict) -> dict:
    return {key: report[key] for key in
            ("run_id", "mode", "source_sha256", "constraints", "ratios", "checks", "limitations")} | {
        "shapes": [{"shape": x["shape"], "workload_sha256": x["workload_sha256"],
                    "workspace_scratch_bytes": x["measurement"]["workspace_scratch_bytes"],
                    "prepared_bytes": x["measurement"]["prepared_bytes"],
                    "fit_result_bytes": x["measurement"]["fit_result_bytes"],
                    "incremental_peak_including_preparation_bytes": x["measurement"]["incremental_peak_including_preparation_bytes"],
                    "selected_genes": x["measurement"]["selected_genes"],
                    "successful_refinements": x["measurement"]["successful_refinements"]} for x in report["shapes"]]}


def compact_sample(row: dict) -> dict:
    keys = ("mode", "shape", "input_sha256", "selected_genes", "attempted_refinements",
            "successful_refinements", "selected_mask_sha256", "finite", "velocity_checksum",
            "velocity_absolute_checksum", "second_residual_checksum",
            "second_residual_absolute_checksum", "gamma_checksum", "offset_checksum",
            "offset2_checksum", "cold_total_s", "execute_sink_s", "load_s",
            "incremental_peak_including_preparation_bytes", "process_peak_bytes",
            "input_storage_bytes", "prepared_bytes", "workspace_scratch_bytes",
            "fit_result_bytes", "logical_velocity_bytes", "logical_second_residual_bytes",
            "stream_block_bytes", "prep_global_new_calls", "prep_global_new_requested_bytes",
            "execution_global_new_calls", "execution_global_new_requested_bytes")
    out = {key: row[key] for key in keys if key in row}
    if "source_materialized_fields" in row:
        out["source_materialized_fields"] = row["source_materialized_fields"]
    if "selected_mask_bits" in row:
        out["selected_mask_sha256"] = mask_digest(row["selected_mask_bits"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--scaling-only", action="store_true")
    args = p.parse_args()
    native = args.native.resolve()
    artifact_dir = ROOT / "benchmarks/artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_file = ROOT / "benchmarks/results/milestone-2.json"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    raw_events = artifact_dir / f"milestone-2-events-{run_id}.jsonl"
    raw_runs = artifact_dir / f"milestone-2-runs-{run_id}.json"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
        os.sched_setaffinity(0, {args.cpu})

    def run(label: str, command: list[str]) -> dict:
        try:
            proc = subprocess.run(command, env=env, check=False, capture_output=True,
                                  text=True, timeout=30, preexec_fn=limits, cwd=ROOT)
            event = dict(label=label, command=command, exit_code=proc.returncode,
                         stdout=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired as exc:
            event = dict(label=label, command=command, timeout_seconds=30,
                         stdout=str(exc.stdout or ""), stderr=str(exc.stderr or ""))
            with raw_events.open("a") as f: f.write(json.dumps(event)+"\n")
            raise RuntimeError(f"{label} exceeded 30 seconds") from exc
        with raw_events.open("a") as f: f.write(json.dumps(event)+"\n")
        if proc.returncode:
            raise RuntimeError(f"{label} failed ({proc.returncode}): {proc.stderr}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} did not emit one JSON result") from exc

    def generate(path: Path, n: int, g: int):
        return run(f"generate-{n}-{g}", [sys.executable, str(ROOT/"tools/benchmark_reference.py"),
            "generate", str(path), "--n", str(n), "--g", str(g), "--k", "32",
            "--density", ".01", "--seed", "20261103"])

    def native_run(path: Path):
        return run("native", [str(native), "--input", str(path), "--block", str(BLOCK)])

    def shape_of(path: Path):
        with path.open("rb") as f:
            head = f.read(16)
        if len(head) != 16:
            raise RuntimeError("input is shorter than the workload header")
        return list(struct.unpack("<2Q", head))

    scaling = []
    if args.scaling_only:
        if not native.is_file():
            raise RuntimeError("native benchmark executable does not exist")
        if result_file.exists():
            paired = json.loads(result_file.read_text())
            if paired.get("native_sha256") != digest(native):
                raise RuntimeError("native benchmark binary differs from the paired record")
            for path, expected in paired.get("native_sources_sha256", {}).items():
                if digest(ROOT/path) != expected:
                    raise RuntimeError(f"native source changed since paired measurement: {path}")
            if paired.get("benchmark_source_sha256") != digest(ROOT/"benchmarks/stochastic_qualification.cpp"):
                raise RuntimeError("benchmark source differs from the paired record")
        workload_shapes = [(256, 256), (256, 8192), (4096, 256), (8192, 256)]
        for n, g in workload_shapes:
            path = artifact_dir / f"stochastic-scaling-n{n}-g{g}.bin"
            gen = generate(path, n, g)
            measured = native_run(path)
            scaling.append(dict(shape=[n, g], workload_sha256=digest(path), generation=gen,
                                measurement=measured))
        scratch = {(x["shape"][0], x["shape"][1]): x["measurement"]["workspace_scratch_bytes"] for x in scaling}
        selected_diagnostics = {(x["shape"][0], x["shape"][1]): x["measurement"]["fit_result_bytes"] for x in scaling}
        same_n_gene_ratio = scratch[(256, 8192)]/scratch[(256, 256)]
        cell_scale_ratio = scratch[(8192, 256)]/scratch[(4096, 256)]
        scaling_checks = dict(scratch_independent_of_genes=0.98 <= same_n_gene_ratio <= 1.02,
                              scratch_linear_in_cells=1.9 <= cell_scale_ratio <= 2.1,
                              gene_diagnostics_reported_separately=selected_diagnostics[(256,8192)] > selected_diagnostics[(256,256)],
                              workers_finite=all(x["measurement"]["finite"] for x in scaling),
                              nonzero_refinement=all(x["measurement"]["successful_refinements"] > 0 for x in scaling))
        report = dict(schema="velocity.stochastic-benchmark.v1", run_id=run_id, mode="shape_scaling",
            source_sha256={"native": digest(native), "benchmark": digest(Path(__file__).resolve()),
                           "cpp": digest(ROOT/"benchmarks/stochastic_qualification.cpp")},
            constraints=dict(cpu=args.cpu, threads=1, per_process_timeout_seconds=30,
                             per_process_virtual_memory_limit_bytes=2*1024**3,
                             fresh_process=True, samples_per_shape=1),
            shapes=scaling, ratios=dict(same_n_gene_scratch=same_n_gene_ratio,
                                        doubled_n_scratch=cell_scale_ratio),
            checks=scaling_checks,
            limitations="Single bounded samples establish workspace scaling only; they do not establish runtime or memory reduction against scVelo.")
        raw_runs.write_text(json.dumps(report, indent=2)+"\n")
        # Preserve paired results in the compact final record when already present.
        compact = json.loads(result_file.read_text()) if result_file.exists() else {}
        compact["shape_scaling"] = compact_scaling(report)
        result_file.write_text(json.dumps(compact, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        if not all(scaling_checks.values()):
            raise SystemExit(1)
        return

    if not native.is_file():
        raise RuntimeError("native benchmark executable does not exist")
    workload = args.input.resolve() if args.input else artifact_dir / "stochastic-canonical.bin"
    generation = None
    if not args.input:
        generation = generate(workload, 4096, 2048)
    if shape_of(workload) != [4096, 2048]:
        raise RuntimeError("paired stochastic comparison requires the canonical 4096 x 2048 workload")
    reference_samples, native_samples = [], []
    for pair in range(args.runs):
        # Alternate order while keeping every worker fresh, sequential, and pinned.
        if pair % 2 == 0:
            ref = run("scvelo", [sys.executable, str(ROOT/"tools/benchmark_stochastic.py"),
                      str(workload), "--min-r2", str(MIN_R2), "--min-ratio", str(MIN_RATIO)])
            candidate = native_run(workload)
        else:
            candidate = native_run(workload)
            ref = run("scvelo", [sys.executable, str(ROOT/"tools/benchmark_stochastic.py"),
                      str(workload), "--min-r2", str(MIN_R2), "--min-ratio", str(MIN_RATIO)])
        reference_samples.append(ref); native_samples.append(candidate)

    med = statistics.median
    ref_time = med(x["cold_total_s"] for x in reference_samples)
    native_time = med(x["cold_total_s"] for x in native_samples)
    ref_execute = med(x["execute_sink_s"] for x in reference_samples)
    native_execute = med(x["execute_sink_s"] for x in native_samples)
    ref_peak = med(x["incremental_peak_including_preparation_bytes"] for x in reference_samples)
    native_peak = med(x["incremental_peak_including_preparation_bytes"] for x in native_samples)
    def close(a, b, rtol=1e-4, atol=1e-4):
        return abs(a-b) <= atol + rtol*max(abs(a), abs(b))
    checksums = {
        field: all(close(ref[key], native[key]) for ref, native in zip(reference_samples, native_samples))
        for field, key in (("velocity", "velocity_checksum"),
                           ("second_residual", "second_residual_checksum"),
                           ("gamma", "gamma_checksum"),
                           ("offset", "offset_checksum"),
                           ("offset2", "offset2_checksum"))
    }
    masks_source = []
    for ref in reference_samples:
        # Workers return SHA-256 over one byte per selected gene.
        masks_source.append(ref["selected_mask_sha256"])
    native_masks = [mask_digest(x["selected_mask_bits"]) for x in native_samples]
    same_masks = masks_source == native_masks
    selected = [x["selected_genes"] for x in reference_samples]
    native_selected = [x["selected_genes"] for x in native_samples]
    refined = [x["successful_refinements"] for x in reference_samples]
    native_refined = [x["successful_refinements"] for x in native_samples]
    summary = dict(
        reference_median_cold_s=ref_time, native_median_cold_s=native_time,
        cold_runtime_ratio=native_time/ref_time,
        reference_median_execute_sink_s=ref_execute, native_median_execute_sink_s=native_execute,
        execute_runtime_ratio=native_execute/ref_execute,
        reference_median_incremental_peak_bytes=ref_peak,
        native_median_incremental_peak_bytes=native_peak,
        incremental_memory_reduction=ref_peak/native_peak if native_peak else None,
        memory_gate=bool(native_peak and ref_peak/native_peak >= 10),
        cold_runtime_gate=native_time <= ref_time,
        selected_gene_counts=dict(scvelo=selected, native=native_selected),
        successfully_refined_counts=dict(scvelo=refined, native=native_refined),
        successful_refinement_counts_match=refined == native_refined,
        selected_mask_hashes=dict(scvelo=masks_source, native=native_masks),
        selection_match=same_masks,
        checksum_gates=checksums,
        all_workers_finite=all(x["finite"] for x in reference_samples+native_samples),
        nonzero_refinement=all(x > 0 for x in refined+native_refined),
        execute_runtime_is_diagnostic_only=True,
    )
    cpu_model = next((x.split(":",1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                      if x.startswith("model name")), "unknown")
    record = dict(schema="velocity.stochastic-benchmark.v1", run_id=run_id, mode="paired_canonical",
        host=dict(cpu_model=cpu_model, kernel=platform.release(), python=platform.python_version(),
                  numpy=importlib.metadata.version("numpy"), scipy=importlib.metadata.version("scipy")),
        scvelo_source_commit="4706c4115963b7cae68883d110d351a29904bc24",
        workload_sha256=digest(workload), generation=generation,
        native_sha256=digest(native),
        benchmark_source_sha256=digest(ROOT/"benchmarks/stochastic_qualification.cpp"),
        runner_sha256=digest(Path(__file__).resolve()),
        worker_sha256=digest(ROOT/"tools/benchmark_stochastic.py"),
        native_sources_sha256={str(x.relative_to(ROOT)): digest(x) for x in sorted((ROOT/"src").rglob("*")) if x.is_file()},
        options=dict(min_r2=MIN_R2, min_ratio=MIN_RATIO, initialization_percentile=[5,95],
                     stochastic_percentile=[5,95], fit_intercept=False, fit_second_intercept=False,
                     refit_all_if_none=False, block_genes=BLOCK, output="stream velocity and second residual"),
        constraints=dict(cpu=args.cpu, threads=1, per_process_timeout_seconds=30,
                         per_process_virtual_memory_limit_bytes=2*1024**3, fresh_process=True,
                         pairs=args.runs, input_loading_and_preparation_in_cold_time=True,
                         peak_metric="process HWM minus RSS immediately after common input load; includes preparation, execution and sink"),
        summary=summary, scvelo=reference_samples, native=native_samples,
        source_allocation_accounting="NumPy/SciPy allocations were not intercepted; HWM and identified materialized array sizes are recorded.",
        raw_events=raw_events.relative_to(ROOT).as_posix())
    raw_runs.write_text(json.dumps(record, indent=2)+"\n")
    compact_record = {key: record[key] for key in ("schema", "run_id", "mode", "host", "scvelo_source_commit",
        "workload_sha256", "generation", "native_sha256", "benchmark_source_sha256", "runner_sha256",
        "worker_sha256", "native_sources_sha256", "options", "constraints", "summary",
        "source_allocation_accounting", "raw_events")}
    compact_record["scvelo_samples"] = [compact_sample(x) for x in reference_samples]
    compact_record["native_samples"] = [compact_sample(x) for x in native_samples]
    compact_record["raw_runs"] = raw_runs.relative_to(ROOT).as_posix()
    if not result_file.exists():
        result_file.write_text(json.dumps(compact_record, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    gates = [summary["memory_gate"], summary["cold_runtime_gate"], summary["selection_match"],
             summary["successful_refinement_counts_match"], summary["all_workers_finite"],
             summary["nonzero_refinement"], *checksums.values()]
    if not all(gates) or min(selected+native_selected+refined+native_refined, default=0) == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
