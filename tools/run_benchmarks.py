#!/usr/bin/env python3
"""Run isolated paired milestone measurements; never overlap worker processes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import importlib.metadata
from pathlib import Path
import resource
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--extra-controls', action='store_true', help='also run retained, CSC and dense controls')
    parser.add_argument('--scaling-only', action='store_true', help='append the two N/G workspace controls without rerunning canonical samples')
    args = parser.parse_args()
    native, workload = args.native.resolve(), args.input.resolve()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    raw_events = ROOT / 'benchmarks/artifacts/milestone-1-events.jsonl'
    raw_events.parent.mkdir(parents=True, exist_ok=True)
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
        os.sched_setaffinity(0, {args.cpu})
    def run(command):
        try:
            p = subprocess.run(command, env=env, check=False, capture_output=True, text=True,
                               timeout=30, preexec_fn=limit, cwd=ROOT)
            event = dict(command=command, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as e:
            event = dict(command=command, timeout_seconds=30, stdout=str(e.stdout or ''), stderr=str(e.stderr or ''))
            with raw_events.open('a') as f:
                f.write(json.dumps(event) + '\n')
            raise
        with raw_events.open('a') as f:
            f.write(json.dumps(event) + '\n')
        if p.returncode:
            raise RuntimeError(f'worker failed with exit {p.returncode}: {p.stderr}')
        return json.loads(p.stdout)
    if args.scaling_only:
        result_path = ROOT / 'benchmarks/results/milestone-1.json'
        raw_path = ROOT / 'benchmarks/artifacts/milestone-1-runs.json'
        report = json.loads(result_path.read_text())
        if digest(native) != report['native_sha256'] or digest(workload) != report['workload_sha256']:
            raise RuntimeError('native binary or canonical workload differs from recorded comparison')
        if digest(ROOT/'benchmarks/qualification.cpp') != report['benchmark_source_sha256']:
            raise RuntimeError('benchmark source differs from recorded comparison')
        for name, expected in report['native_sources_sha256'].items():
            if digest(ROOT/name) != expected:
                raise RuntimeError(f'native source changed: {name}')
        samples = []
        for n, g in ((4096, 256), (8192, 256)):
            path = ROOT / f'benchmarks/artifacts/scaling-n{n}-g{g}.bin'
            generated = run([sys.executable, str(ROOT/'tools/benchmark_reference.py'), 'generate',
                             str(path), '--n', str(n), '--g', str(g), '--k', '32',
                             '--density', '.01', '--seed', '20261103'])
            result = run([str(native), '--input', str(path), '--expression', 'csr',
                          '--graph', 'csr', '--output', 'stream', '--block', '64'])
            samples.append(dict(shape=[n, g], workload_sha256=digest(path), generation=generated,
                                measurement=result))
        canonical_scratch = report['allocation_accounting']['native_canonical_workspace_bytes']
        first, second = (s['measurement']['workspace_scratch_bytes'] for s in samples)
        scaling = dict(samples=samples, canonical_shape=[4096, 2048],
                       canonical_scratch_bytes=canonical_scratch,
                       same_n_gene_scaling_scratch_ratio=first/canonical_scratch,
                       double_n_scratch_ratio=second/first,
                       limitations='One sample per shape; workspace comparisons only, no runtime claim.',
                       runner_sha256=digest(Path(__file__).resolve()))
        report['shape_scaling'] = scaling
        raw = json.loads(raw_path.read_text())
        raw['shape_scaling'] = scaling
        raw_path.write_text(json.dumps(raw, indent=2)+'\n')
        result_path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(scaling, indent=2))
        return
    def native_run(*options):
        return run([str(native), '--input', str(workload), *options])
    reference, candidate = [], []
    # Each pair is sequential and each sample is a fresh process.
    for _ in range(args.runs):
        reference.append(run([sys.executable, str(ROOT/'tools/benchmark_reference.py'), 'run', str(workload)]))
        candidate.append(native_run('--expression', 'csr', '--graph', 'csr', '--output', 'stream', '--block', '64'))
    controls = {}
    control_options = {
        'csr_stream_b8': ['--expression','csr','--output','stream','--block','8'],
        'csr_stream_b32': ['--expression','csr','--output','stream','--block','32'],
    }
    for name, options in control_options.items():
        controls[name] = native_run(*options)
    med = statistics.median
    reference_time = med(r['cold_total_s'] for r in reference)
    native_time = med(r['cold_total_s'] for r in candidate)
    reference_execute = med(r['execute_sink_s'] for r in reference)
    native_execute = med(r['execute_sink_s'] for r in candidate)
    reference_peak = med(r['incremental_execution_peak_bytes'] for r in reference)
    native_peak = med(r['incremental_peak_including_preparation_bytes'] for r in candidate)
    reference_sum = med(r['checksum'] for r in reference)
    native_sum = med(r['sink_sum_fp64'] for r in candidate)
    checksum_ok = abs(native_sum-reference_sum) <= max(1., abs(reference_sum))*1e-4
    summary = dict(reference_median_cold_s=reference_time, native_median_cold_s=native_time,
                   runtime_ratio_native_over_reference=native_time/reference_time,
                   reference_median_prepared_execution_s=reference_execute,
                   native_median_prepared_execution_s=native_execute,
                   prepared_execution_ratio_native_over_reference=native_execute/reference_execute,
                   reference_median_incremental_peak_bytes=reference_peak,
                   native_median_incremental_peak_bytes=native_peak,
                   incremental_memory_reduction=reference_peak/native_peak if native_peak else None,
                   memory_gate=bool(native_peak and reference_peak/native_peak >= 4),
                   cold_runtime_gate=native_time <= reference_time,
                   prepared_runtime_gate=native_execute <= reference_execute,
                   reference_checksum=reference_sum, native_checksum=native_sum,
                   approximate_checksum_pass=checksum_ok)
    if args.extra_controls and all((summary['memory_gate'], summary['cold_runtime_gate'],
                                    summary['prepared_runtime_gate'], checksum_ok)):
        for name, options in {
            'csr_retain_b64': ['--expression','csr','--output','retain','--block','64'],
            'csc_stream_b64': ['--expression','csc','--output','stream','--block','64'],
            'dense_stream_b64': ['--expression','dense','--output','stream','--block','64'],
        }.items():
            controls[name] = native_run(*options)
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')
    compiler = subprocess.run(['clang++-18', '--version'], capture_output=True, text=True, check=False).stdout.splitlines()
    artifact = dict(host=dict(cpu_model=cpu_model, kernel=platform.release(), python=platform.python_version(),
                              numpy=importlib.metadata.version('numpy'), scipy=importlib.metadata.version('scipy'),
                              compiler=compiler[0] if compiler else 'unavailable'),
                    workload_sha256=digest(workload), native_sha256=digest(native),
                    benchmark_source_sha256=digest(ROOT/'benchmarks/qualification.cpp'),
                    reference_worker_sha256=digest(ROOT/'tools/benchmark_reference.py'),
                    native_sources_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'src').rglob('*')) if p.is_file()},
                    constraints=dict(cpu=args.cpu, threads=1, per_process_timeout_seconds=30,
                                     per_process_virtual_memory_limit_bytes=2*1024**3, fresh_process=True,
                                     incremental_peak='process HWM minus RSS immediately after common input load; includes preparation and sink',
                                     cold_timing='common input file load, preparation, computation and sink; excludes interpreter startup/import'),
                    summary=summary, reference=reference, native=candidate, controls=controls)
    raw = ROOT/'benchmarks/artifacts/milestone-1-runs.json'
    out = ROOT/'benchmarks/results/milestone-1.json'
    raw.parent.mkdir(parents=True, exist_ok=True); out.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(artifact, indent=2)+'\n')
    out.write_text(json.dumps(artifact, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    if not all((summary['memory_gate'], summary['cold_runtime_gate'], summary['prepared_runtime_gate'], checksum_ok)):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
