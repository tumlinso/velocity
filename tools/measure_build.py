#!/usr/bin/env python3
"""Measure module rebuild behavior in a disposable source copy."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time


def identity(path):
    stat = path.stat()
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mtime_ns=stat.st_mtime_ns)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anndatapp-prefix', type=Path, default=root/'build/deps/install')
    parser.add_argument('--output', type=Path, default=root/'benchmarks/results/module-build.json')
    args = parser.parse_args()
    artifacts = root/'benchmarks/artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='module-build-', dir=artifacts))
    source, build = run/'source', run/'build'
    source.mkdir()
    for directory in ('src', 'cmake', 'examples'):
        shutil.copytree(root/directory, source/directory)
    shutil.copy2(root/'CMakeLists.txt', source/'CMakeLists.txt')
    shutil.copy2(root/'THIRD_PARTY_NOTICES.md', source/'THIRD_PARTY_NOTICES.md')

    def command(name, argv):
        started = time.perf_counter()
        with (run/f'{name}.log').open('w') as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
        return time.perf_counter()-started

    command('configure', ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
        '-DCMAKE_CXX_COMPILER=clang++-18', '-DCMAKE_BUILD_TYPE=Release',
        f'-DCMAKE_PREFIX_PATH={args.anndatapp_prefix.resolve()}',
        '-DVELOCITY_BUILD_TESTS=OFF', '-DVELOCITY_BUILD_QUALIFICATION=OFF'])
    invocation = ['cmake', '--build', str(build), '--parallel', '2', '--verbose']
    cold = command('fresh-build', invocation)
    example = source/'examples/steady_state.cpp'
    example.write_text(example.read_text()+'\n// Isolated consumer recompilation measurement.\n')
    consumer = command('consumer-recompile', invocation)
    bmis = sorted(build.rglob('*.pcm'))
    if not bmis:
        raise RuntimeError('No module artifacts found in measured build')
    objects = sorted(build.glob('CMakeFiles/velocity*example.dir/examples/*.cpp.o'))
    if not objects:
        raise RuntimeError('No consumer objects found in measured build')
    before_bmis = {str(p.relative_to(build)): identity(p) for p in bmis}
    before_objects = {str(p.relative_to(build)): identity(p) for p in objects}
    private = source/'src/velocity.cpp'
    private.write_text(private.read_text()+'\n// Isolated private implementation rebuild measurement.\n')
    private_time = command('private-rebuild', invocation)
    after_bmis = {str(p.relative_to(build)): identity(p) for p in bmis}
    after_objects = {str(p.relative_to(build)): identity(p) for p in objects}
    result = dict(
        platform=platform.platform(), cpu_count=os.cpu_count(), jobs=2,
        compiler=subprocess.check_output(['clang++-18', '--version'], text=True).splitlines()[0],
        cmake=subprocess.check_output(['cmake', '--version'], text=True).splitlines()[0],
        conditions='Fresh build directory; operating-system caches uncontrolled; no concurrent task benchmarks.',
        fresh_build_s=cold, consumer_recompile_s=consumer, private_rebuild_s=private_time,
        public_bmi_count=len(bmis), private_edit_preserved_bmis=before_bmis==after_bmis,
        private_edit_preserved_consumer=before_objects==after_objects,
        consumer_object_count=len(objects),
        native_source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'src').rglob('*')) if p.is_file()},
        artifacts=str(run.relative_to(root)))
    destination=args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['private_edit_preserved_bmis'] or not result['private_edit_preserved_consumer']:
        raise SystemExit('Private implementation edit rebuilt a public module or consumer')


if __name__ == '__main__':
    main()
