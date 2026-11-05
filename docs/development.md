# Development and Project Control

## Present development state

Milestone 1 implements paired first moments and steady-state velocity.
Milestone 2 implements bounded-memory stochastic fitting. Public
consumption is `import velocity;` linked through `velocity::velocity`.
AnnData++ is required for core arrays, axes, and dataset access. The numeric
interface accepts FP16/FP32/FP64 through `NumericView`; FP16 uses explicit
caller-owned raw-bit buffers and Cellerator's CPU-only binary16 helper.
Native runtime use needs no Python, CUDA, Torch, HDF5, or Zarr backend. CUDA
and AVX2 are quarantined outside the active build; see
[portable segmentation](segmentation.md). Backed orchestration remains
caller-controlled and outside this CPU component. Device allocation and
transfers are outside this CPU-only velocity target.

The qualified local tools are Clang 18.1.3, matching clang-scan-deps,
CMake 3.28.3, and Ninja 1.11.1. CMake module file sets and dependency scanning
establish build order; `import std;` is not required.

## Standalone CPU build

The portable component builds directly from this directory. Install AnnData++
and the CPU-only `CelleratorBinary16` helper separately and pass their
installation prefixes through `CMAKE_PREFIX_PATH`. Use
Clang 18 with matching `clang-scan-deps`, CMake 3.28 or newer, and Ninja:

```sh
cmake -S . -B build/standalone -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH="/path/to/anndatapp/install;/path/to/binary16/install" \
  -DVELOCITY_BUILD_TESTS=OFF \
  -DVELOCITY_BUILD_EXAMPLES=OFF
cmake --build build/standalone
```

This standalone CPU build has no dependency on the full Cellerator build,
CUDA, Python, or Torch. AnnData++ and the CPU-only binary16 helper are
required. Inside the Cellerator checkout, CMake reuses the existing helper
target or adjacent source; a standalone clone discovers the installed
`CelleratorBinary16` package without configuring the parent project. Parent-project integration is optional and
disabled by default; its build alias does not change the installed
`velocity::velocity` target or the `velocity` and `velocity.precision` modules.
Installed velocity consumers find `CelleratorBinary16` through the velocity
CONFIG package.

## Extended build and package checks

For the historical local check workflow, the helper builds an isolated core-only
AnnData++ dependency from a sibling checkout without modifying it:

```sh
sh tools/build_anndatapp.sh /path/to/anndatapp
cmake --workflow --preset native-check

cmake --preset release
cmake --build --preset release
ctest --preset release

cmake --preset sanitizers
cmake --build --preset sanitizers
ctest --preset sanitizers

cmake --preset shared
cmake --build --preset shared
ctest --preset shared
```

Historical M1/M2 dependency source was `d9869da416ca36571421241fb10821d4c0acd19a`.
The coordinated milestone builds directly from the modified sibling anndata
source, with exact tested source hashes in the new validation receipts.
The helper builds that dependency binary in Release. Sanitizers instrument
velocity and its tests/interfaces, not the separately built dependency binary.
Independent installed-package consumers pass in both static and shared
configurations, including two executables sharing one consumer-local module
interface build. Use `find_package(velocity CONFIG REQUIRED)` and link
`velocity::velocity`; consumers compile their own compatible BMIs.

The active CMake project exposes the portable CPU component only. CUDA/ISA
options, toolkit discovery, accelerator source lists, and CUDA package exports
were removed during segmentation. Original sources, integration files, mixed
comparison tools, and accelerator-specific helpers are preserved in
quarantine (`Cellerator/quarantine/velocity-accelerators/` in the parent checkout). The active precision
benchmark accepts `--backend scalar` and `--backend cpu` for its two portable
entrypoints. The ISA-specific availability query was removed.

AnnData++ remains required for its core array and dataset interfaces. It does
not own velocity's FP16 views or binary16 conversion. Velocity's installed
package requires the separate header-only `CelleratorBinary16` helper
transitively; it does not require the full Cellerator runtime. The current
migration checks are summarized in [validation](validation.md); older receipts
do not qualify this ownership change.

The historical [build receipt](../benchmarks/results/module-build-milestone-3.json)
records a warm consumer median ratio of 1.000375 against the frozen baseline
and unchanged BMIs/consumer objects after private CPU, AVX2, CUDA, and codec
edits. Cold build cost increased with the added implementations; the receipt
reports that separately.

## Independent numerical and memory qualification

Milestone 1's optional Python tools use `tests/requirements-reference.txt`, a reference
environment, and the pinned scVelo checkout selected by `SCVELO_SOURCE`.
The source-comparison route executes verified original production definitions
extracted from the source AST, avoiding unrelated Scanpy import-time machinery.
Milestone 2 also runs an independent masked-fit oracle because the chosen
consistent row-mask behavior deliberately differs from scVelo. The tools do
not download biological data.

```sh
reference_python=/path/to/reference-env/bin/python
export SCVELO_SOURCE=/path/to/scVelo
"$reference_python" tools/qualify.py --driver build/release/velocity_interop_driver
"$reference_python" tools/benchmark_reference.py generate benchmarks/artifacts/canonical.bin
"$reference_python" tools/run_benchmarks.py --native build/release/velocity_benchmark \
  --input benchmarks/artifacts/canonical.bin --cpu 0 --extra-controls
python tools/measure_build.py
"$reference_python" tools/qualify_stochastic.py \
  --driver build/release/velocity_stochastic_interop_driver
"$reference_python" tools/run_stochastic_benchmarks.py \
  --native build/release/velocity_stochastic_benchmark --cpu 0
python tools/measure_build.py \
  --output benchmarks/results/module-build-milestone-2.json
```

Stochastic numerical qualification compares the pinned scVelo source and the
independent consistent-mask oracle on both supplied fixtures, including gene
subsets, intercept combinations and trimming modes. Its compact result is
`tests/results/milestone-2.json`. The paired benchmark runner generates a fixed-seed 4,096-cell, 2,048-gene
input when `--input` is omitted, then records results in
`benchmarks/results/milestone-2.json`; `--scaling-only` adds bounded workspace
scaling controls. Choose an allowed idle CPU and set it with `--cpu`. The Milestone 2 module-build output path keeps the historical
Milestone 1 result intact.

The current stochastic qualification command exits 1 because four FP32 source
initialization offsets exceed their original tolerance. Its report separately
records passing estimator, independent FP64 initialization, and production-class
gates. See the explicit [qualification exception](validation.md); do not treat
that exit as a passing strict source-diagnostic suite.

Do not interpret the Milestone 1 records as stochastic-fit evidence.

Select an allowed idle CPU for `--cpu`; workers run sequentially with one
BLAS/OMP thread, a 30-second timeout and a 2 GiB address-space cap. Reports
record source, binary and input hashes. Compact results live under
`tests/results` and `benchmarks/results`; generated fixtures, raw runs, logs,
and disposable source/build copies remain ignored. The build measurement
changes only its disposable copy and checks private edits preserve public
BMIs and consumer objects. See [validation](validation.md) for evidence limits.

## Contribution and scope

Read [AGENTS.md](../AGENTS.md), [STYLE.md](../STYLE.md),
[architecture.md](architecture.md), and [validation.md](validation.md).
Follow the written, protected
[Cellerator coding guide](/home/tumlinson/Cellerator/coding_style.md).
Keep numerical calculations compact and memory/ownership operations explicit.

Inspect current source, method options, and attribution before adapting scVelo
or accelerated-ports code. Preserve upstream license notices for any derived
implementation. Documentation of an opportunity is not a measured improvement.

Use configured Codex subagents for useful bounded work. Coordinate overlapping
files and shared compute. Preserve other agents' changes, upstream checkouts,
supplied data, native workflow state, and historical evidence.

## Project Control

Active work on this relocated component uses the parent Cellerator checkout's
Project Control registration and Todo authority. The `velocity-cppm` workspace
identity and setup record below are historical. The original
`.todo-orchestrator/` now resides at `/home/tumlinson/velocity`; relocation did
not move or recreate that workflow state.

### Historical velocity-cppm identities and workflow

The former workspace and repository alias were both `velocity-cppm`; its
intended authority ID was `velocity-cppm-authority`. Registration used the
actual root `/home/tumlinson/velocity.cppm`. Its local origin was
`git@github.com:tumlinso/velocity.cppm.git`.

The installed [Todo Orchestrator skill](/home/tumlinson/.agents/skills/todo-orchestrator/SKILL.md)
describes supported workflow routes. Routine work in the registered workspace
uses `next_task`, then the opaque handle with `inspect_task`, `coordinate_task`,
and `finish_task`. Observer reads require no claim. `next_task` is not a
readiness check because it can claim or resume work.

Canonical state for that former workspace remains in
`/home/tumlinson/velocity/.todo-orchestrator/authority.sqlite3`, with
project identity in its `project.json`. Those files and SQLite sidecars remain
private/ignored in the old checkout. Never hand-edit the database, generated
projections, leases, or recovery state. Keep owner and profile credentials
outside the checkout; do not publish tokens, private previews, or runtime
artifacts.

Authored planning files are reviewable inputs. Todo's accepted plan and task
state are authority. [Milestone 1](../planning/milestone-1.json) was accepted
through supported authoring. Context assembly failed after native attempt
admission; the user explicitly instructed work to proceed without a handle.
The [workflow receipt](../planning/milestone-1-workflow.json) preserves this
distinction. Local results are not a native accepted outcome.

### Historical setup verification

[The former setup receipt](../planning/setup-receipt.json) records the result,
installed versions, identities, and limitations for `velocity-cppm`; it does
not describe the active Cellerator authority. Check those historical details
independently if they are needed:

1. The registry maps the intended alias to the intended repository root.
2. The initialized Todo authority has matching project/repository/authority
   identities and the expected current plan/task state.
3. Native integration refs and private provider runtime directories exist.
4. Host credentials are scoped to the exact project, authority, principal,
   profile, and permitted actions; routine profiles do not receive the owner
   bearer.
5. A fresh Project Control process can retrieve exact source and semantic
   workflow state. An existing connected process may retain an older registry
   or provider result; report that separately.

Service health alone is not project readiness. A partial response caused only
by an unavailable optional context extension differs from missing authority,
credentials, or source access. Preserve active attempts and unrelated sessions
when resolving setup problems. Do not restart shared services or change their
permissions merely to silence a diagnostic.

## Artifacts and checks

Keep generated biological data, builds, environments, binaries, profiler output,
and raw measurement campaigns ignored and preserved locally. Commit source,
small reproducible fixtures/generators, compact evidence, and source identities
when publication is authorized. Commit and push only with user authorization.

For documentation changes, check relative links, source-reference provenance,
JSON syntax, status claims, secret exclusion, and `git diff --check`.
Run affected native, installed-consumer, sanitizer, numerical, and measurement
checks as appropriate. Do not report unexecuted checks as passing or repeat a
measurement campaign without a new evidence gap.
