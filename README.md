# velocity

A native C++20 RNA velocity toolkit, built around named modules, explicit
numerical memory, and composable scientific algorithms.

Milestone 1 implements first-order neighborhood moments and steady-state
velocity with bounded numerical workspace. Milestone 2 implements
second-order neighborhood moments and stochastic velocity with the same
memory and customization priorities. The reference should substantially
improve on scVelo's memory use while producing numerically similar results;
bitwise identity and exact equality are not requirements. It should be useful
and efficient on its own, with direct calculations, deliberate allocation,
reusable buffers, and practical C++ compilation and iteration.

**Status:** first/second moments, deterministic and stochastic fitting,
streaming outputs, custom stages, and explicit references are implemented.
The active source is portable CPU code with FP16, FP32 and FP64 numeric
views, explicit FP32/FP64 computation, optional source-gene promotion, and
advisory screening. Velocity owns the FP16 view/edit boundary and uses
Cellerator's small CPU-only binary16 conversion helper. AnnData++ integer
arrays are never treated as FP16. The portable component now lives in `Cellerator/src/velocity/`;
CUDA and AVX2/F16C implementations and their supporting files are preserved in
the parent checkout at `Cellerator/quarantine/velocity-accelerators/`.
Cellerator parent-build integration is optional and disabled by default. See
the [segmentation boundary](docs/segmentation.md). Historical numerical and
performance records remain preserved as historical evidence and are not reused
to qualify the FP16 ownership migration. See current migration checks in
[validation](docs/validation.md) and the [planning record](planning/README.md).

```cpp
import velocity;

// data is an anndatapp::AnnData with prepared expression and explicit weights.
velocity::PreparedInput input(velocity::from_anndata(data));
velocity::Workspace workspace(input);
auto output = anndatapp::Array::empty<float>({input.n_cells(), input.n_genes()});
auto fit = velocity::reference::steady_state_into(input, workspace, output.edit());
```

AnnData++ is required for its array, axis, and dataset interfaces. FP16
storage is passed through explicit velocity numeric views over caller-owned
raw bits. The standalone CPU target does not require the full Cellerator
runtime, CUDA, Python, or Torch. See [development](docs/development.md) for
the direct build command and separate qualification workflows.

Milestone 1 measured 23.8x lower preparation-inclusive incremental peak memory
on a qualified 4,096-cell, 2,048-gene sparse workload, with similar runtime.
Its supplied numerical fixtures passed the declared tolerances with no gene
mask disagreements. Milestone 2 measured **37.9x lower incremental peak memory**
(690 MB to 18.2 MB) while refining all 2,048 genes on the canonical sparse
workload; the paired median runtime met the no-regression gate. All 32
stochastic estimator cases passed. See
[validation and measurement limits](docs/validation.md).

## Standalone CPU build

From this directory, configure directly against an installed AnnData++ prefix
with Clang 18, its matching `clang-scan-deps`, CMake 3.28 or newer, and Ninja:

```sh
cmake -S . -B build/standalone -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH="/path/to/anndatapp/install;/path/to/binary16/install" \
  -DVELOCITY_BUILD_TESTS=OFF \
  -DVELOCITY_BUILD_EXAMPLES=OFF
cmake --build build/standalone
```

This builds the CPU library without configuring the Cellerator parent project.
Inside Cellerator, the source build reuses its existing header-only
`Cellerator::binary16` target or the adjacent helper source. A standalone clone
uses the installed CPU-only `CelleratorBinary16` package instead.
AnnData++ supplies the array and dataset interfaces; installed velocity
consumers find the `CelleratorBinary16` package transitively.

## Scientific direction

The scientific scope includes neighborhood statistics, steady-state and
stochastic velocity estimation, transcription/splicing/degradation dynamics,
parameter and latent-time inference, and the associated diagnostics. Velocity
graphs and transition calculations are related, separately composable
operations. Milestone 1 covers first moments and the steady-state path.
Milestone 2 adds second-order moments and stochastic fitting. The remaining
scientific families are not implemented yet.

scVelo supplies mathematical and behavioral reference material. Its Python
object structure, allocation patterns, dense intermediates, and whole-dataset
workflow need not determine the native implementation. Changes to models,
objectives, preprocessing, precision, and statistical conventions must remain
explicit and testable. Different numerical algorithms and solvers are welcome
when they meet the declared numerical-similarity and scientific requirements.

## Cellerator and reference behavior

Cellerator is expected to develop specialized implementations later. It must
be able to reuse, specialize, replace, and fuse meaningful parts
of the computation, including numerical evaluation, fitting, updates, and
batch execution. velocity retains an explicitly callable reference path
for the same declared method and inputs, with downstream substitutions bypassed.

The shared contract establishes scientific meaning and numerical expectations;
implementations can choose different layouts, working sets, and execution.
Reference behavior and fixtures should be stable enough for regression and
cross-implementation checks without freezing the library's architecture before
it has been designed.

The library and standalone CPU build do not depend on the full Cellerator
runtime, Python, Torch, or CUDA. [AnnData++](https://github.com/tumlinso/anndatapp)
supplies core array, axis, and dataset interfaces. `velocity::NumericView` and
`NumericEdit` adapt ordinary FP32/FP64 AnnData views and explicitly wrap
caller-owned FP16 bits; AnnData integer arrays are not used as a half type.
CPU conversion comes from Cellerator's header-only `Cellerator::binary16`
helper. This does not add FP16 codecs or a persistent CellShard format.
Python is used only for optional independent qualification. The portable
component exports `velocity::velocity` and the `velocity` and
`velocity.precision` modules. The accelerator archive is outside its build
and package exports.

## Project documents

- [Architecture and design constraints](docs/architecture.md)
- [scVelo mathematics and existing acceleration evidence](docs/scvelo-foundation.md)
- [Numerical validation and efficiency evidence](docs/validation.md)
- [Native API, workspace, and customization](docs/api.md)
- [Storage, computation, and selective precision](docs/precision.md)
- [Portable source boundary and accelerator quarantine](docs/segmentation.md)
- [Development and Project Control](docs/development.md)
- [Coding style](STYLE.md) and [agent guidance](AGENTS.md)
- [Planning state](planning/README.md)

## Repository

The canonical repository is [tumlinso/velocity](https://github.com/tumlinso/velocity).
Cellerator consumes it as the `src/velocity` Git submodule. Quarantined
accelerators remain in the parent Cellerator repository, outside this repository
and its build.
