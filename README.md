# Velocity++

Velocity++ is a standalone C++20 RNA velocity library with explicit numerical
memory and composable scientific algorithms. Its repository and public package
identity are `velocitypp`; the local ccell submodule directory is `velocity/`.

## Current package boundary

The native modules are `velocitypp` and `velocitypp.precision`, the C++ namespace
is `velocitypp`, and the installed CMake target is `velocitypp::velocitypp`.
The standalone library depends on AnnData++ 1.0 or newer and accepts its arrays
directly through `anndatapp::ArrayView`, `anndatapp::ArrayEdit`, and
`anndatapp::DType`. Expression and graph inputs and outputs support FP32 and
FP64.

There is no FP16 or binary16 path, no Cellerator dependency or target alias, and
no CUDA or AVX2 backend in this package. In the root `ccell` project, wrappers
provide `ccell.velocity` and `ccell` owns preprocessing. Those wrappers do not
change the standalone package boundary.

The current source carries first- and second-order neighborhood moments,
steady-state and stochastic fitting, bounded workspace, explicit reference
stages, and optional precision sensitivity/source selection. Historical
qualification records below are tied to earlier source snapshots; they do not
qualify the current direct-view `velocitypp` source.

```cpp
import velocitypp;

// data is an anndatapp::AnnData with prepared expression and explicit weights.
velocitypp::PreparedInput input{velocitypp::from_anndata(data)};
velocitypp::Workspace workspace{input};
auto fit = velocitypp::reference::steady_state(input, workspace);
```

## Standalone build

Use Clang 18 with its matching `clang-scan-deps`, CMake 3.28 or newer, Ninja,
and an installed AnnData++ 1.0 or newer:

```sh
cmake -S . -B build/standalone -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH="/path/to/anndatapp/install" \
  -DVELOCITYPP_BUILD_TESTS=OFF \
  -DVELOCITYPP_BUILD_EXAMPLES=OFF
cmake --build build/standalone
```

This builds Velocity++ without configuring `ccell` and without Python, Torch,
CUDA, or a half-precision helper. CMake options use the `VELOCITYPP_*` prefix;
example and qualification executables use the `velocitypp_*` prefix.

Installed consumers use:

```cmake
find_package(velocitypp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE velocitypp::velocitypp)
```

## Scientific scope

The scientific scope includes neighborhood statistics, steady-state and
stochastic velocity estimation, transcription/splicing/degradation dynamics,
parameter and latent-time inference, and associated diagnostics. Velocity
graphs and transition calculations are related, separately composable
operations. The earlier first two milestones cover first moments and
steady-state velocity, followed by second-order moments and stochastic fitting.
The remaining scientific families are not part of this standalone package yet.

scVelo supplies mathematical and behavioral reference material. Its Python
object structure, allocation patterns, dense intermediates, and whole-dataset
workflow do not determine the native implementation. Changes to models,
objectives, preprocessing, precision, and statistical conventions must remain
explicit and testable. Different numerical algorithms and solvers are welcome
when they meet declared numerical-similarity and scientific requirements.

## Historical qualification records

The Milestone 1 and Milestone 2 measurements describe earlier `velocity`
source snapshots. Milestone 1 recorded 23.8x lower preparation-inclusive
incremental peak memory on a 4,096-cell, 2,048-gene sparse workload, with similar
runtime and passing supplied numerical fixtures. Milestone 2 recorded 37.9x
lower incremental peak memory (690 MB to 18.2 MB), a passing paired median
no-regression gate, and 32 passing stochastic estimator cases. These are
historical results; they do not establish current `velocitypp` qualification or
a general runtime speedup. See [validation and evidence limits](docs/validation.md).

## Project documents

- [Architecture and design constraints](docs/architecture.md)
- [scVelo mathematics and existing acceleration evidence](docs/scvelo-foundation.md)
- [Numerical validation and efficiency evidence](docs/validation.md)
- [Native API, workspace, and customization](docs/api.md)
- [Storage, computation, and selective precision](docs/precision.md)
- [Current package boundary and historical source segmentation](docs/segmentation.md)
- [Development and Project Control](docs/development.md)
- [Coding style](STYLE.md) and [agent guidance](AGENTS.md)
- [Planning state](planning/README.md)

## Repository

The standalone repository is
[tumlinso/velocitypp](https://github.com/tumlinso/velocitypp).
