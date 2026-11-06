# Velocity++

A C++20 numerical development reference for RNA velocity. The implementation
exposes calculations, borrowed numerical storage and replaceable computational
regions. Its standard reference allocates nothing: callers supply topology,
workspace, results and output buffers. AnnData++ provides existing acquired data
descriptors; no Python, Torch, CUDA or Cellerator runtime is required.

```cpp
import velocitypp;
import velocitypp.precision;  // Optional sensitivity and original-column promotion.
```

FP32 and FP64 storage/computation, dense signed strides, CSR/CSC, deterministic
steady-state fitting, second-order moments and stochastic refinement are supported.
Raw per-gene routines, bound reference stages, static template composition and
runtime function-pointer bindings are available. Consumers can retain novel
layouts across replaced regions or replace the complete algorithm.

The caller constructs AnnData++ storage and acquires addressable descriptors
before invoking the reference. Numerical calls never materialize selections or
create owned result arrays. See [API](docs/api.md), [precision](docs/precision.md)
and the compiled [examples](examples) for caller-bound usage. This is a clean
API cutover from the former allocating workspace/result classes.

## Build

Use Linux, Clang 18 with matching clang-scan-deps, CMake >=3.28, Ninja, and an
installed AnnData++ 1.0 or newer:

```sh
cmake -S . -B build/debug -G Ninja -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH=/path/to/anndatapp/install -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug --parallel 2
ctest --test-dir build/debug --output-on-failure
```

The standalone package remains `velocitypp`, with target
`velocitypp::velocitypp`. In a ccell source checkout this Git submodule is at
`external/velocitypp`; root `import ccell.velocity;` re-exports the same types.
CPU FP16, specialized device code and Cellerator integration belong downstream.

## Contracts and evidence

- [Strict numerical style](STYLE.md)
- [Architecture and scientific contracts](docs/architecture.md)
- [Build and installed consumers](docs/development.md)
- [Source-bound qualification and historical limits](docs/validation.md)

Historical planning JSON, fixtures and measurement receipts remain unchanged;
they do not qualify the refactored source. Current acceptance requires native
and installed-consumer tests, allocation instrumentation over all reference
phases, independent numerical comparisons, and module rebuild checks. Performance
and scientific-equivalence claims require their own source-bound evidence.
