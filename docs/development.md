# Development and Project Control

## Current package boundary

This checkout builds the standalone Velocity++ package. Its modules are
`velocitypp` and `velocitypp.precision`, its C++ namespace is `velocitypp`, and
its installed target is `velocitypp::velocitypp`. It uses AnnData++ 1.0 or
newer and binds already acquired AnnData++ descriptors with explicit dtype tags.
Callers supply all topology, workspace, state and result storage.
Supported numeric storage is FP32 and FP64.

There is no FP16 or binary16 support, Cellerator dependency or alias, CUDA
backend, or AVX2 backend. The source submodule lives at `external/velocitypp` in ccell. The root `ccell` wrappers provide `ccell.velocity`;
preprocessing remains owned by `ccell`. See the
[architecture](architecture.md) and [segmentation boundary](segmentation.md).

## Toolchain and standalone build

Use CMake 3.28 or newer, Clang 18 with the matching `clang-scan-deps` on
`PATH`, Ninja, and an installed AnnData++ 1.0 or newer:

```sh
cmake -S . -B build/debug -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH="/path/to/anndatapp/install" \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure
```

To build the library without tests and examples, set
`VELOCITYPP_BUILD_TESTS=OFF` and `VELOCITYPP_BUILD_EXAMPLES=OFF`. Qualification
drivers are enabled separately with `VELOCITYPP_BUILD_QUALIFICATION=ON`.
`VELOCITYPP_ENABLE_SANITIZERS=ON` enables address and undefined-behavior
sanitizers. All project options use the `VELOCITYPP_*` prefix.

The C++20 build uses the CMake module file sets and dependency scanning. Keep
Clang and `clang-scan-deps` from the same Clang 18 toolchain. A standalone
Velocity++ build does not configure the root ccell project or install any
Cellerator helper package.

## Installed consumers

Install the package with CMake, then expose both its prefix and the AnnData++
prefix when configuring a consumer. Consumer CMake uses:

```cmake
find_package(velocitypp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE velocitypp::velocitypp)
```

Import `velocitypp` for the core API and `velocitypp.precision` for optional
sensitivity and selective precision stages. Consumers create AnnData++ arrays and acquire their descriptors before
entering the allocation-free numerical reference.

Example and qualification executables use the `velocitypp_*` prefix. Use CTest
to run the configured native and installed-consumer checks rather than invoking
private test executables by name.

## Historical qualification and Project Control

The detailed M1/M2/M3 numerical, memory, GPU, and package receipts in
[validation](validation.md) are source-bound historical evidence. Their
recorded test results and exceptions remain intact; they do not qualify the
current direct-view source. The current native Debug and ASan/UBSan results are
summarized separately in [validation](validation.md).

The JSON planning and workflow receipts are preserved historical records, not
the current Project Control or Todo authority. Use the configured Project
Control workflow for authoritative current task state. Do not edit receipts to
make them appear to qualify a different source snapshot.

## Contribution and artifacts

Read [AGENTS.md](../AGENTS.md), [STYLE.md](../STYLE.md),
[architecture.md](architecture.md), and [validation.md](validation.md). Keep
numerical calculations compact, and make ownership and array-view lifetimes
explicit.

Keep generated data, build trees, environments, binaries, profiler output, and
raw measurement campaigns ignored and preserved locally. Preserve source
identities and historic JSON receipts. A source or package rename alone does
not establish numerical or performance qualification.
