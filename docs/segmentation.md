# Current package boundary and historical source segmentation

This checkout is the standalone Velocity++ package. The repository directory
remains `velocity`, while its current module, namespace, CMake package, and
installed target are `velocitypp`, `velocitypp`, `velocitypp`, and
`velocitypp::velocitypp` respectively.

The public module interface files are `src/velocitypp.cppm` and
`src/velocitypp.precision.cppm`. Consumers import `velocitypp` or
`velocitypp.precision`. The standalone source accepts and returns FP32/FP64
AnnData++ arrays directly through `anndatapp::ArrayView`,
`anndatapp::ArrayEdit`, and `anndatapp::DType`.

There are no `NumericView` or `NumericEdit` adapters, FP16 or binary16
conversion paths, Cellerator dependencies or aliases, CUDA sources, or AVX2
sources in this package. Its required data dependency is AnnData++ 1.0 or newer.
FP16 support belongs only to the separate Cellerator checkout.
Within the root `ccell` project, wrappers provide `ccell.velocity` and `ccell`
owns preprocessing.

## Historical relocation and qualification evidence

The [first-party relocation receipt](../planning/velocity-relocation.json) and
[portable segmentation receipt](../planning/portable-segmentation.json) record
an earlier source organization in the Cellerator checkout. Those JSON records
remain unchanged and describe their recorded paths and source snapshot. They do
not describe the current package boundary.

Earlier documentation and receipts describe an FP16 numeric-view migration,
Cellerator binary16 helper, and quarantined AVX2/CUDA implementations. They are
historical evidence for those older source snapshots only. They are not active
dependencies, targets, or implementation evidence for this standalone package.
See [validation](validation.md) for the preserved qualification status.

The standalone repository is `tumlinso/velocitypp`. The ccell submodule keeps
its local directory name `velocity/`.
