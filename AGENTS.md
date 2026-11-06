# Velocity++ standalone package

This directory contains the standalone C++20 RNA velocity package. Its source
checkout inside ccell remains `velocity/`; its repository, package, module,
namespace, and target identity is `velocitypp`.

The public named modules are `velocitypp` and `velocitypp.precision`. The
installed CMake package is `velocitypp`, and consumers link
`velocitypp::velocitypp`. Use the `velocitypp` namespace throughout examples
and current documentation.

The standalone numerical boundary uses FP32 and FP64 arrays directly through
`anndatapp::ArrayView`, `anndatapp::ArrayEdit`, and `anndatapp::DType`. Do not
add parallel numeric view/edit adapters. This checkout has no FP16 or binary16
path, Cellerator dependency, Cellerator alias, CUDA backend, or AVX2 backend.
AnnData++ is independently owned; do not edit its sources or rename its package,
module, or type identifiers here.

The root `ccell` wrappers provide `ccell.velocity`; preprocessing remains owned
by `ccell`. Keep this standalone package focused on velocity algorithms and
their direct AnnData++ inputs and outputs.

Build requirements are CMake 3.28 or newer, Clang 18 with its matching
`clang-scan-deps`, Ninja, and AnnData++ 1.0 or newer. Current CMake options use
the `VELOCITYPP_*` prefix. Example and qualification executables use the
`velocitypp_*` prefix.

Read README.md, docs/api.md, docs/architecture.md, docs/development.md,
docs/precision.md, docs/segmentation.md, docs/validation.md, and STYLE.md
before changing public behavior. Keep compiled numerical implementations
private and module imports small.

Historical JSON receipts and raw artifacts preserve earlier source identities,
including the former FP16 and GPU work. Do not edit those receipts or present
them as qualification for the current `velocitypp` source. See
docs/validation.md and planning/README.md for the historical boundary.
