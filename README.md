# velocity.cppm

A native C++20 RNA velocity toolkit, built around named modules, explicit
numerical memory, and composable scientific algorithms.

The intended foundation is a substantially more efficient implementation of
scVelo's mathematics, especially in its use of memory. Results should be
numerically similar to scVelo; bitwise identity and exact result equality are
explicitly not requirements. The reference should be
a useful, efficient library in its own right: direct calculations, deliberate
allocation, reusable buffers, and practical C++ compilation and iteration.

**Status:** project documentation and workflow setup. No velocity algorithms,
module interfaces, build system, numerical tests, or performance improvements
have been implemented here. Implementation planning is the next step.

## Scientific direction

The scientific scope includes neighborhood statistics, steady-state and
stochastic velocity estimation, transcription/splicing/degradation dynamics,
parameter and latent-time inference, and the associated diagnostics. Velocity
graphs and transition calculations are related, separately composable
operations. Their implementation order and exact public interfaces remain to
be planned.

scVelo supplies mathematical and behavioral reference material. Its Python
object structure, allocation patterns, dense intermediates, and whole-dataset
workflow need not determine the native implementation. Changes to models,
objectives, preprocessing, precision, and statistical conventions must remain
explicit and testable. Different numerical algorithms and solvers are welcome
when they meet the declared numerical-similarity and scientific requirements.

## Cellerator and reference behavior

Cellerator is expected to develop GPU-accelerated and custom implementations
later. It must be able to reuse, specialize, replace, and fuse meaningful parts
of the computation, including numerical evaluation, fitting, updates, and
batch execution. velocity.cppm retains an explicitly callable reference path
for the same declared method and inputs, with downstream substitutions bypassed.

The shared contract establishes scientific meaning and numerical expectations;
implementations can choose different layouts, working sets, and execution.
Reference behavior and fixtures should be stable enough for regression and
cross-implementation checks without freezing the library's architecture before
it has been designed.

The library remains independent of Cellerator, Python, Torch, and CUDA.
AnnData interoperability should reuse [anndata.cppm](https://github.com/tumlinso/anndata.cppm)
and its native memory access rather than duplicate its storage codecs. Exact
adapter packaging and dependency choices belong to the next planning step.

## Project documents

- [Architecture and design constraints](docs/architecture.md)
- [scVelo mathematics and existing acceleration evidence](docs/scvelo-foundation.md)
- [Numerical validation and efficiency evidence](docs/validation.md)
- [Development and Project Control](docs/development.md)
- [Coding style](STYLE.md) and [agent guidance](AGENTS.md)
- [Planning state](planning/README.md)
