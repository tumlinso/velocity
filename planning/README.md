# Planning state

The JSON plans and receipts below are preserved historical records for earlier
`velocity` source snapshots. They retain their original qualification status
and are not rewritten to describe the current `velocitypp` package. None of
their source-bound numerical, performance, FP16, or GPU results qualifies the
current direct-view source.

The completed [first-party relocation](velocity-relocation.json) records the
earlier move into a Cellerator component and its separate quarantine. The
original checkout retains Git/workflow state and raw artifacts; no builds or
tests were run for that move.

The [portable source segmentation](portable-segmentation.json) records an
earlier move of CUDA/AVX2 code and accelerator support outside that source
snapshot's active build. Its later relocation and FP16 ownership notes describe
the previous component layout; they are not the current standalone boundary.
See [the current source boundary](../docs/segmentation.md). These relocations
made no data-package changes and ran no build or test workflow.

That historical setup established project intent, scientific source evidence,
contributor guidance, Project Control registration, and a native Todo authority.

The user approved and authorized [Milestone 1](milestone-1.json): first moments,
steady-state velocity, bounded workspace, customization, and qualification.
The [workflow receipt](milestone-1-workflow.json) distinguishes the accepted
plan from the context-construction failure after native attempt admission.
The user explicitly instructed implementation to proceed without a handle.
Local test/measurement results are not a native accepted outcome.

The user has authorized implementation of [Milestone 2](milestone-2.json):
bounded-memory second-order moments and stochastic velocity, with consistent
fit-row masks, selected-gene refinement, explicit deterministic fallback,
replaceable numerical stages, and independent numerical/performance
qualification. The [local execution receipt](milestone-2-workflow.json)
records the inherited waiver of the unavailable handle. Implementation and
local native, estimator, memory, and module-build qualification are complete.
Four source-initialization offset diagnostics exceed their original tolerance;
the native calculation matches an independent FP64 oracle. This explicit
[qualification exception](../docs/validation.md) remains recorded in the strict
source comparison, rather than being accepted as a tolerance pass. The
Milestone 1 authority and its admitted attempt remain preserved. Local evidence
does not establish native workflow acceptance.

The user also authorized the coordinated [device and precision milestone](milestone-3.json)
with direct implementation in the sibling anndata repository. It covers binary16
and file support, shared device-data contracts, FP32/FP64 scalar and AVX2
computation, complete CUDA execution of the existing methods, explicit source
promotion, advisory screening, and combined qualification. Cellerator remains
outside its scope. Its source-bound dependencies and local receipts are recorded
in both repositories. The execution-handle waiver continues; Scheduler admission
and local validation do not manufacture native Todo outcome acceptance.

For the recorded Milestone 3 source snapshot, implementation and local
qualification were complete. The
[validation receipt](../tests/results/milestone-3.json),
[per-type numerical errors](../tests/results/precision-errors.json),
[runtime/memory measurements](../benchmarks/results/milestone-3.json), and
[build measurements](../benchmarks/results/module-build-milestone-3.json)
retain incremental source-bound evidence. All backend gates pass. One FP32
source R² miss remains as a narrow, independently examined precision exception;
the original strict report and accepted M2 exception remain unchanged.
Specialized AVX2 and CUDA paths demonstrated no end-to-end gain on the measured
sparse workload. All task GPU leases have verified cleanup.

The [setup receipt](setup-receipt.json) records verified bootstrap outcomes and
any remaining runtime limitation for the earlier standalone checkout. It is
sanitized historical evidence, not workflow authority. Its native authority
files remain in the original checkout and do not establish current Project
Control state.

Use the [architecture](../docs/architecture.md),
[scVelo foundation audit](../docs/scvelo-foundation.md), and
[validation](../docs/validation.md) for scope and evidence. Future scientific
families require their own user-directed milestones.

## Current standalone package boundary

The current package is Velocity++ with modules `velocitypp` and
`velocitypp.precision`, namespace `velocitypp`, and installed target
`velocitypp::velocitypp`. It uses FP32/FP64 AnnData++ arrays directly through
`anndatapp::ArrayView`, `ArrayEdit`, and `DType`. FP16/binary16 support,
Cellerator dependencies and aliases, and CUDA/AVX2 implementations are outside
this standalone checkout. The root `ccell` wrappers provide `ccell.velocity`;
`ccell` owns preprocessing. The standalone repository is `tumlinso/velocitypp`;
its local ccell submodule directory remains `velocity/`.
