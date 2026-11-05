# Architecture and design constraints

The portable CPU component now lives under `Cellerator/src/velocity/` and can
be built independently. Cellerator parent-build integration is optional and
disabled by default. CUDA and AVX2/F16C code is preserved in an excluded
quarantine. See [the current boundary](segmentation.md); references below to the
coordinated CUDA milestone describe its historical implementation.

## Purpose and current boundary

velocity is intended to make RNA velocity mathematics directly composable
in native C++20. Its reference implementation should substantially improve on
scVelo's efficiency, especially memory use, while retaining explicit scientific
contracts. Efficient operation and understandable calculations are requirements
of the reference itself.

The numerical target is similarity to the declared scVelo comparison, explicitly
without requiring bitwise identity or exact result equality. Different numerical
algorithms, solvers, reduction orders, and precision choices can be appropriate.
Their output differences must be assessed using scientifically meaningful
quantities and tolerances chosen during planning, rather than an assumed demand
to reproduce every intermediate or source-level numerical choice.

Milestone 1 implements first moments and steady-state velocity. Milestone 2
adds second-order neighborhood moments and stochastic fitting; its precise
scientific and API contract is in the [native API](api.md) and
[milestone record](../planning/milestone-2.json). Dynamical and Bayesian
fitting, velocity graphs, backed orchestration, Python runtime bindings, and
GPU implementations remain outside Milestone 2. The coordinated
[precision milestone](../planning/milestone-3.json) adds those implementations
for the existing methods; it does not introduce another velocity algorithm.

Stochastic refinement uses the deterministic fit to initialize parameters and
select genes. By default it refines selected genes only; if none qualify, the
deterministic result remains in force. Refit-all is an explicit option limited
to enabled genes. Fit-row masks apply consistently to initialization,
second-order scale estimation, and joint fitting, while output predictions
cover all cells. This is an intentional departure from scVelo's all-row
stochastic fit behavior.

The stochastic objective uses per-gene scales to weight `r1 = Mu - gamma*Ms - offset`
and `r2 = B - gamma*A - offset2`. The optional emitted second-order residual
uses the centered expression formula and is distinct from `r2` in the objective. Singular or non-finite refinement keeps
the usable deterministic estimate and reports a fallback status. It does not
silently replace a failed estimate with a neutral stochastic parameter.

## Scientific components

Useful scientific boundaries include neighborhood statistics, regression and
velocity evaluation, kinetic trajectories, objectives and likelihoods,
parameter/latent-time estimation, diagnostic quantities, and graph-based
transition calculations. These are composable responsibilities, not a required
class hierarchy or fixed pipeline.

An operation's contract must specify its mathematical inputs and outputs,
observation/feature ordering, expression representation, normalization, graph
and moment conventions, parameter scaling, time convention, and relevant
precision. Counts, normalized values, smoothed moments, fit parameters, and
velocities must remain distinguishable without requiring a large type system.

scVelo is mathematical and behavioral reference material. Its mutable AnnData
workflow, implicit defaults, Python data structures, broadcasting, allocation
patterns, and execution order need not become the native architecture. Consult
the [foundation audit](scvelo-foundation.md) before treating similarly named
quantities or accelerated mechanisms as equivalent.

## Efficient reference implementation

Prefer direct numerical storage and calculations, bounded working sets,
deliberate materialization, reusable outputs/workspace, and evaluation of only
requested results. Preserve sparse inputs where the algorithm permits; define
the representation of inherently dense outputs explicitly. Small selections
and individual genes should be useful without mandatory whole-dataset work.

Allocation, layout, batching, traversal, and reuse can change deeply. Declare
and test any consequential differences in floating-point evaluation, tie
handling, masks, initialization, optimization, precision, or approximation.
Source inspection can identify opportunities; measured improvement requires
paired evidence under the [validation contract](validation.md).

The reference should be stable in its declared behavior and reproducible test
cases. It can evolve with deliberate contract changes and updated evidence;
stability does not require prematurely freezing every C++ interface.

## Deep Cellerator control and explicit fallback

Cellerator's GPU-accelerated and custom velocity implementations are planned.
Existing Cellerator numerical mechanisms and accelerated-ports probes supply
evidence and possible reuse; they do not establish a complete implementation
of this toolkit.

Downstream code must be able to reuse or replace meaningful internal parts:
model evaluation, objectives and derivatives where applicable, statistics,
parameter updates, convergence checks, fitting strategies, batching, and memory
management. A whole-operation replacement must also remain possible. Cellerator
may change layouts, fuse stages, keep device-resident state, and omit unrequested
intermediates while preserving the operation's declared results.

Keep reference invocation explicit and independent of downstream substitutions.
A comparison or fallback must be able to run the same declared method and input
case through the baseline path. Capture method options and any stochastic state
needed for reproduction. Reference execution must not silently dispatch back
into the custom implementation under test.

Interchangeability is established per contract, not by a common function name.
A different model, likelihood, objective, approximation, or fitting method is
an identifiable variant with its own acceptance conditions. Compare invariant
properties and appropriate diagnostics when exact parameter or output agreement
is not scientifically justified.

Ordinary C++ composition, free functions, plain descriptors, callable objects,
and useful small templates are starting tools. Choose actual extension seams
from real computations during planning. A universal backend registry, tensor
facade, execution runtime, or general plugin framework is not a prerequisite.

## Memory, ownership, and execution

Separate allocation owners from acquired numerical views. Expose relevant
pointers, extents, strides, sparse widths/orientation, mutability, and aliasing.
Document retained and borrowed lifetimes and the work performed at acquisition
or conversion boundaries. Permit explicit trusted access where useful.

Callers control parallel execution and synchronization. Scientific operations
must make their required working storage and ownership visible. Device adapters
must account explicitly for residency, transfer, completion, and lifetime;
host-addressable pointers must never masquerade as device access or vice versa.
Shared scientific meaning does not require identical host and device layouts.

## Modules and dependency direction

Use C++20 named modules and `.cppm` public interfaces, with small exported
surfaces and independently compiled implementations. Keep heavy dependencies
private. Fast imports and incremental consumer compilation are important
deliverables to measure, rather than assume from the use of modules.

The CPU library remains usable without Python, Torch, CUDA, or the full
Cellerator runtime. It depends on AnnData++ for core arrays, axes, and AnnData
access, and on Cellerator's small CPU-only binary16 conversion helper. The
helper target is `Cellerator::binary16`; its installed CONFIG package is
`CelleratorBinary16`, found transitively by the installed velocity package.
This helper is not the full Cellerator runtime.

Use [AnnData++](https://github.com/tumlinso/anndatapp) for supported AnnData
access and serialization. Its FP32/FP64 arrays and views are native inputs and
outputs, including externally borrowed storage. AnnData++ integer arrays are
not an FP16 representation, and its storage codecs and device operations are
not part of the velocity FP16 contract. The required in-tree CMake targets are
`anndatapp::anndatapp` and `Cellerator::binary16`; velocity re-exports
`anndatapp.core` and its numeric-view types through the public named modules.

## Storage and arithmetic

Storage and computation are separate choices. W, S, and U may independently
use FP16, FP32, or FP64 storage. FP16 is supplied as explicit raw-bit
`NumericView`s for dense or CSR/CSC host data; `NumericEdit` supports direct
dense half output. FP32/FP64 AnnData views adapt implicitly. The numerical
kernels call `cellerator::numeric::load_f16(std::uint16_t)` when acquiring half
values and `round_f16(float/double)` when a caller requests half output. The
header-only helper is CPU-only and has no dependency on the full Cellerator
runtime. Velocity does not allocate a full FP32 shadow of half inputs or
outputs, and conversion stays within its bounded numerical traversal.

Numerical blocks use FP32 or FP64 for conversion, duplicate summation,
products, moments, scales, fitting, and diagnostics. Automatic computation
selects FP32 when any W/S/U input is FP16 and none is FP64; otherwise it selects
FP64. Explicit computation precision wins. FP16 output requires a caller
request. The ordinary FP16/FP32 path does not acquire original sources, screen
genes, or retry at higher precision. Selective precision remains a separately
composed CPU path; original sources for promotion must be FP32/FP64.

The scalar reference remains explicitly callable. `velocity::cpu` uses compiled
portable CPU helpers, with no ISA selection or persistent CPU packing contract.
The former CUDA module and its copies of shared arithmetic are in
quarantine (`Cellerator/quarantine/velocity-accelerators/` in the parent checkout). The active
public interfaces and private arithmetic have no CUDA annotations or intrinsic
headers. The CPU package does not discover a toolkit, allocate device storage,
or export accelerator targets. The historical accelerator archive and receipts
do not qualify the current FP16 view/helper path.

[Selective precision](precision.md) is an independent stage composition.
Only a caller mask requests originals. Available requested columns replace
expression moments before global selection and remain fixed for refinement;
missing originals retain half storage. W is unchanged. The optional trimming
sensitivity report is advisory and never promotes genes automatically.
