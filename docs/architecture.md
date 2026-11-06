# Architecture and design constraints

Velocity++ is the standalone C++20 package in this standalone checkout. Its
public modules are `velocitypp` and `velocitypp.precision`, and its installed
target is `velocitypp::velocitypp`. It depends on AnnData++ 1.0 or newer and
uses FP32/FP64 existing AnnData++ byte descriptors acquired by callers.
The standard reference is allocation-free in every phase; topology, workspace,
state and results are borrowed caller storage.
Earlier FP16, AVX2, and CUDA source and qualification records are historical;
they do not describe the current package. See
[the current boundary](segmentation.md).

## Purpose and current boundary

Velocity++ makes RNA velocity mathematics directly composable
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

The scientific baseline includes first moments and steady-state velocity,
followed by second-order neighborhood moments and stochastic fitting. The
precise scientific and API contract is in the [native API](api.md); the
[Milestone 1](../planning/milestone-1.json) and
[Milestone 2](../planning/milestone-2.json) records describe earlier source
snapshots. Dynamical and Bayesian fitting, velocity graphs, backed orchestration,
and Python runtime bindings remain outside this standalone package. The
coordinated [Milestone 3](../planning/milestone-3.json) receipt records older
precision and device work, not current package capabilities.

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

## ccell integration and explicit fallback

The root `ccell` wrappers provide `ccell.velocity`, and preprocessing remains
owned by `ccell`. Velocity++ supplies the standalone numerical algorithms and
explicit reference path; it does not depend on ccell or the Cellerator runtime.

Downstream code must be able to reuse or replace meaningful internal parts:
model evaluation, objectives and derivatives where applicable, statistics,
parameter updates, convergence checks, fitting strategies, batching, and memory
management. A whole-operation replacement must also remain possible. A
downstream implementation may replace connected regions or the whole algorithm
with novel layouts, intermediates and mathematical methods. Only interfaces to
retained code impose compatibility; reference intermediate structures are optional.

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

The reference creates no allocation owners. Callers acquire AnnData++ numerical
views and supply every topology/result/workspace buffer. Expose relevant
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

The standalone library is usable without Python, Torch, CUDA, or ccell. Its
data dependency is AnnData++ 1.0 or newer.

Use [AnnData++](https://github.com/tumlinso/anndatapp) for supported AnnData
access and serialization. Its FP32/FP64 arrays and views are native inputs and
outputs, including externally borrowed storage. Velocity++ imports
`anndatapp.core` and borrows its caller-acquired byte descriptors with explicit
`DType` tags. Reference results and scratch are caller-provided numerical buffers. It has no separate numeric-view adapter or binary16 dependency.

## Storage and arithmetic

Storage and computation are separate choices. Graph weights and expression
arrays may use FP32 or FP64 storage, passed directly as AnnData++ views.
Numerical blocks use FP32 or FP64 for products, moments, scales, fitting, and
diagnostics. Explicit computation precision wins. An optional selective
precision composition can acquire chosen original FP64 columns; original data
must match the prepared cell order, normalization, and units. The sensitivity
report is advisory and never promotes genes automatically.

The scalar reference remains explicitly callable. `velocitypp::cpu` uses
compiled portable CPU helpers, with no ISA selection or persistent CPU packing
contract. The standalone package has no CUDA or AVX2 implementation, toolkit
discovery, device allocation, or accelerator target. Historical GPU receipts
and source archives do not qualify this package.

[Selective precision](precision.md) is an independent stage composition.
Only a caller mask requests original columns. Available requested columns
replace expression moments before global selection and remain fixed for
refinement. W is unchanged. The optional trimming sensitivity report is
advisory and never promotes genes automatically.
