# Architecture and design constraints

## Purpose and current boundary

velocity.cppm is intended to make RNA velocity mathematics directly composable
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

The repository currently establishes intent, source evidence, contributor
guidance, and workflow. Numerical implementation and its plan are deferred to
the next user-directed step. Module names beyond the intended `velocity` entry
point, public types, exact substitution mechanisms, dependency packaging, and
implementation order are not settled by this document.

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

velocity.cppm must remain useful independently of Python, Torch, Cellerator,
and CUDA. Cellerator may consume its contracts, implementations, and fixtures;
the dependency does not reverse. Any shared device-compatible numerical helpers
must have an explicit compilation boundary rather than requiring all code to
use one compiler or execution model.

Reuse [anndata.cppm](https://github.com/tumlinso/anndata.cppm) for AnnData data access and
serialization. An integration boundary can select layers/metadata and acquire
native views, while numerical operations remain usable with external buffers
or synthetic inputs. Exact adapter dependencies and packaging await planning.
