# Numerical validation and efficiency evidence

## Current evidence

No velocity.cppm algorithms, numerical tests, or benchmark results exist yet.
The [source audit](scvelo-foundation.md) records existing scVelo, accelerated-port,
and Cellerator evidence with its limits. Those results belong to their inspected
sources and workloads; they are not validation of this repository.

This document sets evidence expectations for later authorized implementation.
It does not begin experiments or select the implementation plan.

The acceptance target is numerical similarity to scVelo, not bitwise identity
or exact result equality. Improved numerical algorithms and solvers are allowed.
Planning must define meaningful comparison quantities, tolerances, and treatment
of discrete outputs such as masks; numerical differences alone are not failures.

## Establish the scientific contract

For each operation record the equations, input/output ordering and shapes,
normalization, graph/moment conventions, scaling, time and velocity units,
masks, invalid/degenerate cases, and expected outputs. Declare storage,
arithmetic, accumulation, and output types separately where they differ.

For fitting also record initialization, parameter bounds, objective,
stopping/convergence conditions, latent-variable conventions, stochastic seeds,
and diagnostics. Equivalent velocity predictions need not imply uniquely
identified parameters or absolute biological time.

## Independent correctness evidence

- Use analytic solutions, hand-computable cases, and mathematical invariants
  where available, including degenerate rates, empty selections, signed inputs
  where admissible, and normalization boundaries.
- Pin upstream scVelo source/version and dependencies for differential checks.
  Reproduce its method-specific options; record deliberate improvements and
  unsupported cases rather than silently treating every behavior as universal.
- Compare useful intermediate quantities as well as final velocities: sufficient
  statistics, moments, objective values, residuals, fitted trajectories, masks,
  and convergence status as appropriate.
- Define absolute/relative tolerances and near-zero handling before judging
  results. Report nonfinite behavior, reduction sensitivity, and the qualified
  precision range. Exact equality is not the project acceptance requirement.
- Test real AnnData integration independently from numerical-buffer tests using
  anndata.cppm and pinned upstream readers when that adapter is implemented.

The native reference is useful for regression and later Cellerator comparison;
analytic and upstream checks remain necessary to detect defects shared by both
implementations. Passing numerical checks does not establish general biological
validity or identifiability.

## Reference and custom implementations

Reference execution must be explicitly selectable and bypass custom
substitutions. Preserve the same input case, method options, ordering, and
reproducible stochastic state for comparisons. Verify relevant behavior for
individual components and composed operations.

Implementations may use different representations, allocations, batched/fused
execution, internal algorithms, solvers, and precision within the declared
numerical-similarity contract. Identify consequential approximations and their
qualified range. Changes to the scientific model or objective are explicit
variants with acceptance conditions appropriate to that difference.

## Efficiency evidence

Freeze source identity, inputs/seeds, toolchain, flags, thread counts, numerical
method, and dtype. Compare the same requested quantities under the declared
numerical-similarity requirements and include setup,
conversion, allocation, and materialization costs where the claim requires them.
Measure runtime and memory separately enough to explain the mechanism.

Record peak process/working memory, allocation calls and requested bytes,
persistent outputs, scratch capacity, and relevant traffic or backend calls.
Cumulative allocation is not peak memory; a selected-block measurement is not
an end-to-end memory bound. Include adversarial geometry such as sparsity,
neighborhood sizes, feature batches, and candidate-time grids when relevant.

Begin with representative bounded probes, initially capped at 30 seconds.
Stop repeating cases after a bottleneck is established. Larger confirmation
requires an explicit time and memory/I/O budget. Preserve the baseline, raw
evidence, and exact comparison conditions. Do not infer a speedup from source
changes or extrapolate blindly from small cases.

Future GPU evidence must distinguish resident execution from transfers,
preparation, launch, and end-to-end costs. Use the authorized Scheduler and
isolate measurements. No GPU performance qualification is claimed here.

## Native and development checks

When implementation begins, establish native numerical/ownership tests,
ASan/UBSan checks, and independent installed C++20 module consumers. Measure
cold/import and incremental consumer builds, including a private-implementation
edit that should not rebuild consumers. No such build configuration or gate
exists at this setup stage.
