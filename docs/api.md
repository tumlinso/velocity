# Native caller-storage API

`import velocitypp;` exposes deterministic and stochastic RNA-velocity
calculations. `import velocitypp.precision;` adds advisory sensitivity and
explicit original-column promotion. All standard reference sizing, validation,
preparation, binding and execution allocate nothing. This is a development API
cutover: the former allocating prepared/workspace/result owners are removed.

## Acquisition, lifetime and validation

The caller constructs AnnData++ storage and acquires addressable byte descriptors
before entering the reference. `DenseInput` and `SparseInput` are aliases of the
existing AnnData++ descriptor types, not independent matrix/storage types.
`AcquiredMatrix` borrows a pointer to one descriptor together with its dtype,
layout and independent sparse-index/offset dtype tags. Descriptors and underlying
allocations must outlive every call and bound state using them. Acquired dense
descriptors must have representable byte strides; deliberately malformed trusted
AnnData descriptors are outside the numerical contract. Callers perform any
allocation-capable descriptor acquisition/stride diagnostics before entry. Dense strides
remain signed byte strides in the existing descriptor; no implicit copy or
materialization occurs. Irregular gathers and selected sparse data require
caller-owned acquisition/materialization outside the reference.

`Input` binds spliced/unspliced/weight descriptors, optional borrowed gene indices,
and optional byte masks with signed strides. Layout and dtype tags must describe
the actual descriptor storage; byte descriptors cannot prove same-width type or
orientation provenance. Validation cannot establish the size of an arbitrary
external allocation. Null gene indices select all source
genes; a nonnull pointer and zero count selects no genes. Repeated indices retain
requested order. The fit mask is indexed by source observations, gene mask by
source genes, results by selected-gene position.

`validate(Input)` scans shape/type/structure/value contracts without allocating.
`Status` carries `Code`, location and required capacity; code `ok` means success.
`preparation_size` reports integer and Entry capacities; `prepare` validates then
fills caller-supplied `PreparationBuffers`, returning a borrowed `PreparedInput`.
`prepare_unchecked` assumes valid shapes, values and structures, while retaining
capacity diagnostics. Structural buffers and masks must remain stable during
execution. Changes invalidate topology and require preparation again.

Sparse logical duplicates are summed in original stored order before weighting
and products. Prepared topology stores indices and original positions, never a
shadow copy of expression values. Dense, CSR and CSC inputs, independent integer
widths and FP32/FP64 value storage are supported. Finite expression and weights
are required. Weights are supplied by the caller; neighborhood preparation is a
separate operation, never an implicit normalization step.

## Neighborhood construction

`reference::neighborhood_size` reports sufficient output/scratch capacities for
positive-support/self-loop row-normalized FP32 CSR. Sparse output capacity can
be an upper bound, avoiding quadratic duplicate counting with no scratch.
`prepare_neighborhood` receives caller values/columns/offsets and scratch buffers,
returns actual used entries, and never creates an Array owner. Each logical
positive edge and self edge occurs once, with equal row weight. Output columns
and offsets use native `size_t`; the caller may adopt/copy into AnnData++ storage
outside the reference. Capacity failure reports required capacity before writes.

## Workspace and results

`workspace_size<T>(prepared, options, stochastic, size)` reports scalar/mask
capacities, selected block width, and bytes. `RunOptions::unspliced_second` explicitly
requests optional second-order unspliced storage; it is absent by default.
Stochastic numerical scratch is independent of total gene count at fixed rows
and block width, using a supplied result diagnostic for global selection scratch.
T is float or double and explicitly
chooses computation precision independently of source storage. `RunOptions`
selects maximum block width and numerical workspace budget. Zero block width,
overflow and insufficient budgets are status failures.

`bind_workspace` splits the supplied scalar buffer into visible moment, score,
second-order and source scratch pointers, and binds a separate byte-mask buffer.
The descriptor owns nothing. Every worker requires independent mutable workspace.
Input, topology, workspace, result and numerical output regions must be mutually
disjoint; the arrays within each result are pairwise disjoint. Pointer provenance,
actual backing allocation lengths and these aliasing obligations belong to the
caller. Binding checks options, pointer presence and supplied capacities; it is
not a general allocation-provenance/overlap checker. The caller supplies all
`BasicFitResult<T>` parameter, selection and status arrays;
`BasicStochasticFitResult<T>` additionally binds initialization and stochastic
parameter/scale/objective arrays. Capacities count elements, not bytes. Binding
initializes explicitly required diagnostics; no retained output cache is created.

Reference results distinguish scientific statuses (constant response, singular,
insufficient data, no signal) from argument/capacity `Status`. Deterministic
fitting supports trimming, optional intercepts, row masks, ratio bounds and gene
selection. Low-signal recovery is an explicit option. Stochastic refinement uses
the deterministic initialization, with optional refit-all when none are selected;
invalid or singular refinement retains the usable deterministic estimate.
Fit-row masks apply consistently to initialization and refinement; predictions
cover every observation. No blanket fast-math or contraction change is made.

## Free routines and composition

Raw `fit_gene`, `stochastic_scales`, `stochastic_fit_gene`, moments, residuals and
original-column moment routines accept borrowed pointers and explicit strides.
They are trusted numerical routines: capacities, nonaliasing, valid options,
ranges and stage ordering are caller preconditions. Use explicit validation
before unchecked execution. Inputs are never silently acquired, converted,
sorted into a new owner, or synchronized. CPU reference calls complete within
the call.

`bind_stages` binds caller state/results to deterministic or stochastic reference
stages. Individual moments, fit, scales and emit calls remain usable separately.
`reference::steady_state`/`stochastic` drive these stages using supplied results;
`*_into` additionally write caller numerical output with explicit element
capacities and signed row strides. Capacity spans include gaps between rows;
the pointer denotes logical row zero even for reverse traversal. Invalid pitch,
span overflow and insufficient capacity return before result/output writes.
Compute T and output O are independently float or double; conversion occurs
directly into the chosen output, without full precision-shadow staging. Sinks receive borrowed
reference moment/fit blocks and must consume them before workspace reuse.

`steady_state_with` and `stochastic_with` are narrow exported templates whose
consumer-defined intermediates and results may be entirely unlike the reference.
A `run()` method bypasses the full reference schedule; `process_block()` replaces
a fused region. Stochastic `initialize()` replaces initialization plus selection.
Ordinary reference kernels and FP32/FP64 instantiations remain separately compiled.
Template scheduling requires positive block width and valid operation contracts.

`RuntimeOperations` is a per-execution table of typed function pointers and a
caller-owned context. Intermediate representations and results stay in that
context. A whole-run callback bypasses phases; fused-block callbacks bypass
individual moment/scales/fit/emit calls. An initialization-region callback bypasses
the initial-block loop and selection barrier. Callback failures return their status immediately
without rollback. Callbacks report the usable prefix/phase in their chosen status
contract. Consumers synchronize changes between invocations and own device
submission, dependencies and completion. No global registry or implicit fallback
can route an explicit reference call through a replacement.

## Includes and build boundary

Public interfaces use only necessary scalar headers and existing imported
AnnData++ descriptors. Production numerical implementations use explicit loops
and compiler math/memory builtins; no STL containers, generic STL algorithms or
allocating reference helper survives this cutover. AnnData++ construction,
annotation, storage I/O, and potentially allocating acquisition are caller work.
Consumer replacements may allocate explicitly when their own algorithm needs it.
