# Native API and execution contracts

`import velocitypp;` exports the standalone velocity interfaces and
`anndatapp.core`. The public modules are `velocitypp` and
`velocitypp.precision`, the C++ namespace is `velocitypp`, and the installed
CMake target is `velocitypp::velocitypp`. Inputs and outputs use AnnData++
`ArrayView`, `ArrayEdit`, and `DType` directly. The supported expression and
graph storage types are FP32 and FP64.

## Acquisition and preparation

`Input` contains spliced/unspliced and weight `anndatapp::ArrayView`s, an
`anndatapp::Axis` gene selection, and optional AnnData fit-row and gene masks.
`from_anndata()` acquires the named layers and pairwise array without changing
the dataset; its default keys are `spliced`, `unspliced`, and `connectivities`.
The supplied weight matrix is used as supplied. The gene axis and optional
fit-row/gene masks remain AnnData types; fit diagnostics are AnnData arrays.
Call `reference::prepare_neighborhood()` explicitly to obtain positive-support,
self-linked, row-normalized scVelo-style weights.

`PreparedInput` retains AnnData++ array views and prepares traversal metadata.
Pass FP32/FP64 views directly; no additional numeric view wrapper is needed.
Dense signed strides and unselected CSR/CSC layouts are supported. Sparse
selections and irregular dense gathers require the caller's explicit copy;
alternatively select genes through `Input::genes` while passing the original
views. Sparse index and offset widths are acquired independently.

Owned arrays remain alive through retained descriptors. Wrapping external
borrowed storage does not transfer ownership: keep its bytes alive. Keep shapes,
strides, topology and selected coordinates fixed during preparation and use;
structural changes require re-preparation. Value updates may occur between
runs, never concurrently with execution.

### Direct AnnData++ array views

Use `anndatapp::ArrayView` for expression, weights, and optional masks, and use
`anndatapp::ArrayEdit` for caller-owned writable outputs. `anndatapp::DType`
describes the underlying array type. This package accepts FP32 and FP64 numeric
arrays; unsupported dtypes are rejected during checked preparation.

AnnData++ owns array descriptor, stride, sparse-layout, and value-access
semantics. Keep externally borrowed storage alive for every view that uses it.
Preserve array structure and selected coordinates for the prepared run; changes
to values may occur between runs, not concurrently with execution.

Full validation checks values at construction and at each reference run.
`Validation::none` omits optional value-content scans; necessary shape/layout and
structural acquisition still occurs. `validate_values()` explicitly scans values.

## Workspace and results

`Workspace(input, RunOptions)` owns reusable moment, percentile and conversion
scratch. The default budget is 64 MiB and the maximum block width is 64 genes.
`scratch_bytes()` reports accounted numerical storage; `storage_bytes()` on the
prepared input separately reports its traversal storage. Inputs, returned
diagnostics, and caller-retained outputs are separate costs.

`reference::steady_state()` emits requested fields to `BlockSink` and returns
`FitResult`. `reference::steady_state_into()` writes to a caller-supplied dense
`anndatapp::ArrayEdit`. Velocity is requested by default; moment output is opt-in. The
default sink velocity dtype is FP32 for FP32 computation, and otherwise the
widest expression dtype with a minimum of FP32. `RunOptions::output_dtype` uses
`anndatapp::DType` to select an FP32 or FP64 output.
`RunOptions::compute_precision` chooses FP32 or FP64 for moments, fitting,
reductions, and diagnostics. Gene order is the requested order.

Sink views alias reusable workspace and must be consumed before the callback
returns. Copy explicitly when retaining a block. Execution is synchronous and
single-threaded; concurrent runs require separate workspaces and disjoint
outputs. Outputs must not overlap inputs or scratch.

`FitResult` owns compute-type `gamma`, `offset`, and `r2` arrays, a boolean `genes`
selection array, uint8 `status`, and a low-signal recovery flag. `FitStatus`
distinguishes ordinary fits, insufficient data, absent signal, singular fits,
and constant responses. Invalid fits remain excluded from gene selection.

## Fitting and custom operations

`FitOptions` defaults to tails `(5,95)` with no intercept (therefore upper 95th
percentile selection), unbounded gamma, unadjusted R², and minimum R²/ratio .01.
It supports all-data/upper/tail selection, nonnegative intercept fitting, bounds,
adjusted R², and explicit masks. Requested filters remain in force by default.
`LowSignal::scvelo_percentile` opts into relaxed recovery, with activation
reported. Undefined/singular quantities have finite neutral parameters and
explicit status instead of silently propagating undefined fits.

`ReferenceStages` exposes `moments`, `fit`, `emit`, and `finish`. Compose a custom
callable object around these stages to replace individual parts. The small
`steady_state_with()` driver also supports a whole-block `process_block()`
operation. Custom stage values may be opaque handles; the driver never acquires
host numerical buffers from them. See the customization test for both routes.

`reference::steady_state()` uses fixed compiled reference stages independently
of custom operations. No global registration or runtime backend switch can
intercept its nested mathematics. The root `ccell` wrappers can replace or fuse
operations while retaining this explicit comparison/fallback path.

The standalone `reference::moments`, `fit_gene`, and `residuals` primitives expose
the underlying calculations. Borrowed raw-pointer primitive inputs carry their
documented extent/stride/workspace preconditions; checked acquisition belongs
outside element loops. Numerical similarity, not exact identity to scVelo,
governs qualification.

The existing double primitives and stage aliases remain available. `Basic<T>`
blocks and the `32` stage aliases expose FP32 computation without templates for
the scientific implementation in consumer modules. `velocitypp::cpu` calls
compiled portable CPU helpers; `velocitypp::reference` calls scalar kernels. See
[precision](precision.md) for source overrides and sensitivity diagnostics,
and [segmentation](segmentation.md) for the excluded accelerator archive.

## Stochastic estimator (Milestone 2)

The stochastic operation uses the same retained numeric views and literal
supplied weights as the deterministic path. `reference::prepare_neighborhood()` remains
an explicit choice. It derives `A = 2*W(S*S) - Ms` and
`B = 2*W(S*U) + Mu`; optional `Muu = W(U*U)` is evaluated only when requested.
These adjusted scVelo quantities are not labeled centered variance/covariance.

Stochastic refinement initializes from deterministic fits and their selected
genes. Selected genes are refined by default. If the selection is empty, the
reference retains deterministic results; matching scVelo's refit-all behavior
is an explicit option. Explicit gene filters remain active. The fit-row mask
applies consistently to initialization, percentile selection, scale
estimation, and joint fitting. Final velocity predictions cover every row.

The first scale is the population standard deviation of the deterministic
first-order residual. The second is estimated from an untrimmed regression of
`B` on `A`; `fit_second_intercept` controls its intercept and
`initialization.positive_intercept` controls whether a negative initial
intercept falls back to zero. The joint fit minimizes
`sum((r1/sigma1)^2 + (r2/sigma2)^2)`, normalized by twice the number of fit
rows for the reported objective, with
`r1 = Mu - gamma*Ms - offset` and `r2 = B - gamma*A - offset2`. Percentile
exclusions zero first-order expression and response while leaving intercept
rows and second-order observations in the objective. The optional second-order
output residual `(B - 2*Ms*Mu) - gamma*(A - 2*Ms*Ms) + 2*offset*Ms - offset2`
is a different quantity from the GLS residual. The fit supports the first
intercept through `initialization.fit_intercept` and a separate second
intercept through `fit_second_intercept`; both final intercepts are
unconstrained. A slope bound applies to the final fit, with enabled intercepts
recomputed when the bound is active. The scaled rank test treats a singular-value
ratio at or below `1e-12` as rank deficient. A zero/nonfinite scale, too few
rows, or rank-deficient fit retains a usable deterministic estimate and returns
a distinct fallback status.

The stochastic workspace is separate from deterministic workspace. Numerical
scratch is bounded by configured workspace bytes and block width; full
second-moment fields are not retained. Numerical source-column acquisition
sums duplicate expression coordinates before nonlinear products on each run,
respecting value updates. Graph duplicates remain additive. Callers can request
streamed blocks or write to caller-owned outputs, and may select custom
implementations of moments, scales, fitting, or whole-block processing. The
`reference::` operation is compiled independently and bypasses these custom
stages.

The public interface names these contracts directly:

- `StochasticOptions` contains deterministic `initialization` options,
  `fit_second_intercept`, and `refit_all_if_none` (default false).
- `StochasticRunOptions` bounds `workspace_bytes` (default 64 MiB) and
  `block_genes` (maximum 64). It selects velocity by default and second
  residuals, first/second moments, and output dtype explicitly.
- `StochasticWorkspace::scratch_bytes()` reports accounted numerical buffers;
  it does not include retained inputs, prepared traversal metadata, returned
  `StochasticFitResult` diagnostics, or outputs retained by the caller.
- `StochasticFitResult` owns the deterministic `initialization` diagnostics and
  final arrays `gamma`, `offset`, `offset2`, `first_scale`, `second_scale`,
  `status`, and normalized `objective`. `refit_all` reports fallback-policy
  activation. `StochasticStatus` distinguishes success, not selected, invalid
  initialization, insufficient data, zero scale, singularity, and nonfinite
  refinement.
- `StochasticMomentBlock` carries row-major pointers and an explicit row stride
  for `Ms`, `Mu`, `Mss`, `Mus`, and optionally `Muu`. Its addresses are borrowed
  for the stage operation. `StochasticOutputBlock` exposes only requested
  outputs as `anndatapp::ArrayView`s. Sink views alias reusable workspace and
  must be consumed before the callback returns.

`StochasticReferenceStages` is the replaceable pipeline contract. The caller
runs `initial_moments`, `initial_fit`, and `accept_initialization` for every
gene block, then `resolve_selection`, followed by `moments`, `scales`, `fit`,
and `emit` for each block. Blocks are ordered, nonempty, within
`StochasticWorkspace::block_genes()`, and cover the selected gene sequence.
Custom stages may return opaque handles through `stochastic_with`; they do not
need to expose host numerical buffers. A fused `process_block` operation may
replace the second pass. `reference::stochastic()` is the compiled reference
composition. `reference::stochastic_into()` writes caller-owned dense
FP32/FP64 velocity and optionally second-residual outputs with shape cells
by prepared gene order;
output arrays must not overlap each other, input, or workspace.

Standalone `reference::second_moments()` accepts optional pointers for each
requested raw field, with `n_cells()*count` contiguous doubles per field.
Its two caller-owned source-column buffers each hold `n_cells()` doubles when
the corresponding expression is needed. Outputs and scratch must be mutually
disjoint and must not overlap retained input; checked preparation revalidates
values on entry. `reference::stochastic_residuals()` writes a single borrowed
moment/fit block into matching dense outputs and checks overlapping spans.

`reference::stochastic_scales()` and `stochastic_fit_gene()` expose per-gene
calculations on borrowed double arrays: each pointer addresses the first row,
successive rows are `stride` doubles apart, and an optional mask has one byte
per row. The fitter's score scratch holds at least `rows` doubles and must not
overlap observations or other live state. Callers guarantee actual buffer
capacities and lifetimes; entry checks cover pointers, options, and stride
arithmetic. No allocation or checked array acquisition occurs inside their
numerical loops.
