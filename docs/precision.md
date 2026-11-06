# Precision and selected originals

`import velocitypp.precision;` provides advisory sensitivity calculations and
explicit selective use of original expression columns. Reference routines use
caller-owned buffers and allocate nothing. The caller acquires AnnData++ views,
allocates outputs and scratch, and retains their storage before execution.
Compute precision is explicit through the selected FP32 or FP64 workspace and
function overload; it does not change source array storage precision.

Reference deterministic and stochastic calculations do not screen or promote
sources automatically. Sensitivity is an advisory result; the caller chooses
whether to request original data and how to use the result.

## Sensitivity to trimming thresholds

`sensitivity_gene` takes an existing per-gene initialization fit, its spliced
and unspliced moment columns, signed element stride, fit options, optional fit
mask, and caller-owned score scratch. With trimming enabled, scratch has at
least `rows` elements and is disjoint from the input columns. The fit mask
controls trimming and regression; prediction and residual norms use all rows.
A report recomputes two stress fits after sorting the scores.

The default cutoff band is
`0.5 * 2^-11 * max(1, abs(active cutoffs))`. The report flags status or
selection changes, absolute R² change over 0.05, or prediction change over 10%
of the baseline residual L2 norm. The denominator floor is
`32 * epsilon_compute * max(response L2, baseline prediction L2)`. The caller
may change these values. When trimming is disabled, no stress fits run; a
degenerate baseline is unscoreable. This report measures sensitivity to
trimming bounds, not an error bound or an automatic promotion decision.

`validate_sensitivity` explicitly checks options, pointer/stride requirements,
and scratch capacity. It returns `PrecisionCheck` with an error code, location,
and required element count. The numerical `sensitivity_gene` overloads are
trusted entry points: the caller supplies valid pointers and capacity and keeps
all storage alive during the call. Empty inputs require no numerical storage.

## Selecting original columns

`Promotion` is caller-owned state containing the source-gene mask, optional
prepared-to-source gene index array, caller-provided prepared-order status
array, and typed source/replacement callbacks with caller-owned contexts. Keep
the source mask and gene mapping immutable until both passes finish.
`validate_promotion` checks capacities and gene indices. `initialize_promotion`
resets statuses and promotion count. The caller then calls `promote_initial`
for each processed block before fitting that block. Missing originals retain
the baseline values and receive `PromotionStatus::unavailable`; unrequested
positions remain `not_requested`. The count includes duplicate source genes
when they appear at distinct prepared positions.

An `OriginalSource` callback supplies an already acquired pair of columns as
`OriginalColumn`. The two pointers address the first logical observation;
strides are signed **bytes**, including zero for a repeated value. Their full
`(rows - 1) * stride` extent must fit `ptrdiff_t`; invalid extents are rejected
before the replacement writes. Spliced and unspliced columns can independently
use FP32 or FP64. Values must use the same
cell order, normalization, and units as the prepared reference input and remain
valid until the replacement callback returns. Reacquiring a gene for the second
pass must reproduce the same values. The provider handles any sparse
materialization and duplicate summation before supplying a column. A consumer
replacement may allocate explicitly; its memory use and lifetime are its own
contract.

`ReferenceMoments<T>` binds an existing prepared input and the active first- or
second-order moment blocks. Set the corresponding block pointer before each
block and use `replace_reference32` or `replace_reference64` as the
`MomentReplacement` callback. The replacement recomputes only requested
columns using the prepared reference graph and writes directly to the existing
caller-owned block buffers. Custom replacements may instead use entirely
different representations and state.

In stochastic calculations the caller invokes `promote_second` on each second
moment block after the same source choices were made during initialization.
Previously promoted columns are reacquired. If an original disappears between
passes, the function reports `unavailable` at that prepared position rather
than silently mixing sources. Graph weights are unchanged. Blocks cover each
prepared position once per pass; callback/state changes require caller
synchronization. A failure leaves earlier positions complete, while a failing
callback may have partially written its current position; no rollback occurs.

For sensitivity-guided source choice, screen the baseline initialization fit,
choose the mask explicitly, recompute chosen moments, and then fit the promoted
block. The caller can also run screening and source selection as separate
passes with reusable scratch.
