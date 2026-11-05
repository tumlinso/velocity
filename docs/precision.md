# Precision and selected originals

`import velocity.precision;` provides an optional advisory sensitivity report
and stage compositions for explicit higher-precision source genes. Ordinary
`reference::steady_state` and `reference::stochastic` never inspect an original
source or perform screening; FP16 storage with FP32 computation remains a
fully independent path. Set `compute_precision` explicitly where the execution
precision matters. Storage choices for graph weights and expression are separate.
The default `ComputePrecision::automatic` resolves once: it selects FP32 when
at least one of W/S/U is FP16 and none is FP64, and otherwise selects FP64.
Explicit FP32 or FP64 overrides this. Original-data promotion does not change
the chosen computation precision. FP16 storage bytes and persistent encodings
are outside this API. `NumericView` and `NumericEdit` expose explicit raw-bit
views; Cellerator's small CPU-only, header-only `Cellerator::binary16` helper
provides `cellerator::numeric::load_f16()` and `round_f16()`. AnnData++ does
not provide this storage or conversion contract. The ISA-specific conversion
and CUDA implementations are
quarantined (`Cellerator/quarantine/velocity-accelerators/` in the parent checkout).

`sensitivity_gene` accepts one existing, local initialization fit and the
corresponding spliced/unspliced moment columns, plus a caller-owned rows-element
score scratch. It uses the fit-row mask during trimming/regression and all rows
for prediction/residual norms, as the existing R² does. The report recomputes
only two stress fits after sorting scores once. Defaults use a cutoff band of
`0.5 * 2^-11 * max(1, abs(active cutoffs))` and flag a changed status/selection,
absolute R² change over 0.05, or prediction change over 10% of baseline
residual L2. The denominator floor is `32 * epsilon_compute * max(response L2,
baseline_prediction L2)`. Thresholds are configurable. With trimming disabled,
no stress fits run; a degenerate baseline is explicitly unscoreable. A report
measures susceptibility to changed trimming bounds; it is not an error bound
or an automatic promotion decision.

For bounded CPU screening, `SensitivityWorkspace32(input, workspace, stages,
budget_bytes, staging_bytes)` allocates the score buffer only after verifying
it, the bound stage workspace, and declared source staging all fit the budget.
Use `SensitivityWorkspace` with FP64 computation. A manually supplied score
buffer must be accounted for by the caller. Report arrays are caller-owned and
counted separately as O(genes) diagnostics.

`OriginalArraySource` retains two dense host `NumericView`s. It returns
zero-copy single-column numeric views only when a requested source gene is
acquired. Borrowed raw
arrays still require their underlying owner. The source object must outlive the
stages using `source()`. `OriginalSource` instead accepts a callback for a
backed/custom provider: return one dense N×1 FP32/FP64 S/U `NumericView` pair
or `false` if unavailable, and declare the provider's total staging bytes. A pair with either
column in FP16 is also unavailable: widening already-rounded numbers cannot
restore the original values. Sum sparse duplicates when materializing a column.
Values and owners must remain valid when used and remain consistent between
initialization and stochastic refinement. Original columns must use the same
cell order, normalization and units as the FP16 source.

Construct `SelectiveStages32` or `SelectiveStochasticStages32` around the
corresponding reference stage, workspace and input, and provide a host byte mask
indexed by source gene. The mask stays immutable for the run. Promotion happens
only for requested genes with available original pairs; unavailable genes stay
FP16 and get an `unavailable` status. `promotion_status()` is in prepared-gene
order and is null when no mask or genes were supplied. `promotion_count()`
counts prepared-gene occurrences, including repeated selections. Initialization
is complete before global low-signal and
refit-all selection; the stochastic second pass reuses that same choice. If a
previously available original disappears between passes, execution fails before
mixing source precisions. Graph weights are not promoted.

The CPU convenience wrapper acquires one requested gene at a time to bound
staging. For caller-controlled multigene batching, the CPU reference stages also
expose `replace_moments` on a contiguous range, reusing the same prepared graph
and workspace. To screen half-derived moments and apply a chosen mask within a
custom first pass, obtain baseline moments and fit, report sensitivity, replace
explicitly chosen columns, recompute that block's initialization, and only then
accept it. Alternatively run the report and promotion as separate explicit runs.

The former CUDA source-promotion and sensitivity implementation is preserved
with its original documentation in the accelerator quarantine. It is not part
of the active portable component or its package exports.

See the [selective precision example](../examples/selective_precision.cpp) for
a complete CPU stochastic run with FP16 storage, FP32 computation and a mask
that promotes one gene from retained original arrays.
