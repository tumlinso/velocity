# Precision and selected originals

`import velocitypp.precision;` provides an optional advisory sensitivity
report and stage compositions that acquire explicitly selected original gene
columns. The standalone package stores expression and graph arrays as FP32 or
FP64 AnnData++ arrays. `anndatapp::ArrayView`, `ArrayEdit`, and `DType` are used
directly.

The default `ComputePrecision::automatic` resolves to FP64 for supported
FP32/FP64 inputs. Set `compute_precision` explicitly to use FP32. Storage and
computation precision remain separate choices; selecting a compute precision
does not change source array dtypes. Unless `RunOptions::output_dtype` is
explicitly set, FP32 computation produces FP32 output. FP64 computation uses
the widest spliced/unspliced input dtype, with FP32 as the minimum, so output
is FP64 when either expression input is FP64 and FP32 otherwise.

Ordinary `reference::steady_state` and `reference::stochastic` do not inspect
original sources or run screening. The sensitivity report is advisory and does
not request source acquisition or promotion.

## Sensitivity to trimming thresholds

`sensitivity_gene` accepts an existing initialization fit, the corresponding
spliced/unspliced moment columns, and caller-owned rows-element score scratch.
It uses the fit-row mask during trimming and regression and all rows for
prediction and residual norms, as the existing R² does. The report recomputes
only two stress fits after sorting scores once.

Defaults use a cutoff band of
`0.5 * 2^-11 * max(1, abs(active cutoffs))` and flag a changed status or
selection, an absolute R² change over 0.05, or a prediction change over 10% of
baseline residual L2. The denominator floor is
`32 * epsilon_compute * max(response L2, baseline_prediction L2)`. Thresholds
are configurable. With trimming disabled, no stress fits run; a degenerate
baseline is explicitly unscoreable. This measures susceptibility to changed
trimming bounds; it is not an error bound or an automatic promotion decision.

For bounded CPU screening, `SensitivityWorkspace32` allocates score storage
only after verifying that the score buffer, bound stage workspace, and declared
source staging fit the numeric budget. Use `SensitivityWorkspace` with FP64
computation. A manually supplied score buffer must be accounted for by the
caller. Report arrays are caller-owned and counted separately as O(genes)
diagnostics.

## Selecting original columns

`OriginalArraySource` retains two dense host `anndatapp::ArrayView`s and returns
zero-copy single-column views when a requested source gene is acquired.
Borrowed storage still requires its owner. Keep the source object alive while
its views are in use.

`OriginalSource` accepts a callback for a backed or custom provider. It returns
a dense N×1 FP32 or FP64 spliced/unspliced `ArrayView` pair, or reports that the
pair is unavailable, and declares the provider's total staging bytes. Sum
sparse duplicates when materializing a column. Values and owners must remain
valid while used and consistent between initialization and stochastic
refinement. Original columns must use the same cell order, normalization, and
units as the prepared input.

Construct `SelectiveStages32` or `SelectiveStochasticStages32` around the
corresponding reference stage, workspace, and input. Supply a host byte mask
indexed by source gene and keep it immutable for the run. Only requested genes
with available original pairs are selected; missing originals retain the
baseline input and receive an `unavailable` status.
`promotion_status()` is in prepared-gene order and is null when no mask or genes
were supplied. `promotion_count()` counts prepared-gene occurrences, including
repeated selections.

Initialization is complete before global low-signal and refit-all selection;
the stochastic second pass reuses the same source choice. If a previously
available original disappears between passes, execution fails before mixing
source data. Graph weights are not promoted.

The CPU convenience wrapper acquires one requested gene at a time to bound
staging. For caller-controlled multigene batching, reference stages also expose
`replace_moments` on a contiguous range while reusing the prepared graph and
workspace. To apply a chosen source mask within a custom first pass, obtain
baseline moments and fit, report sensitivity, replace explicitly chosen
columns, recompute that block's initialization, and only then accept it.
Alternatively run sensitivity reporting and source selection as separate
explicit runs.

See the [selective precision example](../examples/selective_precision.cpp) for
a CPU stochastic run with an explicit source-gene mask.
