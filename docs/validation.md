# Numerical validation and efficiency evidence

Historical results below describe recorded source snapshots before the current
[portable segmentation](segmentation.md) and FP16 ownership migration.
CUDA/AVX2 sources and support are quarantined and excluded from the active
build. Historical reports and raw artifacts are preserved unchanged; they do
not qualify the current `NumericView`, `NumericEdit`, or `CelleratorBinary16`
path. Current migration checks are recorded separately below.

## FP16 numeric-view ownership migration

The current CPU implementation passed all 12 native debug tests and all 12
ASan/UBSan tests with `detect_leaks=0`; this is not a LeakSanitizer result. Two
installed-module consumers exercised explicit mixed FP16/FP64 inputs and direct
FP16 output. The actual Cellerator root-integrated velocity target built and
ran with CUDA and hardware probing disabled, using a fresh HDF5-only AnnData++
build. The binary16 helper passed exhaustive decoding of all 65,536 encodings
and 390,465 Python/NumPy oracle cases. All seven qualification drivers compiled,
but none was run; no benchmark or GPU execution was performed. See the
[ownership-migration receipt](../tests/results/fp16-ownership.json) for source
identities, commands, and detailed results.

## Historical coordinated device and precision milestone

The [Milestone 3 receipt](../tests/results/milestone-3.json) binds an older
source snapshot, binaries, commands, package checks, and Scheduler receipts.
Its companion records describe the then-tested AnnData++ and velocity builds;
they are historical evidence only and do not describe current AnnData++ storage,
device APIs, or the velocity FP16 numeric-view contract. The receipt and its
companion artifacts remain unchanged.

The full numerical matrix ran 64 cases and 192 driver executions across the
scalar reference, compiled AVX2 implementation, and CUDA. All same-input
backend gates, FP32/FP64 comparison gates, and explicit branch contracts pass.
Storage quantization is reported separately from backend error. Inputs include
independent W/S/U storage choices, masks, signed strides, CSR/CSC duplicates,
mixed index widths, and degenerate fits. Additional contract tests cover reusable
gene batches, block sinks, custom stages, foreign device owners, source
availability, promotion choices, and disabled-screening behavior.
The [per-type error report](../tests/results/precision-errors.json) records
observed deviations alongside the nominal gates. FP16 storage with FP32
computation uses the FP32 backend gates; this does not bound quantization error
relative to original data. Across these 64 cases, observed scalar/AVX2/CUDA
differences were zero ULP for the reported numerical fields in every tested
storage/compute profile. This is fixture evidence, not an identity requirement
or a universal error bound. The report also separates all-gene FP32/FP64
compute sensitivity from the condition/status-filtered qualification gates.

For backend comparisons, each gene's scale is `max(1,max(abs(reference)))`.

| Quantity | FP64 scaled atol / rtol | FP32 scaled atol / rtol |
| --- | --- | --- |
| Moments | 1e-12 / 1e-11 | 2e-6 / 2e-5 |
| Parameters, scales, velocities, residuals | 1e-10 / 1e-9 | 2e-5 / 2e-4 |

R² uses absolute tolerances 1e-10 and 2e-4. Normalized-objective limits are
1e-9 and 5e-4. Parameter gates cover scaled condition numbers up to 1e3;
degenerate or branch-sensitive cases also check statuses and decisions.
Thresholds were retained unchanged during qualification.

One strict source diagnostic remains outside its nominal limit: pancreas
gene 16 with joint intercepts and FP32 computation has native initialization
R² -9.3208255768 versus scVelo -9.3210362876. The absolute difference is
0.0002107108, exceeding 0.0002 by 0.0000107108 (5.36%). The deterministic
and stochastic cases share this initialization. Evaluating native parameters
in FP64 changes R² by only 2.47e-6; most of the source difference comes from
scVelo's FP32 fit parameters. Native parameters are closer to an independent
FP64 OLS solution on the identical stored data: both native coefficients are
three FP32 representable steps from the nearest FP32 OLS coefficients, with
scaled fit condition 24.72. Native R² is 4.78e-6 from that FP64 OLS result;
the source result is 2.155e-4 away. This supports retaining the native
calculation, without proving an absolute lower bound for the format. The user
permits a narrow precision-related exception supported by this evidence. The strict raw
report still exits 1 and records this source miss; it is not relabeled a
tolerance pass. The accepted Milestone 2 offset exception below is preserved
separately.

The historical V100 receipt records that the then-built AnnData++ and
velocity device suites passed CUDA memcheck, racecheck, and synccheck, and that
installed mixed Clang/NVCC consumers ran. Those builds are quarantined and
their receipts do not qualify the active CPU-only component or current FP16
ownership. The recorded checks remain bounded to the V100, toolchain, and
fixtures; they do not establish general biological validity or qualification
on other GPUs.

The [precision measurement record](../benchmarks/results/milestone-3.json)
separates CPU, resident GPU, transfers, packing, allocation, and end-to-end
costs. Ordinary C++ remains the default: paired scalar/AVX2 timings overlap
on the qualified sparse workload, with no demonstrated gain from specialized
vector kernels. AVX2 remains an explicit build option. Performance observations
are bounded to the recorded workloads, with every worker capped at 30 seconds.

Three paired samples on the 4,096-cell, 2,048-gene sparse workload produced
the following median end-to-end times. All 2,048 genes were selected and
stochastically refined. GPU end-to-end includes context setup, packing,
preparation, transfers, and output consumption; separate resident timings are
in the receipt. The CUDA reference demonstrates no end-to-end speedup here.

| Storage / computation | Method | Scalar CPU | CUDA V100 |
| --- | --- | ---: | ---: |
| S/U FP16, W FP32 / FP32 | Steady | 0.528 s | 0.715 s |
| S/U FP16, W FP32 / FP32 | Stochastic | 1.851 s | 1.953 s |
| FP32 / FP32 | Steady | 0.515 s | 0.712 s |
| FP32 / FP32 | Stochastic | 1.843 s | 1.992 s |
| FP64 / FP64 | Steady | 0.505 s | 0.813 s |
| FP64 / FP64 | Stochastic | 1.819 s | 2.455 s |

With 64-gene blocks, CUDA FP32 numerical workspace was 458,752 bytes for both
256×256 and 256×8,192 inputs. It grew to 7,340,032 bytes at 4,096 cells and
14,680,064 bytes at 8,192 cells. Prepared structure, inputs, diagnostics, and
outputs are accounted separately. Half-expression canonical storage used
2,177,216 input bytes versus 2,511,104 for FP32; no full FP32 shadow was needed.

The [module-build receipt](../benchmarks/results/module-build-milestone-3.json)
records five alternating warm consumer pairs: baseline median 0.415379 s,
current 0.415535 s, ratio 1.000375 against the 1.25 limit. Private CPU, AVX2,
CUDA, HDF5, and Zarr edits preserved all measured public BMIs and consumer
objects by hash and timestamp. Cold configure/build totals increased from
12.865 s for the frozen baseline to 21.739 s for current scalar, 23.241 s for
AVX2, and 36.025 s for CUDA. Additional compiled implementations increase the
cold build while leaving ordinary consumer iteration effectively unchanged.

## Milestone 2 acceptance contract

Stochastic velocity uses the pinned scVelo production `compute_stochastic`,
`second_order_moments`, and `leastsq_generalized` definitions as ordinary-case
comparison targets. It adds consistent fit-row masks and explicit deterministic
fallbacks. These deliberate differences are independently checked, rather than
treated as failures to reproduce undefined upstream results.

Raw moments must agree at `atol=1e-6, rtol=1e-5`; well-conditioned slopes and
residual scales at `atol=1e-5, rtol=1e-4`; intercepts and residuals at
gene-scale-adjusted `atol=1e-5, rtol=1e-4`; normalized weighted objectives within
`1e-4`. Selection agrees away from declared threshold margins. Degenerate cases
are checked against analytic behavior, finite diagnostics, and fallback reasons.
Numerical similarity is required; identity is not.

The canonical stochastic measurement must actually refine genes using matched
inputs, options and output requests. Three sequential fresh-process pairs, each
bounded to 30 seconds and 2 GiB, must establish at least 10x lower preparation-
inclusive incremental peak memory and no median end-to-end runtime regression.
Record prepared storage, accounted scratch, output retention, allocation counts,
and process peaks separately. Fixed-block shape controls must establish scratch
independence from total gene count and linear scaling in cell count.

## Milestone 2 implementation evidence

- [Native/package checks](../tests/results/milestone-2-native.json)
- [Stochastic numerical comparison](../tests/results/milestone-2.json)
- [Independent solver review](../tests/results/milestone-2-solver-review.json)
- [Deterministic regressions](../tests/results/milestone-2-deterministic-regression.json)
- [Stochastic memory/runtime](../benchmarks/results/milestone-2.json)
- [Module builds](../benchmarks/results/module-build-milestone-2.json)

Native, Release, shared-library, and ASan/UBSan configurations each pass seven
CTest groups, including two installed consumer executables. Durable tests
exercise independently replaced numerical stages, opaque state, padded
multiblock values, unselected-gene reference moments, sparse duplicates,
caller-output overlap rejection, value mutation, bounds, and fallback statuses.
The independent solver comparison passed 720 seeded mask/intercept/trimming/
bound combinations. The four Milestone 1 numerical comparisons and degenerate
checks pass against the current binary; their historical report is preserved.

All 32 stochastic estimator cases pass, covering both real fixture slices,
all four intercept combinations, trimming on/off, and all-gene versus fit-row/
HVG masks. Selection masks agree exactly. The independent production-class
cross-check and independent SciPy FP64 initialization check also pass all
32 cases. The latter checks the selected genes using independently calculated
FP64 moments and centered regressions.

**Qualification exception:** the strict all-source-diagnostics `pass_all` flag
is false. Four dentate-gyrus cases with an initial intercept and no trimming
exceed the original FP32 source-initialization intercept tolerance. For gene
142 the all-row source/native offsets are approximately 0.074706376 and
0.074686333: error `2.004e-5` versus tolerance `1.813e-5`. The fit-subset error
is `1.579e-5` versus `1.111e-5`. Native agrees with the independent FP64
calculation to about `2e-16`; design condition numbers are approximately
14.2–14.5. This is a documented precision-related exception to the planned
source-diagnostic gate, not an ill-conditioned-case exclusion or a tolerance
pass. The more accurate native calculation is retained. Final stochastic
parameters, residual scales, velocities, masks, and normalized objectives pass
their estimator gates.
The user subsequently accepted this precision-related difference. The strict
source-diagnostic result and its nonzero command exit remain recorded as
observed; this acceptance does not change the measured comparisons or establish
native Project Control outcome acceptance.

Gene-scaled second-residual absolute tolerances use the maximum magnitude of
the equation's second-order terms per gene, with a floor of one, rather than
the first-order velocity scale. The coefficient remains `1e-5` and relative
tolerance `1e-4`. This makes the declared abundance-squared scale explicit for
a residual that subtracts large terms. Raw errors and the earlier narrower
first-order-scale comparison are retained in the numerical artifacts.

Three paired fresh-process runs on 4,096 cells, 2,048 genes, about 32 neighbors
per cell, and 1% sparse FP32 expression measured median incremental peaks of
690,081,792 bytes for scVelo and 18,214,912 bytes for native streaming:
**37.89x lower**, including preparation. Both implementations computed velocity
and second residuals for all 2,048 refined genes. Explicit benchmark selection
thresholds were `min_r2=min_ratio=-100`; no automatic fallback supplied the fit
population. Selected masks and refinement counts matched exactly, and all five
parameter/output checksum gates passed.

Median cold load/prepare/compute/sink times were 2.278 s and 1.966 s. Sample
ranges overlap (2.115–2.322 s and 1.773–2.181 s), so this establishes the planned
median no-regression gate, not a robust general runtime speedup. Each worker
used one pinned CPU/thread, a 30-second limit, and a 2-GiB address-space cap.
Input normalization, neighbor discovery, interpreter startup, and imports are
outside the timed scientific computation.

Canonical numerical scratch was 10,588,160 bytes and prepared traversal storage
7,127,496 bytes. Returned per-gene diagnostics occupied 153,600 bytes. Execution
made 61 counted C++ allocation calls requesting 156,720 bytes, including results
and descriptors. Source NumPy/SciPy allocation calls were not intercepted;
source field sizes and measured process peaks remain separate evidence.

Single-run shape controls used identical 661,760-byte scratch at 256 cells and
256 versus 8,192 genes. At 64-gene blocks and 256 total genes, doubling cells
from 4,096 to 8,192 doubled scratch from 10,588,160 to 21,176,320 bytes. Prepared
indices and owning diagnostics are separately accounted and may grow with
input geometry. These controls establish workspace scaling, not runtime gains.

The isolated build probe measured a fresh build at 13.05 s, consumer
recompilation at 0.416 s, and private-implementation rebuilding at 3.52 s.
All eight public BMIs and both example consumer objects were unchanged by the
private edit. Operating-system caches were uncontrolled; the report establishes
recompilation isolation and records build cost without a cross-library speed
claim.

## Milestone 1 evidence

Milestone 1 has native, installed-consumer, sanitizer, numerical-reference, and
bounded memory/runtime qualification. Machine-readable records retain source,
binary, fixture, and workload hashes:

- [Numerical comparison](../tests/results/milestone-1.json)
- [Memory and runtime](../benchmarks/results/milestone-1.json)
- [Module build behavior](../benchmarks/results/module-build.json)

Both supplied 100-cell, 200-gene pancreas and dentate-gyrus fixtures pass the
declared numerical tolerances for all-gene and fit-subset/HVG cases, with zero
mask disagreements. Zero/constant inputs have finite neutral diagnostics and
invalid status. Source AST definitions are pinned independently of native code.

For the canonical 4,096-cell, 2,048-gene, ~32-neighbor, 1%-dense FP32 workload,
three sequential fresh-process pairs on one pinned CPU measured preparation-
inclusive incremental peak memory of 301,973,504 bytes for scVelo and 12,693,504
bytes for native streaming: **23.79x lower**. Median cold load/prepare/compute/sink
times were 0.49043 s and 0.48206 s respectively. The small timing difference and
three-sample ranges do not establish a robust runtime speedup. These timings
exclude interpreter startup and imports. Total process peaks are reported
separately; output retention and other layouts have their own single-run controls.

At a 64-gene block size, numerical scratch was 5,279,744 bytes for both
4,096x256 and 4,096x2,048 inputs; increasing cells to 8,192 at 256 genes used
10,559,488 bytes, exactly twice as much. These one-shot shape controls establish
workspace scaling, not runtime performance. Prepared indices and total process
memory remain separately counted.

The synthetic source fallback selected 410 genes, while the explicit default
filters selected none. The reference worker reapplied the explicit filter policy
for the declared comparison. Timed runs check a numerical velocity checksum;
elementwise and gene-mask agreement are covered by the independent fixture tests.

Native, Release, ASan/UBSan, and shared-library configurations each pass all four
CTest groups, including true component substitution, opaque custom state,
explicit reference independence, and installed consumers. Sanitizers cover
velocity and test interfaces; the separately built anndata binary is Release.
Centered FP64 intercept tests cover large baselines with small variation.
Extreme finite values around 1e200 can overflow raw through-origin sums and
are outside the qualified numeric range.

An isolated module probe measured fresh build 11.75 s, consumer recompilation
0.416 s, and private-implementation rebuild 2.47 s. All eight public dependency/
velocity BMIs and the consumer object were unchanged after the private edit.
OS caches were uncontrolled; these are recorded build conditions, not a
cross-library compilation-speed claim.

The [source audit](scvelo-foundation.md) remains historical upstream evidence;
its older acceleration results are not velocity.cppm results.

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
  AnnData++ and pinned upstream readers when that adapter is implemented.

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

GPU evidence must distinguish resident execution from transfers, preparation,
launch, and end-to-end costs. Use the authorized Scheduler and isolate
measurements. The coordinated milestone above records the bounded CUDA checks;
historical Milestone 1/2 reports contain CPU evidence only.

## Native and development checks

The native suite covers numerical/ownership contracts, ASan/UBSan, true custom
stages, and independent installed C++20 module consumers. Build probes run in
isolated source copies so measurements never edit the live implementation.
