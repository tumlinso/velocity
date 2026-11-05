# scVelo mathematical foundation and efficiency evidence

This is the source audit recorded on 2026-11-03 before implementation. It retains
the inspected upstream provenance; current implementation and qualification are
described in [the API](api.md) and [validation](validation.md).
velocity.cppm is intended to be an independently useful, substantially more efficient
native C++20 RNA velocity implementation and an explicit reference for substitutions.
The reference itself should use sound algorithms, compact storage, reuse, and bounded
workspace. Reproducing Python allocation patterns is not a compatibility requirement.
Numerical similarity to scVelo is required; bitwise identity and exact result equality
are explicitly not required. The observations below identify scientific/numerical
choices to assess, not a requirement to reproduce every upstream implementation detail.
No velocity.cppm efficiency improvement or biological validity has been measured here.

## Source identity and existing acceleration

The inspected scVelo checkout, `/home/tumlinson/accelerated-ports/scVelo`, was clean at
`4706c4115963b7cae68883d110d351a29904bc24`. Its native work was added in `18aa190`
(standalone moments probe), `eef1b09` (resident direct/Torch consumers), and `868927a`
(focused qualification record). There is no Cellerator dispatch in the production
`scvelo/` package: the accelerated computations live in `probes/native_moments/`.
They compare finalized upstream inputs and feed native means to unchanged scVelo
steady-state fitting; they do not replace the complete stochastic or dynamical route.

The inspected Cellerator HEAD was `82ec3217c9bad1779d47cddda94ed223832d3f72`.
That checkout contains substantial unrelated uncommitted work. The decisive files
listed below were individually clean against that HEAD when inspected; do not use
the HEAD alone to identify the entire working tree or a previously built package.

| Source file | SHA-256 of inspected bytes |
| --- | --- |
| scVelo `scvelo/preprocessing/moments.py` | `9189d13d1257d86f49b6bb8d1816d7bbdcd647cb330cca922d738f36bf98714e` |
| scVelo `probes/native_moments/resident_moments.py` | `f132ebc97b9ccd455a2dd0a1116e36a17d0ceed60714458eb5fb48ce25a34bc9` |
| Cellerator `include/Cellerator/compute/operation/prepared_relation.hh` | `e98c3a7aaa6c1ff58ad8dea34d96e1ef4cce6442cc44bd62c89b593f4d434586` |
| Cellerator `src/compute/operation/prepared_relation.cu` | `b3e560a890f35aebfb9e6c976c2cf93e6f5c243f304ad9ef678c4494197b9273` |
| Cellerator `include/Cellerator/compute/operation/device_elementwise.hh` | `c6a4cb1950f114b6f893cfc1a78c713b835f64489be3a3f116b6950ed75be3f2` |

The retained [probe record](/home/tumlinson/accelerated-ports/scVelo/docs/native-moments-probe.md:52)
reports fixed numerical gates on small pancreas, dentate-gyrus, CellRank, and analytic
fixtures, including deterministic fit coefficients, residuals, and exact gene masks.
Its original performance results concern synthetic resident FP32 compositions only.
The later direct/Torch/C++ common-input qualification makes no speedup claim and
defers its large synthetic performance campaign and Compute Sanitizer follow-on.
These are historical source-bound results, not experiments rerun for this document,
end-to-end scVelo speedups, or broad scientific validation. The resident consumer
checks scoped source, extension, executable, and input hashes before comparison.

## Mathematical objects and distinctions

Let `S,U` contain spliced/unspliced abundances with cells as rows and genes as columns.
Their normalization, selected genes, cell order, and units are scientific inputs.
Let `W` have destination cells as rows and source cells as columns. In scVelo's
[graph preparation](/home/tumlinson/accelerated-ports/scVelo/scvelo/preprocessing/neighbors.py:462),
positive selected graph entries become binary support, the diagonal is set to one,
and rows are normalized; optional recursive-neighbor behavior changes that recipe.
Passing the original weighted neighbor graph directly to multiplication is therefore
not generally equivalent. A generic rectangular relation is useful computationally;
it is not automatically the same scientific neighborhood model.

The [moments implementation](/home/tumlinson/accelerated-ports/scVelo/scvelo/preprocessing/moments.py:84)
contains the following distinct operations, where products inside `W(...)` are
elementwise and all fields retain explicit cell/gene order:

| Quantity | Formula |
| --- | --- |
| Means | `Ms = W S`, `Mu = W U` |
| Raw second/cross moments | `Mss = W(S*S)`, `Mus = W(S*U)`, `Muu = W(U*U)` |
| Centered identities | `Mss-Ms*Ms`, `Mus-Ms*Mu`, `Muu-Mu*Mu` |
| `second_order_moments(adjusted=True)` | `2*Mss-Ms`, `2*Mus-Mu` |
| Stochastic fitting inputs | `var_ss = 2*Mss-Ms`, `cov_us = 2*Mus+Mu` |

The last row follows [stochastic fitting](/home/tumlinson/accelerated-ports/scVelo/scvelo/tools/velocity.py:119),
not the helper's adjusted cross moment. Neither adjusted quantity should be renamed
ordinary variance/covariance. Centered identities have their statistical meaning
under the stated normalized-weight assumptions and can lose small variances through
cancellation. Unrequested `Muu` or centered fields need not be computed merely because
the probe exports them. Clipping a negative computed variance is a numerical policy,
not an innocuous layout optimization.

For [deterministic steady-state fitting](/home/tumlinson/accelerated-ports/scVelo/scvelo/tools/velocity.py:62),
each gene fits `Mu ≈ a + r*Ms` and reports `v = Mu-r*Ms-a` (or `a=0`). With
unweighted through-origin fitting, `r = sum(Ms*Mu)/sum(Ms*Ms)`; the actual route
also has group masks, extreme-percentile selection, optional nonnegative intercept,
ratio constraints, R-squared calculation, and gene-selection/fallback rules.
The [linear regression source](/home/tumlinson/accelerated-ports/scVelo/scvelo/core/_linear_models.py:59)
normalizes each input by its gene-wise maximum before percentile selection; it is
not simply independent trimming of S and U. Here `r` is an effective rate ratio
under the steady-state convention, not an independently identified degradation rate.
Stochastic fitting combines first- and corrected second-moment residuals using
residual-scale weighting and [generalized least squares](/home/tumlinson/accelerated-ports/scVelo/scvelo/tools/optimization.py:123)
or its separately selected likelihood route; replacing it with ordinary regression
would change the method.

The kinetic model has `du/dt = alpha-beta*u`, `ds/dt = beta*u-gamma*s`.
For constant rates and `beta != gamma`, the [implemented solution](/home/tumlinson/accelerated-ports/scVelo/scvelo/core/_models.py:100)
is `u(t)=u0*exp(-beta*t)+(alpha/beta)*(1-exp(-beta*t))` and
`s(t)=s0*exp(-gamma*t)+(alpha/gamma)*(1-exp(-gamma*t))`
`+((alpha-beta*u0)/(gamma-beta))*(exp(-gamma*t)-exp(-beta*t))`.
Induction/repression use different transcription regimes joined at a switching time;
fitting includes abundance scaling, latent time/branch assignment, initialization,
optional connected-state regularization and steady states. The source's
[DynamicsRecovery](/home/tumlinson/accelerated-ports/scVelo/scvelo/tools/_em_model_core.py:124)
coordinates several parameter fits using Nelder-Mead. Its
[loss and likelihood](/home/tumlinson/accelerated-ports/scVelo/scvelo/tools/_em_model_utils.py:1076)
are separate computations: summed/mean standardized distance with optional
regularization is not interchangeable with every reported likelihood mode.
Time/rate rescaling and latent assignments require declared conventions; a fitted
RNA derivative alone does not identify a universal biological time or causal model.

Directed velocity-graph scores are another operation: centered cosine between a
cell's velocity and expression displacements to selected neighbors, with optional
signed-square-root transforms, recursion, and uncertainty calculations. See
[construction](/home/tumlinson/accelerated-ports/scVelo/scvelo/tools/velocity_graph.py:215)
and [correlation](/home/tumlinson/accelerated-ports/scVelo/scvelo/tools/utils.py:55).
These scores, the moments graph W, and downstream transition probabilities have
different meanings and should not share an implicit normalization contract.

## Concrete memory opportunities

For `N` cells, `G` genes, `E` graph edges, scalar width `b` bytes and a gene block
width `B`, one dense field costs `b*N*G`. Blocking can bound *workspace* by `N*B`;
retaining `r` complete result fields still costs at least `r*b*N*G`. Sparse inputs
can produce dense smoothed outputs. Neither C++ nor sparse graph storage removes
that output cost; selective outputs, a sink, or deliberate recomputation changes
what must remain resident. The following are source-derived opportunities, not
measurements or an agreed implementation plan.

| Observed allocation/computation | Opportunity with its semantic constraint |
| --- | --- |
| `moments.py` converts layers to CSR, then separately multiplies, casts to FP32 and densifies each mean/second moment. | Prepare graph once; consume sparse input without whole-input densification; compute gene blocks into explicit outputs. Preserve implicit zeros, graph order and conversion/rounding boundaries. |
| `_linear_models.py:75` stacks both whole inputs, normalized values and trimming masks; `optimization.py:157` stacks two N-row systems. | Select/fit per gene or bounded gene block, accumulate small sufficient statistics or solve directly. Percentile selection and rank-deficient solver behavior affect comparisons; improved estimators or solvers need declared numerical-similarity checks. |
| `resident_moments.py:303` allocates 5 source and 13 destination fields before every repeated composition. For square graphs this is `18*4*N*G` bytes, plus weights and prepared storage. Means-only download still computes all ten outputs. | Share/reuse scratch by lifetime; request only required fields; fuse graph traversal with products/reductions where the arithmetic policy permits. Preserve an inspectable unfused mathematical route without requiring it to allocate all fields. |
| `_em_model_utils.py:195` projects each cell onto 200–500 curve points using broadcast cell×point×2 differences and squared-distance temporaries. | Evaluate the grid with bounded cell/point tiles or a running minimum. Account for candidate points and first-minimum tie handling when comparing results; a revised grid or time estimator is also possible with explicit numerical assessment. |
| `_em_model_core.py:544` and `_em_model.py:43` allocate T, Tau and Tau_ over all cells/genes; default NumPy allocation uses FP64, totaling `24*N*G` bytes before other state. Recovery retains fitted per-gene objects before collecting results. | Make selected-gene extent, latent outputs, histories and result sinks explicit; release completed work. Preserve requested outputs and any later alignment/shared-time dependency. |
| `velocity_graph.py:71` densifies sparse layers before gene selection, copies X/V and V_raw, and accumulates Python edge lists before CSR construction. | Select before densifying; reuse expression/velocity storage where lifetimes allow; fill bounded edge batches into final sparse output. Preserve recursion, duplicate treatment, zero/nonfinite rules and transforms. |

Buffer reuse, batching and layout changes can preserve the defined method while
changing floating-point evaluation order. Fusing reductions, FMA/reassociation,
precision changes, centered algorithms and revised special-case formulas therefore
need explicit numerical assessment. They can satisfy the similarity target without
reproducing the original intermediate values or final results exactly.

## Existing Cellerator mechanisms and integration limits

Cellerator already supplies relevant general execution mechanisms. Its
[moments design note](/home/tumlinson/Cellerator/docs/design/native-neighborhood-moments.md:3)
describes a reference composition, not a dedicated scientific moments API or a full
velocity toolkit. Graph/scientific policy, regression, kinetic fitting, model outputs,
and complete standalone velocity behavior remain outside that composition.

| Mechanism | Reusable contract and material boundary |
| --- | --- |
| [Host relation](/home/tumlinson/Cellerator/include/Cellerator/compute/operation/native_numeric/host_relation.hh:22) | Synchronous forward overwrite, homogeneous FP32/FP64, copied topology, caller-owned values/outputs, no run-time allocation; stable logical edge order and duplicate contributions. This is a generic host mechanism, not velocity.cppm's implemented reference. |
| [Prepared CUDA relation](/home/tumlinson/Cellerator/include/Cellerator/compute/operation/prepared_relation.hh:9) | Owned preparation/projections, explicit axes, structure epoch, logical edge order, value generation and stream/device. Host topology is consumed during preparation; borrowed values must live through completion. Cached topology is reusable across value updates. Preparation/capability is not inferred from a descriptor's validity. |
| [Current CUDA provider](/home/tumlinson/Cellerator/src/compute/operation/prepared_relation.cu:25) | FP32/FP64 relation/field pairs promote to FP64 if either operand is FP64. Admission requires FMA/reassociation permission, propagating nonfinites and nonaliasing input/output. Prepared masked topology rejects duplicate endpoints within a row; shared CSR traversal and the host route accept duplicates. Canonicalization changes edge/reduction order and must be explicit. |
| [Device elementwise operations](/home/tumlinson/Cellerator/include/Cellerator/compute/operation/device_elementwise.hh:22) | Homogeneous FP32/FP64 multiply, square and affine operations with caller storage; allocation-kind/capacity and overlap restrictions apply. This differs from the older FP32-only resident-vector surface used by the moments probe. Do not assume identical alias or binding contracts. |
| [Direct CUDA bindings](/home/tumlinson/Cellerator/python/cellerator/cuda.py:1) and [Torch adapter](/home/tumlinson/Cellerator/python/cellerator/torch/cuda.py:61) | Explicit streams/buffers/PreparedCsr; Torch supplies storage and transfers and calls native arithmetic. The inspected Torch adapter requires FP32, so newer C++ FP64 support does not imply Python-probe FP64 coverage. |

The later [FP64 result record](/home/tumlinson/Cellerator/docs/results/fp64-qualification.md:24)
separates prepared cases from shared-CSR duplicate cases. Its cancellation-heavy
variance witness loses the variance even in FP64. This supports a centered numerical
oracle and explicit precision policy, not a claim that FP64 universally fixes moments.
The older FP32 scVelo probe's qualification must not be broadened to these newer
mechanisms merely because they are present in the current source.

## Reference and substitution contracts

The standalone reference must be directly callable with substitutions bypassed,
including nested primitives: an installed provider must not silently intercept its
moment application, objective evaluation or solver. Cellerator is an optional
consumer/provider, not a prerequisite for reference semantics. Intermediate results
must remain independently consumable so its custom route can replace a graph apply,
moment computation, objective, time assignment, update rule, solver or batching
strategy without forking the complete scientific method. This calls for explicit
inputs, outputs, workspace, precision, diagnostics and lifetime contracts at those
boundaries, rather than only a final-pipeline backend switch.

Future kinetic milestones will need separate compatibility targets. scVelo's public recovery includes `recover_dynamics` and
[ExpectationMaximizationModel](/home/tumlinson/accelerated-ports/scVelo/scvelo/tools/_em_model.py:82);
agreement with one is not automatic agreement with the other. The kinetic source
explicitly leaves `beta=0`, `gamma=0` and `beta=gamma` unresolved. Stable limits
and `expm1`-style evaluation are native candidates, but changes near singular
parameters require their own comparison and acceptance contract. Latent-time,
branch, history, objective, solver, precision, and nonfinite-output choices must
also be declared for those later operations.

## Milestone 2 stochastic contract

Milestone 2 makes the steady-state stochastic equations an explicit native
operation. With first moments `Ms = W S` and `Mu = W U`, and raw second
moments `Mss = W(S*S)` and `Mus = W(S*U)`, define
`A = 2*Mss - Ms` and `B = 2*Mus + Mu`. `A` and `B` retain the scVelo
stochastic convention; they are not ordinary centered variance and covariance.
`Muu` is computed only when requested.

For each gene, the reference first performs the deterministic fit and measures
`sigma1` from its first-order residuals. It then performs an untrimmed
second-order regression of `B` on `A` to estimate `sigma2` from its residuals.
`fit_second_intercept` controls this regression's intercept, and
`initialization.positive_intercept` controls whether a negative initial
intercept falls back to zero. The final
joint fit minimizes `sum((r1/sigma1)^2 + (r2/sigma2)^2)` over included fit
rows, where `r1 = Mu - gamma*Ms - offset` and
`r2 = B - gamma*A - offset2`. Percentile exclusions zero the corresponding
first-order `Ms` and `Mu`; they do not remove the first-order intercept rows or
alter second-order observations.

The optional *output* second-order residual is a distinct quantity:
`(B - 2*Ms*Mu) - gamma*(A - 2*Ms*Ms) + 2*offset*Ms - offset2`. It is not
the residual used in the GLS objective. The initial second-order regression
used to obtain `sigma2` is untrimmed. Final optional intercepts are
unconstrained. A supplied slope bound is honored; when it activates, enabled
intercepts are refit at that fixed slope.

The default refinement population is the deterministic selected-gene mask. If
none qualify, keep deterministic results; matching scVelo's refit-all fallback
is an explicit option, and explicit gene filters always remain enforced. The
fit-row mask applies consistently to deterministic initialization, percentile
selection, both residual scales, and joint fitting. Predictions still cover
all cells. These selection and mask rules deliberately address scVelo's
all-row stochastic fit and none-selected refit-all behavior.

A zero or nonfinite residual scale, insufficient rows, or a rank-deficient
joint system does not produce an undefined stochastic parameter. If the
initial deterministic result is usable, retain it and report why refinement
fell back. If it is not usable, preserve its invalid status. This behavior is
qualified against the declared method; scVelo's undefined degenerate fits are
not the oracle for these cases.

For sparse expression data, elementwise products operate on logical matrix
values. Duplicate coordinates are summed before squaring or forming a
spliced-unspliced product, so `(a+b)^2` is represented rather than `a^2+b^2`.
This coalescing occurs during source-column acquisition for each numerical run,
so updated value buffers are respected. Graph duplicates retain their additive
behavior. Products and graph application are blockwise and do not require
complete `N x G` second-moment matrices or a doubled GLS observation matrix.
