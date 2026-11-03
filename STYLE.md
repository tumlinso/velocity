# Coding style

Write compact numerical and systems code in the tradition of classic C and
Fortran scientific libraries. Keep the calculation and its relationship to
memory visible. This transparency is valuable independently of runtime speed.

- **Keep calculations locally understandable.** Use complete mathematical
  expressions, ordinary loops, concise notation, and direct control flow.
  Keep relevant invariants near the calculation. Long names and forwarding
  helpers do not substitute for a clear algorithm.
- **Use direct representations.** Pointers, arrays, references, plain structs,
  dimensions, strides, and sparse buffers should reveal the working data.
  Make units, mutability, ownership, aliasing, and lifetimes explicit.
- **Make costs visible.** Allocation, initialization, copying, conversion,
  packing, transfer, materialization, and synchronization should be deliberate.
  Reuse caller-controlled outputs and workspace when the operation benefits.
  Retain only intermediates that serve the calculation or a requested result.
- **Keep machinery minimal.** Standard-library facilities, dependencies,
  templates, RAII, and helpers need a concrete benefit. Use small templates
  and callable structs when they express useful composition or specialization.
  Avoid wrapper hierarchies, callback layers, registration systems, and generic
  scaffolding introduced merely for extensibility.
- **Separate acquisition from computation.** Resolve types, layouts, ownership,
  and preconditions at suitable boundaries. Provide explicit borrowed or trusted
  access where justified. Acquired numerical loops should not pay repeated
  library dispatch, checks, allocation, or synchronization.
- **Expose scientific components.** Models, objectives, statistics, update
  rules, and diagnostics should remain understandable and reusable. Allow
  meaningful downstream substitution without hiding the reference behavior.
- **Preserve mathematical intent.** Explain changes to numerical evaluation,
  precision, reduction order, approximation, solver, and convergence behavior.
  Numerical similarity is the target; bitwise identity and exact result equality
  are not required. Reduced allocation alone does not establish that similarity.
- **Keep compiled interfaces small.** Preserve C++20 modules, independently
  compiled implementations, and private implementation dependencies. Assess
  specialization using both execution cost and consumer build cost.

Follow the written guidance in
[Cellerator's coding_style.md](/home/tumlinson/Cellerator/coding_style.md), not
its current implementation as an example. Keep that protected guide unchanged.
Fast imports and consumer recompilation remain very high priorities here;
report material runtime/build tradeoffs with measurements.
