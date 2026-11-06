# Velocity++ numerical development style

Velocity++ follows the ccell numerical development style: compact C-style
scientific code exposing memory and execution, with C++ language features for
real specialization and composition. This is a numerical development reference
for competent programmers, including intentionally unchecked access.

The binding policy is self-contained for standalone checkouts:

- Use direct arrays, pointers, indices, signed strides, arithmetic, loops,
  plain operation-specific structs, and deliberate calls. Preserve compact
  mathematical expressions, short conventional notation, ternaries, and coherent
  algorithm phases. Labels/goto are permitted where they reveal the calculation.
- No containers, generic standard-library algorithms, optional/string/span,
  smart pointers, or hidden owning/growing machinery in shipped numerical code
  or examples. Tests and qualification tooling may use ordinary library facilities.
- Include only necessary scalar headers (`stddef.h`, `stdint.h`, `float.h`).
  Use compiler math/memory builtins with required numerical semantics. No
  cstdlib/stdlib.h, cmath/math.h, algorithm, limits, numeric, utility, type_traits,
  memory, or other broad runtime headers in production numerical source.
- All standard reference sizing, validation, preparation, binding, diagnostics,
  topology and execution allocate nothing. The caller supplies sufficient
  buffers and retains them. Return capacity/status/location without allocating
  an error. Do not introduce allocate/release helpers or numerical owners.
- Callers perform AnnData++ construction and descriptor acquisition. The
  reference borrows existing library descriptors; it does not define a parallel
  matrix/storage model. State lifetime and aliasing remain explicit.
- Callers control threads, worker-local workspace, scheduling and reductions.
  CPU reference calls are synchronous; no implicit synchronization or global
  mutable dispatch registry. Unsafe paths transfer documented preconditions.
- Consumers may replace a primitive, connected region, fused block, or whole
  algorithm with different layouts and mathematics. Only retained boundaries
  impose compatibility. Templates preserve arbitrary concrete intermediate types;
  runtime typed function pointers retain custom state/results in caller context.
  Replacements may allocate explicitly where necessary; that exception does not
  apply to standard reference implementations.
- Keep ordinary kernels and common specializations separately compiled. Export
  small descriptors/declarations and narrow composition templates. Private edits
  must preserve public BMIs and consumer objects; measure rebuild behavior.
- Preserve declared precision, evaluation/reduction order, masks, convergence
  and fallback semantics. No blanket fast-math. Style compliance is distinct
  from scientific equivalence; performance and equivalence claims need evidence.

The protected Cellerator `coding_style.md` is the baseline compact style; its
independent source must not be changed. In a ccell checkout, the root
`coding_style.md` adds the full contributor contract. Do not weaken these rules
without explicit user authorization.
