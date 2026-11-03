# Working on velocity.cppm

## Intent and scope

Read [README.md](README.md), [architecture](docs/architecture.md), and the
[source-grounded scVelo foundation](docs/scvelo-foundation.md) before design or
implementation. This is a native RNA velocity toolkit and an efficient,
explicitly callable reference for planned Cellerator acceleration and custom
implementations. Substantially improve allocation, memory traffic, working-set
size, and computation rather than mechanically translate scVelo's Python code.
Preserve declared scientific meaning and identify deliberate method changes.
Require numerical similarity to the declared scVelo comparison target, not
bitwise identity or exact result equality. Improved algorithms, solvers,
evaluation order, layouts, and precision are allowed when their numerical
differences are qualified. Set concrete acceptance quantities and tolerances
during planning rather than imposing identity by default.

The present request establishes documentation and Project Control. The user
will plan implementation in the next step. These documents and future tasks do
not authorize starting algorithms, a build system, benchmarks, or GPU work.

## Cooperative agents and preservation

Use the user's configured profiles: `scout`, `engineer`, `executor`,
`optimizer`, `integrator`, `verifier`, `investigator`, `reviewer`, and
root-allocated `principal`. Root retains user intent, cross-project direction,
workflow lifecycle, integration, and final acceptance. Delegate coherent,
independent work when it improves delivery. Use configured Codex subagents,
never Project Control local coding workers for engineering.

Workers are not alone in the workspace. Preserve unknown edits, active work,
Git history, supplied data, and evidence. Do not reset, clean, force-push, or
overwrite another agent's work. Commit and push only when authorized.

## C++ design and coding style

- Follow [STYLE.md](STYLE.md): compact numerical/HPC code, direct calculations,
  visible memory operations, minimal machinery, and small compiled interfaces.
- Also follow the written [Cellerator coding style](/home/tumlinson/Cellerator/coding_style.md),
  rather than Cellerator's current implementation as an example. For
  `velocity.cppm`, fast module imports and consumer recompilation times remain
  **very high priorities**; preserve small public interfaces and implementation
  isolation when applying its hot-path guidance. Measure and report
  runtime/build tradeoffs explicitly. Keep the referenced guide unchanged.
- Target C++20 named modules with `.cppm` interfaces and ordinary compiled
  implementations. `import velocity;` is the intended entry point, not an
  implemented API. Do not require C++23 `import std;`.
- Keep ownership, borrowing, mutability, strides, index widths, numeric types,
  allocation, copying, transfers, and synchronization explicit. Resolve contracts
  at acquisition boundaries and provide deliberate trusted paths where useful.
- Keep numerical loops free of avoidable repeated checks, type-erased dispatch,
  allocation, and library synchronization. Callers control execution.
- Keep scientific components reusable or replaceable at meaningful boundaries.
  Preserve explicit access to the reference implementation when custom paths
  are selected. Do not impose a universal tensor, backend, or scheduler framework.
- Preserve dependency direction: Cellerator may consume this library;
  velocity.cppm does not depend on Cellerator or its runtime. Reuse anndata.cppm
  for AnnData storage and memory integration when that adapter is designed.

## Scientific and performance evidence

Follow [validation.md](docs/validation.md). Inspect pinned upstream source,
the modified accelerated-ports checkout, and actual Cellerator mechanisms.
Their existence is not proof of full velocity support or a measured speedup.
Declare normalization, graph construction, moments, masks, scaling, time,
likelihood, solver, and precision conventions. Distinguish representation-only
improvements from changes to the numerical method or scientific assumptions.

Measure runtime, peak/working memory, allocation count/bytes, and intermediate
sizes on identical workloads before claiming an improvement. Start future
performance probes with bounded, geometry-faithful cases capped at 30 seconds;
larger confirmation needs an explicit time and memory/I/O budget. Keep source
identities, seeds, compact summaries, and raw evidence. Do not run biological
data or benchmark campaigns merely to write documentation.

For authorized GPU design or execution, consult the
[KDA entrypoint](/home/tumlinson/kda/docs/agent-entry.md) and the
[CUDA skill](/home/tumlinson/.agents/skills/cuda/SKILL.md), including the host's
current Scheduler authority and measurement-isolation requirements.

## Project Control and Todo

Workspace/repository alias: `velocity-cppm`. Follow the installed
[Todo Orchestrator skill](/home/tumlinson/.agents/skills/todo-orchestrator/SKILL.md)
and supported Project Control routes. `.todo-orchestrator/authority.sqlite3`
is semantic authority; never hand-edit it, generated projections, or recovery
state. Keep credentials private and outside the repository.

For authorized execution, use `next_task(project="velocity-cppm", ...)` and
the returned handle with `inspect_task`, `coordinate_task`, and `finish_task`.
Use observer reads for orientation and readiness; `next_task` claims/resumes
work and is not a readiness probe. Authored planning files are inputs, not
authority. Registration, initialization, native-provider setup, and fresh-client
workflow visibility require separate verification. See
[development.md](docs/development.md) and [planning](planning/README.md).
