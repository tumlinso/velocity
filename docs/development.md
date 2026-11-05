# Development and Project Control

## Present development state

The repository currently contains documentation and workflow setup. It has no
numerical implementation, CMake configuration, executable module example,
numerical test suite, or benchmark. The next user-directed step is implementation
planning. Do not infer permission to begin implementation from the architecture
or source-opportunity documents.

The intended public entry point is `import velocity;` using C++20 named modules.
Use [AnnData++](https://github.com/tumlinso/anndatapp/blob/main/docs/development.md)
as a tooling reference:
Clang 18 with matching `clang-scan-deps`, CMake >=3.28, and Ninja >=1.11 are a
reasonable initial target. They are not a qualified velocity.cppm toolchain
until a real module build and installed consumer have passed. Use CMake's
module file sets and dependency scanning when that build is introduced.

## Contribution and scope

Read [AGENTS.md](../AGENTS.md), [STYLE.md](../STYLE.md),
[architecture.md](architecture.md), and [validation.md](validation.md).
Follow the written, protected
[Cellerator coding guide](/home/tumlinson/Cellerator/coding_style.md).
Keep numerical calculations compact and memory/ownership operations explicit.

Inspect current source, method options, and attribution before adapting scVelo
or accelerated-ports code. Preserve upstream license notices for any derived
implementation. Documentation of an opportunity is not a measured improvement.

Use configured Codex subagents for useful bounded work. Coordinate overlapping
files and shared compute. Preserve other agents' changes, upstream checkouts,
supplied data, native workflow state, and historical evidence.

## Project Control identities and workflow

The workspace and repository alias are both `velocity-cppm`; the intended
authority ID is `velocity-cppm-authority`. Registration uses the actual root
`/home/tumlinson/velocity.cppm`. The local origin is
`git@github.com:tumlinso/velocity.cppm.git`.

The installed [Todo Orchestrator skill](/home/tumlinson/.agents/skills/todo-orchestrator/SKILL.md)
describes supported workflow routes. Routine work in the registered workspace
uses `next_task`, then the opaque handle with `inspect_task`, `coordinate_task`,
and `finish_task`. Observer reads require no claim. `next_task` is not a
readiness check because it can claim or resume work.

Canonical local state is `.todo-orchestrator/authority.sqlite3`, with project
identity in `.todo-orchestrator/project.json`. Those files and SQLite sidecars
are private/ignored. Never hand-edit the database, generated projections,
leases, or recovery state. Keep owner and profile credentials outside the
checkout; do not publish tokens, private previews, or runtime artifacts.

Authored planning files are reviewable inputs. Todo's accepted plan and task
state are authority. No implementation plan or tasks are to be installed in
this setup: the user explicitly reserved planning for the next step.

## Setup verification

[The setup receipt](../planning/setup-receipt.json) records the actual result,
installed versions, identities, and limitations. Check these independently:

1. The registry maps the intended alias to the intended repository root.
2. The initialized Todo authority has matching project/repository/authority
   identities and an empty plan/task state.
3. Native integration refs and private provider runtime directories exist.
4. Host credentials are scoped to the exact project, authority, principal,
   profile, and permitted actions; routine profiles do not receive the owner
   bearer.
5. A fresh Project Control process can retrieve exact source and semantic
   workflow state. An existing connected process may retain an older registry
   or provider result; report that separately.

Service health alone is not project readiness. A partial response caused only
by an unavailable optional context extension differs from missing authority,
credentials, or source access. Preserve active attempts and unrelated sessions
when resolving setup problems. Do not restart shared services or change their
permissions merely to silence a diagnostic.

## Artifacts and checks

Keep generated biological data, builds, environments, binaries, profiler output,
and raw measurement campaigns ignored and preserved locally. Commit source,
small reproducible fixtures/generators, compact evidence, and source identities
when publication is authorized. Commit and push only with user authorization.

For documentation changes, check relative links, source-reference provenance,
JSON syntax, status claims, secret exclusion, and `git diff --check`.
Native build, interoperability, and performance gates will be established
with the implementation plan; do not report unexecuted checks as passing.
