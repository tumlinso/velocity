# Portable velocity component

This is first-party Cellerator source with its own CMake entrypoint. Configure
this directory directly to build without the rest of Cellerator or a GPU.
The public named modules remain `velocity` and `velocity.precision`; the CMake
targets `Cellerator::velocity` and `velocity::velocity` name the same component.

Read README.md, docs/api.md, docs/architecture.md and STYLE.md. Follow the
protected parent coding_style.md when working in the Cellerator checkout.
Keep compiled numerical implementations private and module imports small.

Preserve FP16 through velocity's explicit `NumericView` and `NumericEdit`
types, FP32/FP64 computation, scalar references, scientific masks/fallbacks,
and bounded workspace. `Cellerator::binary16` is a small CPU-only, header-only
numeric helper; its installed package is `CelleratorBinary16`. It is not the
full Cellerator runtime. AnnData++ arrays do not supply FP16 storage, codecs,
or device transfers. Keep raw half-bit borrows and retained owners explicit;
never reinterpret an AnnData integer array as FP16.

The canonical repository is `tumlinso/velocity`, registered as Cellerator's
`src/velocity` Git submodule. CUDA and AVX2/F16C implementations are preserved
in the parent checkout at `quarantine/velocity-accelerators/`, outside this
repository. They have
no active build target, toolkit discovery, package export, or source dependency.
AnnData++ remains independently owned: do not overwrite its sources or rename
its package/module/type identifiers as part of this work. Keep the active
velocity target CPU-only; the historical accelerator archive is not active
implementation evidence for this FP16 ownership migration.

The original velocity checkout retains its Git history, workflow authority,
build trees, and ignored raw artifacts. Authored records here are historical
inputs/evidence, not a separate active Todo authority. Follow the parent
workflow for new component work. The user explicitly requested this relocation
without builds or tests; the relocation record must not claim qualification.

The historical local checkout now resides at `/home/tumlinson/velocity`.
Recorded historical source paths remain provenance, not active aliases.
No old-path symlink or forwarding checkout is maintained.
