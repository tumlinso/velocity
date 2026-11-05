# Portable source segmentation and relocation

The portable CPU component now lives at `Cellerator/src/velocity/` and can be
built independently. The accelerator archive lives at
`Cellerator/quarantine/velocity-accelerators/`. Cellerator parent-build
integration is optional and disabled by default. This relocation preserves the
component, package target, and module names; it does not change the numerical
methods or establish new qualification results.

## Portable component

The active component consists of:

- `src/velocity.cppm` and `src/velocity.precision.cppm`, their compiled
  implementations, and private `src/detail` arithmetic and acquisition code.
- `velocity::NumericView` and `NumericEdit` for FP16/FP32/FP64 storage,
  mixed W/S/U inputs, explicit borrowed/retained FP16 buffers, bounded workspace,
  deterministic/stochastic estimators, and scalar/compiled CPU entrypoints.
- Cellerator's header-only, CPU-only binary16 helper:
  `<Cellerator/ops/numeric/binary16.hh>`, `load_f16(std::uint16_t)`,
  `round_f16(float/double)`, and target `Cellerator::binary16`.
- CPU source-promotion and advisory sensitivity utilities.
- The ordinary CMake package/module-provider files and presets, CPU examples,
  CPU test sources/consumers, and CPU qualification/benchmark support.

The active build has no CUDA module, toolkit discovery, accelerator target,
ISA selection, intrinsic header, or dependency on the quarantine. The compiled
CPU entrypoints remain available; the AVX2-specific availability query is
removed. The precision benchmark calls the compiled portable path `cpu`.

AnnData++ remains required for its core arrays, axes, and dataset interface; it
does not supply velocity FP16 storage, codecs, or device transfers. Velocity's
numeric views wrap caller-provided raw half bits and retain an explicit owner
when requested. The helper converts values on CPU without the full Cellerator
runtime. Builds inside Cellerator use its adjacent `ops/numeric/binary16`
target; standalone clones discover the installed `CelleratorBinary16` package;
the installed velocity CONFIG package finds `CelleratorBinary16` transitively.
This view contract is not a persistent format or a future CellShard format.

## Quarantine and evidence

CUDA and AVX2/F16C implementations and their accelerator-specific support live
in quarantine/velocity-accelerators (`Cellerator/quarantine/velocity-accelerators/` in the parent checkout).
Original shared/mixed sources were preserved before selecting their portable
bodies. The original comparison tools and compact reports are available with
that archive; existing historical reports remain at their original paths.

The combined CPU/AVX2/CUDA comparison runners are archived. The active tree
retains the portable precision driver and CPU benchmark source, along with
the earlier CPU-only qualification tools. Its CPU benchmark accepts
`--backend scalar` or `--backend cpu`.

The [relocation inventory](../planning/velocity-relocation.json) records the
completed move and preserved file identities. The earlier
[segmentation inventory](../planning/portable-segmentation.json) records
which paths were moved, snapshotted, or edited. When included by Cellerator, the
parent build exposes `Cellerator::velocity` only when
`CELLERATOR_ENABLE_VELOCITY` is enabled; that option defaults to `OFF`. Standalone
and installed use retains `velocity::velocity` and the `velocity` and
`velocity.precision` modules. This is source organization, not a numerical or
performance change claim. No configure, build, test, benchmark, or GPU run was
performed for the relocation; historical receipts remain unchanged. The later
FP16 ownership migration has separate native and installed-consumer evidence in
[validation](validation.md), and the relocation receipts do not qualify it.
