# Worker Briefing

Workers must build the GPU core as a backend-neutral compiler layer. Do not build Triton first and
then backfill abstractions.

## First Work Package

Start with G1 and G2 together:

- package shell,
- target options,
- backend registry,
- fake backend,
- GPU IR structures,
- tensor ABI structures,
- fail-closed diagnostics,
- architecture scanners.

This is the foundation every other GPU slice builds on.

## Allowed References

- `../tsonic-rust` target pack and fail-closed backend pattern.
- `../tsonic-csharp` provider/fact discipline.
- `../tsonic-python` host target spec.
- `../gpu-triton` backend spec.

Use references for architecture only. Do not copy target-specific policy.

## Stop Conditions

Stop and report if:

- a required TSTS public API is missing,
- host target artifact integration is unclear,
- a backend-specific concept appears necessary in GPU core,
- tensor facts require a Python-specific assumption,
- tests need a real GPU to prove core behavior.

## Review Checklist

Before any PR:

- no backend names in GPU core control flow,
- no Python package emitter in GPU core,
- no CPU fallback,
- no `.analysis/` imports,
- all unsupported constructs fail with diagnostics,
- fake backend tests prove capability matching,
- examples from `09-user-code-examples.md` are represented in tests.

