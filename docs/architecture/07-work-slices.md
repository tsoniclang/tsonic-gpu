# Work Slices

The slices are large enough for meaningful progress and narrow enough to keep review quality high.

## G1 — Target Shell And Fail-Closed Core

Build:

- npm package and ESM target pack,
- GPU target options,
- backend selection validation,
- fake backend registration,
- fail-closed backend for every kernel attempt,
- diagnostics with source spans,
- architecture scanners.

Proof:

- target registers,
- missing backend fails,
- unknown backend fails,
- non-kernel host code is ignored by GPU target,
- explicit kernel marker produces unsupported diagnostic until extraction exists.

## G2 — GPU IR And Tensor ABI

Build:

- GPU IR types,
- tensor/dtype/shape/device facts,
- launch ABI structure,
- memory effect records,
- fake backend artifact emission from IR.

Proof:

- hand-authored test IR validates,
- invalid dtype/device/shape records reject,
- fake backend accepts known IR and rejects unsupported IR,
- no Triton/Python imports in GPU core.

## G3 — Kernel Extraction And Legality

Build:

- extraction from explicit kernel marker facts,
- source-to-IR for scalar arithmetic,
- tensor load/store,
- simple conditionals,
- counted loops,
- legality diagnostics.

Proof:

- vector add kernel extracts,
- masked write extracts,
- host call inside kernel rejects,
- dynamic `any` inside kernel rejects,
- unbounded loop rejects.

## G4 — Backend Capability Negotiation

Build:

- backend capability schema,
- operation-to-capability mapping,
- selected backend validation,
- fake backend capability tests,
- backend artifact contract.

Proof:

- reduction accepted only by backend reporting reduction capability,
- atomic rejected when backend lacks atomic capability,
- dtype mismatch diagnosed before lowering.

## G5 — Python Host Integration Contract

Build:

- host artifact contribution interface,
- dependency contribution interface,
- launch wrapper request model,
- test fixture with `tsonic-python` shape but no direct dependency on Python target internals.

Proof:

- GPU backend artifacts can be packaged by a fake host,
- GPU core does not write Python project files,
- missing host integration reports deterministic error.

## G6 — Real Kernel Coverage

Build:

- elementwise binary/unary kernels,
- broadcast rules,
- reductions,
- tiled matrix multiply IR,
- common math intrinsics,
- shape constraints.

Proof:

- examples in `09-user-code-examples.md` pass through fake backend,
- backend capability rejection tests cover unsupported variants.

## G7 — Final Gates

Build:

- capability ledger,
- inventory audit,
- integration documentation,
- full architecture scanner suite,
- cross-repo host/backend proof with `tsonic-python` and `gpu-triton` once those APIs are available.

Proof:

- all GPU rows classified as implemented, capability-gated, or hard-reject,
- no unclassified GPU capability remains,
- no CPU fallback, no hardcoded backend, no Python leakage.

