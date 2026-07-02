# tsonic-gpu

Backend-neutral GPU target family for Tsonic, published as `@tsonic/target-gpu`.

`tsonic-gpu` owns the shared GPU compiler model that every concrete GPU backend must obey:
kernel discovery from explicit marker facts, a structured GPU IR, GPU legality validation,
the tensor shape/dtype/device/layout ABI, memory-effect records, the launch ABI, backend
capability negotiation, and backend-neutral diagnostics. Concrete backends (such as a
Triton backend) and host targets (such as the Python target) live in sibling repositories
and plug in through explicit contracts.

## Principles

- GPU compilation is explicit. Ordinary host code never silently becomes a GPU kernel.
- Backends are selected explicitly through target options and report their capabilities
  explicitly; nothing is inferred from imported libraries.
- Unsupported constructs fail closed with source spans and capability ids. Diagnostics and
  artifacts never coexist.
- The GPU core names no concrete backend, host language, or tensor library in product code.
- There is no CPU recovery path; a kernel the selected backend cannot lower is a compile error.

## Layout

- `src/descriptor/` — `createGpuTargetPack()`, the `@tsonic/target-api` target pack entry.
- `src/options/` — public GPU target options (`backendId`, `hostTargetId`, ...) with fail-closed validation.
- `src/ir/` — GPU IR structures, tensor/dtype/shape/device model, launch ABI, effects, and IR validation.
- `src/capabilities/` — backend capability sets and IR-to-capability matching.
- `src/backends/` — the backend plugin contract, registry, and the fake backend used by tests.
- `src/backend/` — the fail-closed target backend, planner, and kernel extraction (source AST to GPU IR).
- `src/source/` — GPU fact keys, the GPU language surface (`kernel` marker and `gpu` intrinsics),
  the generic tensor provider-package model, and the target semantics extension.
- `src/session/` — a TSTS session-to-compile-input bridge for integration tests and tooling.
- `test/architecture/` — scanners that enforce the boundaries above.

## Build and Test

The sibling `../tsonic` repository must be built first; this repository never builds or
writes into it.

```sh
npm install
npm test
```

The default test suite requires no GPU hardware and no concrete backend: capability
negotiation and artifact contracts are proven against the fake backend.
