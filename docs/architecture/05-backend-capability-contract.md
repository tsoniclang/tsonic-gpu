# Backend Capability Contract

Every concrete GPU backend must describe what it can lower. The GPU core validates against that
contract before asking the backend to emit target artifacts.

## Backend Interface

The backend interface needs these operations:

```text
describeCapabilities() -> GpuBackendCapabilitySet
validate(irModule, launchPlan, facts) -> diagnostics
lower(irModule, launchPlan, hostContext) -> backendArtifacts
```

The backend must not re-run TypeScript analysis. It receives GPU IR and facts.

## Capability Set

The capability set should declare:

- supported scalar dtypes,
- supported tensor ranks,
- supported layouts,
- supported devices,
- supported arithmetic ops,
- supported math intrinsics,
- supported reductions,
- supported atomics,
- supported memory spaces,
- supported barrier operations,
- supported shape expression forms,
- launch model,
- host artifact requirements.

## Capability Matching

```text
GPU IR operation
       |
       v
required capability id
       |
       v
selected backend capability set
       |
       +-- present -> lower
       |
       +-- absent  -> diagnostic
```

Example:

```text
operation: atomicAdd(out[i], value)
requires: gpu.atomic.add.float32
backend: triton
result: accepted only if Triton capability row says supported
```

## Backend Artifact Contract

Backends return artifact declarations, not arbitrary writes:

```text
GpuBackendArtifacts
  modules
    - path
    - language
    - content model or printer tree
  dependencies
    - python package
    - version constraints
  launch wrappers
    - host function name
    - kernel name
    - meta parameter requirements
```

The host target decides final file placement.

## Fake Backend

`tsonic-gpu` must include a fake backend for tests. It should:

- accept a narrow deterministic capability set,
- emit structured test artifacts,
- prove capability negotiation,
- allow GPU core tests without Triton or Python installed.

The fake backend is not a product backend.

