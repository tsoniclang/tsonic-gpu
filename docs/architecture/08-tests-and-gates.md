# Tests And Gates

The GPU target must be tested at four levels: API, IR, capability, and integration.

## Required Test Families

### Target Pack Tests

- registers with target registry,
- validates options,
- rejects unknown options,
- rejects missing backend,
- rejects invalid host target relation.

### Kernel Extraction Tests

- explicit kernel marker discovered,
- ordinary function not treated as kernel,
- parameters mapped to ABI records,
- selected TSTS declarations/signatures preserved.

### GPU IR Tests

- vector add IR,
- masked load/store IR,
- conditional IR,
- loop IR,
- reduction IR,
- shape constraint IR,
- memory effects.

### Legality Tests

- host call inside kernel rejects,
- dynamic call rejects,
- unsupported dtype rejects,
- mismatched devices reject,
- unbounded loop rejects,
- allocation inside kernel rejects,
- recursion rejects.

### Backend Capability Tests

- fake backend accepts supported operation,
- fake backend rejects missing capability,
- artifact contract deterministic,
- diagnostics include capability id.

### Architecture Tests

Scanners must reject:

- Triton imports in `tsonic-gpu`,
- Python host project writes in `tsonic-gpu`,
- PyTorch hardcoding in GPU core,
- generated code string inference,
- CPU fallback,
- `.analysis/` imports,
- TSTS internal imports.

## Hardware-Free Rule

The default GPU target test suite must not require a GPU. Hardware tests belong in a separate gate.

Default tests use:

- fake backend,
- generated artifact inspection,
- syntax-only tests where appropriate,
- capability diagnostics.

## Hardware Gate

When a real backend is wired:

- GPU tests must be isolated behind explicit opt-in environment markers,
- absence of GPU must skip hardware tests rather than weaken product code,
- CPU fallback must not be used to pass GPU hardware tests.

