# Tsonic GPU Starter Kit

This directory is the working specification for the backend-neutral GPU target family.

`tsonic-gpu` is not a Triton backend, not a CUDA backend, not a Python target, and not a
runtime convenience layer. It owns the shared GPU compiler model that every concrete GPU
backend must obey: kernel extraction, GPU legality, tensor shape/dtype/device facts, memory
effects, launch ABI, backend capability negotiation, diagnostics, and test gates.

## Read Order

1. `01-context-and-boundaries.md`
2. `02-architecture-and-abi.md`
3. `03-kernel-extraction-and-legality.md`
4. `04-tensor-shape-dtype-device-model.md`
5. `05-backend-capability-contract.md`
6. `06-host-target-integration.md`
7. `07-work-slices.md`
8. `08-tests-and-gates.md`
9. `09-user-code-examples.md`
10. `10-worker-briefing.md`

## One-Line Direction

Compile explicit TypeScript GPU kernels into a target-neutral GPU IR, prove legality and
memory behavior once, then hand that IR to selected GPU backend plugins such as Triton.

## Non-Negotiables

- GPU compilation is explicit. Normal loops do not silently become GPU kernels.
- TSTS remains the TypeScript authority. GPU checks are target legality checks after TS semantics.
- `tsonic-gpu` owns no Python package emitter, JS runtime, Node runtime, PyTorch provider, or Triton syntax.
- Backends are selected explicitly and report capabilities explicitly.
- Unsupported GPU constructs fail closed with source spans and capability ids.
- No CPU fallback for GPU code unless a user-selected backend policy explicitly asks for a CPU backend.
- No product dependency on `.analysis/`, generated binding sidecars, or open runtime reflection.

## Architecture Sketch

```text
 TypeScript source
       |
       v
     TSTS
  TS checking + extension facts
       |
       v
 tsonic-gpu target family
  +-----------------------------+
  | kernel extraction           |
  | GPU legality checks         |
  | tensor dtype/device facts   |
  | memory/effect analysis      |
  | launch ABI construction     |
  | backend capability matching |
  +-----------------------------+
       |
       v
 selected GPU backend plugin
  +-------------------+      +-------------------+
  | gpu-triton        |      | e.g. gpu-cuda     |
  | Triton/Python     |      | CUDA/C++          |
  +-------------------+      +-------------------+
       |
       v
 selected host target
  +-------------------+
  | tsonic-python     |
  | package/project   |
  +-------------------+
```

## Completion Bar

The first useful product path is:

1. TypeScript kernel source with explicit GPU declaration.
2. TSTS checked program with normal TS semantics.
3. GPU IR with explicit tensors, dtypes, device ids, shapes, memory effects, and launch metadata.
4. Backend capability match against a fake backend and `gpu-triton`.
5. Python host output through `tsonic-python`.
6. Generated Python imports and launches Triton code without compiler-side guesses.

