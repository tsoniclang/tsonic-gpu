# Context And Boundaries

The GPU target exists because high-performance kernels need a different legality and execution
model from ordinary host code. The host target decides how to package and call generated code.
The GPU target decides what a valid GPU kernel means.

## What This Repository Owns

- Target pack identity for the GPU target family.
- Public GPU target options.
- Kernel discovery from explicit source declarations.
- GPU IR schema and validation.
- Tensor dtype, rank, shape, layout, stride, device, and mutability facts.
- Memory-space and effect model.
- Launch ABI.
- Backend capability negotiation.
- Backend-neutral diagnostics.
- Fake backend used for tests.

## What This Repository Does Not Own

- Python package layout.
- Python syntax printing.
- Triton syntax printing.
- PyTorch API definitions.
- NumPy API definitions.
- JavaScript compatibility runtime.
- Node compatibility runtime.
- CUDA-specific syntax.
- Hardware-specific autotuning policy.

## Explicit Backend Selection

The target must require explicit backend selection through installed plugins. Every GPU-family
package declares the core host manifest in package.json:

```json
"tsonic": { "kind": "plugin", "contractVersion": 1, "entry": "." }
```

and exports `createTsonicPlugin()`. The returned plugin object carries the plugin kind:
`kind: "target"` for `@tsonic/target-gpu`, `kind: "gpu-backend"` for backend packages, and
`kind: "gpu-host"` for host integration packages. Plugin kinds live on returned objects, never
in package.json metadata. Routing discovered gpu-backend/gpu-host plugins into the GPU target
plugin is a tsonic core host requirement (`docs/core-host-requests.md`); local composition
passes sub-plugin entries to `createTsonicPlugin({ plugins })`, and the selected backend and
host come from target options:

```json
{ "id": "gpu", "options": { "backendId": "triton", "hostTargetId": "python" } }
```

The exact API can differ, but the semantic contract cannot:

- backend id is explicit,
- host target is explicit,
- missing backend is a configuration error,
- unsupported backend capability is a diagnostic,
- no backend is inferred from imported libraries.

## Relationship To Python And PyTorch

PyTorch is a Python library and belongs to the Python target/library/provider model. A user should
be able to import PyTorch-owned values through Python library/provider packages, then pass tensors
to GPU kernels through facts and ABI rows.

`tsonic-gpu` must not hardcode PyTorch as the tensor source of truth. The tensor ABI must be able to
serve PyTorch, NumPy-like tensors, custom tensor libraries, and any host target.

## Relationship To Triton

Triton is the first concrete GPU backend. `tsonic-gpu` must not contain Triton syntax, names, or
Python import paths. Triton features enter through a backend capability contract.

## No Runtime By Default

There is no default `gpu-runtime` repository in this architecture.

A shared GPU runtime becomes justified only if all of these become true:

1. The behavior is compiler-owned rather than backend-owned.
2. The behavior is shared by multiple GPU backends.
3. The behavior cannot be expressed through generated host code or backend plugin code.
4. Repeating it would create divergent semantics.

Examples that do not justify a runtime:

- Triton launch syntax.
- PyTorch tensor allocation.
- CUDA-specific stream helpers.
- Backend-local cache helpers.

Examples that could justify a runtime after proof:

- stable cross-backend tensor descriptor ABI,
- deterministic shape-check error objects shared across host targets,
- common kernel launch metadata serialization required by more than one backend.

