# Architecture And ABI

The GPU architecture has four layers:

```text
             +-----------------------------------------+
             | TypeScript source                       |
             | explicit kernel declarations             |
             +--------------------+--------------------+
                                  |
                                  v
             +-----------------------------------------+
             | TSTS checked program                    |
             | types, symbols, signatures, facts        |
             +--------------------+--------------------+
                                  |
                                  v
             +-----------------------------------------+
             | GPU target core                         |
             | extraction, IR, legality, ABI, effects   |
             +--------------------+--------------------+
                                  |
                    capability-checked GPU IR
                                  |
                                  v
             +-----------------------------------------+
             | selected GPU backend                    |
             | Triton, CUDA, Metal, other backends      |
             +--------------------+--------------------+
                                  |
                    host artifact contributions
                                  |
                                  v
             +-----------------------------------------+
             | selected host target                    |
             | Python first                            |
             +-----------------------------------------+
```

## Core Data Model

The GPU core should define these structures in product code:

- `GpuKernelDeclaration`
- `GpuKernelParameter`
- `GpuTensorType`
- `GpuScalarType`
- `GpuShapeExpr`
- `GpuDeviceRef`
- `GpuLayout`
- `GpuMemorySpace`
- `GpuEffect`
- `GpuIrModule`
- `GpuIrFunction`
- `GpuIrBlock`
- `GpuIrOperation`
- `GpuLaunchPlan`
- `GpuBackendCapabilitySet`
- `GpuBackendDiagnostic`

Names can change, but the concepts must exist as first-class structures.

## Kernel ABI

Every kernel must produce a complete ABI record:

```text
Kernel ABI
  name
  source span
  parameters
    - role: input | output | inout | scalar | shape | meta
    - dtype
    - rank
    - shape symbols
    - layout
    - device
    - mutability
  launch
    - grid dimensions
    - block dimensions or backend meta parameters
    - stream/device policy
  effects
    - reads
    - writes
    - atomics
    - barriers
    - aliasing constraints
```

The ABI must be deterministic. The backend and host target cannot infer missing parameter roles.

## GPU IR Requirements

GPU IR must be structured, not target syntax strings. It needs at minimum:

- constants,
- scalar arithmetic,
- tensor element load/store,
- vectorized masks,
- conditionals,
- counted loops when legal,
- reductions,
- function calls to whitelisted GPU intrinsics,
- barriers where backend supports them,
- atomics where backend supports them.

The IR must preserve source spans for diagnostics.

## Host Artifact Boundary

The GPU target can produce host artifact contributions:

- generated kernel module request,
- dependency request,
- launch wrapper request,
- test/run metadata request.

It must not directly decide Python project file layout. `tsonic-python` owns Python packaging.

## Fact Discipline

GPU compilation consumes:

- TSTS selected declarations,
- TSTS selected signatures,
- source primitive facts,
- target/provider facts for tensor libraries,
- explicit kernel marker facts,
- host library facts.

It must not consume:

- file naming heuristics,
- provider member names without selected identity,
- emitted code strings,
- hidden side tables that duplicate TSTS state.

