# Kernel Extraction And Legality

GPU kernels must be explicit. The compiler cannot transform arbitrary source loops into GPU kernels
because that would silently change execution, memory, error, and performance semantics.

## Kernel Declaration Forms

The first supported form should be a source-core or GPU-package marker that attaches a neutral
kernel fact to a function:

```ts
import { kernel } from "@tsonic/gpu/lang.js";
import type { Tensor } from "@acme/tensor";

export const add = kernel(function add(a: Tensor<float32, 1>, b: Tensor<float32, 1>, out: Tensor<float32, 1>) {
  const i = gpu.globalId(0);
  out[i] = a[i] + b[i];
});
```

The exact marker spelling belongs to the GPU source package, not TSTS. The GPU target consumes the
finalized marker facts.

## Extraction Pipeline

```text
TSTS function declaration
       |
       v
kernel marker fact?
       |
       +-- no  -> ordinary host target handles function
       |
       +-- yes -> GPU extraction
                 |
                 v
            Type and fact validation
                 |
                 v
            GPU legality walk
                 |
                 v
            GPU IR construction
                 |
                 v
            backend capability check
```

## Legality Categories

### Always Legal In The Core GPU Subset

- numeric scalar arithmetic,
- tensor indexing with proven integer indices,
- explicit load/store,
- simple `if`,
- bounded loops when bounds are scalar/kernel meta values,
- common math intrinsics if backend reports support,
- returns from `void` kernels.

### Legal Only With Backend Capability

- reductions,
- atomics,
- barriers,
- shared/local memory,
- vectorized loads,
- fast math,
- approximate math,
- half/bfloat16 arithmetic,
- matrix/tensor core operations,
- dynamic shapes.

### Hard Reject

- allocation inside kernel,
- host library calls,
- Python/JS dynamic value calls,
- exception throwing,
- async/await,
- closures capturing host objects,
- reflection,
- string operations,
- object property enumeration,
- unbounded loops,
- recursive calls,
- prototype mutation,
- `any`/`unknown` dynamic operations inside device code.

## Legality Diagnostics

Every rejection needs:

- source span,
- kernel name,
- capability id,
- selected backend id,
- reason,
- suggested rewrite when deterministic.

Example:

```text
GPU_UNSUPPORTED_KERNEL_OPERATION
kernel: normalize
backend: triton
capability: gpu.device.dynamic-call
source: x.apply(scale)
reason: device code cannot perform dynamic method calls
```

## No Host Leakage

GPU extraction must not depend on Python, Triton, PyTorch, or Rust names. It can depend on tensor
facts and backend capabilities.

