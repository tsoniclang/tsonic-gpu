# User Code Examples

These examples define the desired user-facing shape. Exact import names can change, but the semantic
flow must remain.

## Vector Add

```ts
import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Tensor } from "@acme/tensor";

export const add = kernel(function add(
  a: Tensor<float32, [N]>,
  b: Tensor<float32, [N]>,
  out: Tensor<float32, [N]>,
) {
  const i = gpu.globalId(0);
  if (i < N) {
    out[i] = a[i] + b[i];
  }
});
```

GPU core output:

```text
kernel add
parameters:
  a readonly tensor float32 [N]
  b readonly tensor float32 [N]
  out mutable tensor float32 [N]
launch:
  one-dimensional grid over N
effects:
  reads a[i], b[i]
  writes out[i]
```

## Fused Activation

```ts
export const geluApprox = kernel(function geluApprox(
  x: Tensor<float32, [N]>,
  out: Tensor<float32, [N]>,
) {
  const i = gpu.globalId(0);
  const v = x[i];
  out[i] = 0.5 * v * (1.0 + gpu.tanh(0.79788456 * (v + 0.044715 * v * v * v)));
});
```

Backend requirement:

- float32 arithmetic,
- `tanh`,
- scalar constants,
- tensor load/store.

## Reduction

```ts
export const sum = kernel(function sum(
  values: Tensor<float32, [N]>,
  out: Tensor<float32, [1]>,
) {
  const partial = gpu.blockReduceSum(values);
  if (gpu.localId(0) === 0) {
    out[0] = partial;
  }
});
```

This requires backend reduction capability. Backends without it reject the kernel.

## Matrix Multiply

```ts
export const matmul = kernel(function matmul(
  a: Tensor<float32, [M, K]>,
  b: Tensor<float32, [K, N]>,
  c: Tensor<float32, [M, N]>,
) {
  const row = gpu.globalId(0);
  const col = gpu.globalId(1);
  let acc = 0.0;
  for (let k = 0; k < K; k++) {
    acc += a[row, k] * b[k, col];
  }
  c[row, col] = acc;
});
```

GPU core proves:

- shared `K`,
- valid indexing,
- write to `c[row, col]`,
- no host calls,
- bounded loop.

Backend decides:

- direct loop lowering,
- tiling,
- tensor cores,
- block sizes.

## Hard Reject Example

```ts
export const bad = kernel(function bad(values: Tensor<float32, [N]>, out: Tensor<float32, [N]>) {
  const i = gpu.globalId(0);
  out[i] = JSON.parse("{}").x;
});
```

This rejects because device code cannot call host JSON APIs.

