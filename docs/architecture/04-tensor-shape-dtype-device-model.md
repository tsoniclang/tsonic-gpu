# Tensor, Shape, Dtype, And Device Model

The tensor model is the most important shared ABI in the GPU architecture. It must be generic enough
for PyTorch, NumPy-like libraries, custom tensor libraries, and any host target.

## Tensor Type Record

A tensor fact should carry:

```text
Tensor
  elementType
    - scalar dtype
    - signedness
    - width
  rank
  shape
    - static dimensions
    - symbolic dimensions
    - unknown dimensions
  strides
  layout
    - contiguous
    - strided
    - channels-last
    - backend-specific layout id
  device
    - cpu
    - cuda
    - rocm
    - metal
    - backend-specific device id
  mutability
    - readonly
    - mutable
  aliasing
    - noalias
    - may-alias
```

## Dtype Contract

Core dtype ids should be backend-neutral:

- `bool`
- `int8`
- `uint8`
- `int16`
- `uint16`
- `int32`
- `uint32`
- `int64`
- `uint64`
- `float16`
- `bfloat16`
- `float32`
- `float64`

Backends can reject unsupported dtypes.

## Shape Expressions

Shape expressions should support:

- integer literal dimensions,
- symbols,
- products/sums of symbols,
- equality constraints,
- divisibility constraints,
- backend meta parameters.

Example:

```text
A: Tensor<float32, [M, K]>
B: Tensor<float32, [K, N]>
C: Tensor<float32, [M, N]>
```

The GPU target validates that `K` is shared. The Triton backend decides block sizes.

## Device Semantics

The GPU core must reject impossible device combinations:

```ts
kernel(function add(a: Tensor<float32, 1, "cuda">, b: Tensor<float32, 1, "cpu">, out: Tensor<float32, 1, "cuda">) {
  const i = gpu.globalId(0);
  out[i] = a[i] + b[i];
});
```

This fails before backend lowering because the kernel mixes device domains.

## Host Library Facts

Python/PyTorch integration should produce tensor facts for values returned by library calls. The GPU
core consumes those facts without knowing PyTorch-specific APIs.

```ts
import { torch } from "@python/torch";

const a = torch.randn([1024], { device: "cuda", dtype: torch.float32 });
```

The Python provider package can record:

```text
runtime carrier: torch.Tensor
tensor fact:
  dtype float32
  rank 1
  shape [1024]
  device cuda
```

The GPU core does not hardcode `torch.randn`.

