// Every GPU legality-reject lane, as compilable source. The capability
// ledger test proves each lane fails closed end-to-end; the architecture
// inventory audit proves no reject capability exists without a lane here.

const header = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor, Int32Tensor, Float32Matrix, Float32HostTensor } from "@acme/tensor";
import type { int32 } from "@tsonic/core/types.js";
`;

function lane(name, capability, body) {
  return { name, capability, files: { "index.ts": `${header}\n${body}` } };
}

export const ledgerLanes = [
  lane(
    "host call inside kernel",
    "gpu.device.host-call",
    `function hostScale(value: number): number {
  return value * 2;
}

export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = hostScale(a[i]);
});
`,
  ),
  lane(
    "parameter without tensor or scalar fact",
    "gpu.kernel.parameter-type",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor, cb: any) {
  const i = gpu.globalId(0);
  out[i] = a[i];
});
`,
  ),
  lane(
    "unbounded while loop",
    "gpu.kernel.unbounded-loop",
    `export const k = kernel(function k(out: Float32Tensor) {
  while (true) {
    out[0] = 1.0;
  }
});
`,
  ),
  lane(
    "for loop without condition",
    "gpu.kernel.unbounded-loop",
    `export const k = kernel(function k(out: Float32Tensor) {
  for (;;) {
    out[0] = 1.0;
  }
});
`,
  ),
  lane(
    "mutable local reassignment",
    "gpu.kernel.mutable-local",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  let acc = 0.0;
  acc = acc + a[i];
  out[i] = acc;
});
`,
  ),
  lane(
    "tensor parameter used as a value",
    "gpu.kernel.tensor-value",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const t = a;
  const i = gpu.globalId(0);
  out[i] = a[i];
});
`,
  ),
  lane(
    "mixed operand dtypes",
    "gpu.kernel.mixed-dtype",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i] + 1;
});
`,
  ),
  lane(
    "kernel returning a value",
    "gpu.kernel.return-value",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i];
  return 1.0;
});
`,
  ),
  lane(
    "kernel writing no tensor output",
    "gpu.kernel.no-output",
    `export const k = kernel(function k(a: Float32Tensor) {
  const i = gpu.globalId(0);
  const v = a[i];
});
`,
  ),
  lane(
    "kernel marker on a non-function value",
    "gpu.kernel.declaration-form",
    `export const k = kernel(42);
`,
  ),
  lane(
    "kernel assigned to a let binding",
    "gpu.kernel.declaration-form",
    `export let k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i];
});
`,
  ),
  lane(
    "block-scoped binding shadowing a kernel value",
    "gpu.kernel.binding",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  if (i < 1) {
    const i = 2;
    out[i] = 1.0;
  }
});
`,
  ),
  lane(
    "non-int32 tensor index",
    "gpu.kernel.index-dtype",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[a[i]];
});
`,
  ),
  lane(
    "unsupported statement form",
    "gpu.kernel.statement",
    `export const k = kernel(function k(out: Float32Tensor) {
  out[0] = 1.0;
  debugger;
});
`,
  ),
  lane(
    "unsupported expression form",
    "gpu.kernel.expression",
    `export const k = kernel(function k(out: Float32Tensor) {
  const f = () => 1.0;
  out[0] = 1.0;
});
`,
  ),
  lane(
    "unsupported operator",
    "gpu.kernel.operator",
    `export const k = kernel(function k(n: int32, out: Float32Tensor) {
  const m = n & n;
  out[0] = 1.0;
});
`,
  ),
  lane(
    "non-bool conditional guard",
    "gpu.kernel.condition",
    `export const k = kernel(function k(n: int32, out: Float32Tensor) {
  if (n) {
    out[0] = 1.0;
  }
});
`,
  ),
  lane(
    "loop with unsupported bound form",
    "gpu.kernel.loop-form",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor, n: int32) {
  for (let k2 = 0; k2 <= n; k2++) {
    out[k2] = a[k2];
  }
});
`,
  ),
  lane(
    "non-literal thread index dimension",
    "gpu.kernel.thread-index",
    `export const k = kernel(function k(n: int32, out: Float32Tensor) {
  const i = gpu.globalId(n);
  out[0] = 1.0;
});
`,
  ),
  lane(
    "kernel capturing a host value",
    "gpu.kernel.host-capture",
    `const factor = 2.0;

export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i] * factor;
});
`,
  ),
  lane(
    "single index into a rank-2 tensor",
    "gpu.kernel.index-arity",
    `export const k = kernel(function k(m: Float32Matrix, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = m[i];
});
`,
  ),
  lane(
    "indexing through a non-identifier target",
    "gpu.kernel.indexing",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = (a)[i];
});
`,
  ),
  lane(
    "block reduce over a non-tensor operand",
    "gpu.kernel.reduce-operand",
    `export const k = kernel(function k(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  const v = a[i];
  out[i] = gpu.blockReduceSum(v);
});
`,
  ),
  lane(
    "assignment used as an expression",
    "gpu.kernel.assignment",
    `export const k = kernel(function k(out: Float32Tensor) {
  out[0] = (out[1] = 1.0);
});
`,
  ),
  lane(
    "storing a mismatched dtype",
    "gpu.kernel.store-dtype",
    `export const k = kernel(function k(out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = i;
});
`,
  ),
  lane(
    "mixed tensor device domains",
    "gpu.device.mixed",
    `export const k = kernel(function k(a: Float32HostTensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i];
});
`,
  ),
];
