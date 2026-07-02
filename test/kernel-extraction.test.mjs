import { test } from "node:test";
import assert from "node:assert/strict";
import { artifactText, capabilityIds, compileGpu } from "./helpers/gpu-session.mjs";

const vectorAddSource = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";
import type { int32 } from "@tsonic/core/types.js";

export const add = kernel(function add(a: Float32Tensor, b: Float32Tensor, out: Float32Tensor, n: int32) {
  const i = gpu.globalId(0);
  if (i < n) {
    out[i] = a[i] + b[i];
  }
});
`;

test("vector add kernel with masked write extracts through the fake backend", () => {
  const { result } = compileGpu({ files: { "index.ts": vectorAddSource } });
  assert.deepEqual(result.diagnostics, []);
  const record = JSON.parse(artifactText(result, "kernels/add.gpu-fake.json"));
  assert.equal(record.kernel, "add");
  assert.equal(record.hostTarget, "python");
  assert.deepEqual(
    record.parameters.map((parameter) => [parameter.name, parameter.kind, parameter.role]),
    [
      ["a", "tensor", "input"],
      ["b", "tensor", "input"],
      ["out", "tensor", "output"],
      ["n", "scalar", "scalar"],
    ],
  );
  assert.equal(record.launch.gridDimensions, 1);
  assert.deepEqual(record.effects, [
    { kind: "read", parameter: "a" },
    { kind: "read", parameter: "b" },
    { kind: "write", parameter: "out" },
  ]);
  const launchPlan = JSON.parse(artifactText(result, "gpu/launch-plan.json"));
  assert.equal(launchPlan.backend, "fake");
  assert.deepEqual(launchPlan.launchWrappers, [{ hostFunctionName: "add", kernelName: "add", metaParameters: [] }]);
});

test("math intrinsics extract into intrinsic operations", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const smooth = kernel(function smooth(x: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  const v = x[i];
  out[i] = gpu.tanh(v);
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.diagnostics, []);
  const record = JSON.parse(artifactText(result, "kernels/smooth.gpu-fake.json"));
  assert.equal(record.kernel, "smooth");
});

test("counted loops extract as loop operations", () => {
  const source = `import { kernel } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";
import type { int32 } from "@tsonic/core/types.js";

export const copy = kernel(function copy(a: Float32Tensor, out: Float32Tensor, n: int32) {
  for (let k = 0; k < n; k++) {
    out[k] = a[k];
  }
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.diagnostics, []);
  const record = JSON.parse(artifactText(result, "kernels/copy.gpu-fake.json"));
  assert.equal(record.kernel, "copy");
  assert.equal(record.operationCount >= 2, true);
});

test("ordinary host code is ignored by the GPU target", () => {
  const source = `export function greet(name: string): string {
  return "hello " + name;
}

export const answer = 42;
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.artifacts, []);
  assert.deepEqual(result.diagnostics, []);
});

test("host calls inside kernels reject", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

function scale(value: number): number {
  return value * 2;
}

export const bad = kernel(function bad(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = scale(a[i]);
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.artifacts, []);
  assert.ok(result.diagnostics.some((diagnostic) => diagnostic.code === "GPU_UNSUPPORTED_KERNEL_OPERATION"));
  assert.ok(capabilityIds(result.diagnostics).includes("gpu.device.host-call"));
  assert.ok(result.diagnostics.every((diagnostic) => diagnostic.sourceSpan !== undefined));
});

test("dynamic any parameters and calls reject", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const bad = kernel(function bad(a: Float32Tensor, out: Float32Tensor, cb: any) {
  const i = gpu.globalId(0);
  out[i] = cb(i);
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.artifacts, []);
  assert.ok(result.diagnostics.some((diagnostic) => diagnostic.code === "GPU_MISSING_TARGET_FACT"));
  assert.ok(capabilityIds(result.diagnostics).includes("gpu.kernel.parameter-type"));
});

test("unbounded loops reject", () => {
  const whileSource = `import { kernel } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const spin = kernel(function spin(out: Float32Tensor) {
  while (true) {
    out[0] = 1.0;
  }
});
`;
  const whileResult = compileGpu({ files: { "index.ts": whileSource } }).result;
  assert.deepEqual(whileResult.artifacts, []);
  assert.ok(capabilityIds(whileResult.diagnostics).includes("gpu.kernel.unbounded-loop"));

  const forSource = `import { kernel } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const spinFor = kernel(function spinFor(out: Float32Tensor) {
  for (;;) {
    out[0] = 1.0;
  }
});
`;
  const forResult = compileGpu({ files: { "index.ts": forSource } }).result;
  assert.deepEqual(forResult.artifacts, []);
  assert.ok(capabilityIds(forResult.diagnostics).includes("gpu.kernel.unbounded-loop"));
});

test("mutable locals reject", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const bad = kernel(function bad(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  let acc = 0.0;
  acc = acc + a[i];
  out[i] = acc;
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.artifacts, []);
  assert.ok(capabilityIds(result.diagnostics).includes("gpu.kernel.mutable-local"));
});

test("kernels writing no tensor output reject", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const noop = kernel(function noop(a: Float32Tensor) {
  const i = gpu.globalId(0);
  const v = a[i];
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.artifacts, []);
  assert.ok(capabilityIds(result.diagnostics).includes("gpu.kernel.no-output"));
});

test("backend capability mismatches surface as compile diagnostics", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float64Tensor } from "@acme/tensor";

export const wide = kernel(function wide(a: Float64Tensor, out: Float64Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i];
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.artifacts, []);
  assert.ok(result.diagnostics.some((diagnostic) => diagnostic.code === "GPU_BACKEND_CAPABILITY_MISSING"));
  assert.ok(capabilityIds(result.diagnostics).includes("gpu.dtype.float64"));
});

test("kernel markers on non-function values fail closed", () => {
  const source = `import { kernel } from "@tsonic/gpu/lang.js";

export const bad = kernel(42);
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.artifacts, []);
  assert.ok(capabilityIds(result.diagnostics).includes("gpu.kernel.declaration-form"));
});
