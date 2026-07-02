import { test } from "node:test";
import assert from "node:assert/strict";
import {
  createFakeGpuBackend,
  matchGpuModuleAgainstCapabilities,
  validateGpuIrModule,
} from "../dist/index.js";
import { artifactText, capabilityIds, compileGpu } from "./helpers/gpu-session.mjs";

const geluSource = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const geluApprox = kernel(function geluApprox(x: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  const v = x[i];
  out[i] = 0.5 * v * (1.0 + gpu.tanh(0.79788456 * (v + 0.044715 * v * v * v)));
});
`;

const reductionSource = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const total = kernel(function total(values: Float32Tensor, out: Float32Tensor) {
  const partial = gpu.blockReduceSum(values);
  const lane = gpu.localId(0);
  if (lane === 0) {
    out[0] = partial;
  }
});
`;

// A second test backend proving the accept direction of reduction
// negotiation without widening the deliberately narrow fake backend.
function reducingBackend() {
  const fakeCapabilities = createFakeGpuBackend().describeCapabilities();
  const capabilities = {
    backendId: "reducing",
    maxTensorRank: fakeCapabilities.maxTensorRank,
    capabilityIds: [...fakeCapabilities.capabilityIds, "gpu.reduce.sum.float32", "gpu.reduce.max.float32"],
  };
  return {
    id: "reducing",
    describeCapabilities() {
      return capabilities;
    },
    validate(module) {
      return [...validateGpuIrModule(module), ...matchGpuModuleAgainstCapabilities(module, capabilities)];
    },
    lower(module, context) {
      const kernels = [...module.kernels].sort((left, right) => left.name.localeCompare(right.name, "en"));
      return {
        modules: kernels.map((kernel) => ({
          path: `kernels/${kernel.name}.gpu-reducing.json`,
          language: "gpu-reducing-module",
          text: `${JSON.stringify({ backend: "reducing", hostTarget: context.hostTargetId, kernel: kernel.name })}\n`,
        })),
        dependencies: [],
        launchWrappers: kernels.map((kernel) => ({
          hostFunctionName: kernel.name,
          kernelName: kernel.name,
          metaParameters: [],
        })),
      };
    },
  };
}

const matmulSource = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Matrix } from "@acme/tensor";

export const matmul = kernel(function matmul<M extends number, K extends number, N extends number>(
  a: Matrix<M, K>,
  b: Matrix<K, N>,
  c: Matrix<M, N>,
) {
  const row = gpu.globalId(0);
  const col = gpu.globalId(1);
  let acc = 0.0;
  const kDim = gpu.dim(a, 1);
  for (let k = 0; k < kDim; k++) {
    acc += a.at(row, k) * b.at(k, col);
  }
  c.set(row, col, acc);
});
`;

test("matmul with shared shape symbols compiles through the fake backend", () => {
  const { result } = compileGpu({ files: { "index.ts": matmulSource } });
  assert.deepEqual(result.diagnostics, []);
  const record = JSON.parse(artifactText(result, "kernels/matmul.gpu-fake.json"));
  assert.equal(record.kernel, "matmul");
  assert.deepEqual(
    record.parameters.map((parameter) => [parameter.name, parameter.role, parameter.rank]),
    [
      ["a", "input", 2],
      ["b", "input", 2],
      ["c", "output", 2],
    ],
  );
  assert.equal(record.launch.gridDimensions, 2);
});

test("launch meta parameters flow into launch wrappers", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const tiled = kernel(function tiled(a: Float32Tensor, out: Float32Tensor) {
  const block = gpu.meta("BLOCK");
  out[block] = a[block];
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assert.deepEqual(result.diagnostics, []);
  const record = JSON.parse(artifactText(result, "kernels/tiled.gpu-fake.json"));
  assert.deepEqual(record.launch.metaParameters, ["BLOCK"]);
  const launchPlan = JSON.parse(artifactText(result, "gpu/launch-plan.json"));
  assert.deepEqual(launchPlan.launchWrappers, [{ hostFunctionName: "tiled", kernelName: "tiled", metaParameters: ["BLOCK"] }]);
});

test("the gelu approximation example compiles through the fake backend", () => {
  const { result } = compileGpu({ files: { "index.ts": geluSource } });
  assert.deepEqual(result.diagnostics, []);
  const record = JSON.parse(artifactText(result, "kernels/geluApprox.gpu-fake.json"));
  assert.equal(record.kernel, "geluApprox");
  assert.deepEqual(
    record.parameters.map((parameter) => [parameter.name, parameter.role]),
    [
      ["x", "input"],
      ["out", "output"],
    ],
  );
});

test("block reductions are rejected end-to-end by a backend without reduction capability", () => {
  const { result } = compileGpu({ files: { "index.ts": reductionSource } });
  assert.deepEqual(result.artifacts, []);
  assert.ok(result.diagnostics.some((diagnostic) => diagnostic.code === "GPU_BACKEND_CAPABILITY_MISSING"));
  assert.ok(capabilityIds(result.diagnostics).includes("gpu.reduce.sum.float32"));
});

test("block reductions compile end-to-end through a backend that reports reduction capability", () => {
  const target = { id: "gpu", options: { backendId: "reducing", hostTargetId: "python" } };
  const { result } = compileGpu({
    files: { "index.ts": reductionSource },
    target,
    backends: [createFakeGpuBackend(), reducingBackend()],
  });
  assert.deepEqual(result.diagnostics, []);
  const record = JSON.parse(artifactText(result, "kernels/total.gpu-reducing.json"));
  assert.equal(record.kernel, "total");
  const launchPlan = JSON.parse(artifactText(result, "gpu/launch-plan.json"));
  assert.equal(launchPlan.backend, "reducing");
});
