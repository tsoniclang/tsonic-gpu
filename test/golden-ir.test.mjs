import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createFakeGpuBackend } from "../dist/index.js";
import { compileGpu } from "./helpers/gpu-session.mjs";

// Golden IR tests: the exact module handed to the backend for each reference
// kernel, spans stripped. Goldens under test/golden/ are reviewed by hand;
// regenerate only after verifying the new IR is correct, never to make a
// failing test pass.

const goldenRoot = join(dirname(fileURLToPath(import.meta.url)), "golden");

function captureBackend(store) {
  const fake = createFakeGpuBackend();
  return {
    id: "capture",
    describeCapabilities: () => ({ ...fake.describeCapabilities(), backendId: "capture" }),
    validate: () => [],
    lower(module) {
      store.module = module;
      return { modules: [], dependencies: [], launchWrappers: [] };
    },
  };
}

function stripSpans(value) {
  if (Array.isArray(value)) {
    return value.map(stripSpans);
  }
  if (value !== null && typeof value === "object") {
    const result = {};
    for (const [key, entry] of Object.entries(value)) {
      if (key === "span") {
        continue;
      }
      result[key] = stripSpans(entry);
    }
    return result;
  }
  return value;
}

function extractModule(source) {
  const store = {};
  const target = { id: "gpu", options: { backendId: "capture", hostTargetId: "python" } };
  const { result } = compileGpu({ files: { "index.ts": source }, target, backends: [captureBackend(store)] });
  assert.deepEqual(result.diagnostics, []);
  assert.notEqual(store.module, undefined);
  return stripSpans(store.module);
}

function golden(name) {
  return JSON.parse(readFileSync(join(goldenRoot, `${name}.json`), "utf8"));
}

test("golden IR: vector add", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";
import type { int32 } from "@tsonic/core/types.js";

export const add = kernel(function add(a: Float32Tensor, b: Float32Tensor, out: Float32Tensor, n: int32) {
  const i = gpu.globalId(0);
  if (i < n) {
    out[i] = a[i] + b[i];
  }
});
`;
  assert.deepEqual(extractModule(source), golden("vector-add"));
});

test("golden IR: gelu approximation", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const geluApprox = kernel(function geluApprox(x: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  const v = x[i];
  out[i] = 0.5 * v * (1.0 + gpu.tanh(0.79788456 * (v + 0.044715 * v * v * v)));
});
`;
  assert.deepEqual(extractModule(source), golden("gelu"));
});

test("golden IR: block reduction", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const total = kernel(function total(values: Float32Tensor, out: Float32Tensor) {
  const partial = gpu.blockReduceSum(values);
  const lane = gpu.localId(0);
  if (lane === 0) {
    out[0] = partial;
  }
});
`;
  assert.deepEqual(extractModule(source), golden("reduction"));
});

test("golden IR: matmul with shared shape symbols and loop-carried accumulator", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
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
  const module = extractModule(source);
  assert.deepEqual(module, golden("matmul"));
  const [matmul] = module.kernels;
  const shapes = matmul.parameters.map((parameter) => parameter.tensor.shape.map((dimension) => dimension.name));
  assert.deepEqual(shapes, [
    ["M", "K"],
    ["K", "N"],
    ["M", "N"],
  ]);
  assert.deepEqual(
    matmul.launch.grid.map((dimension) => dimension.name),
    ["M", "N"],
  );
});
