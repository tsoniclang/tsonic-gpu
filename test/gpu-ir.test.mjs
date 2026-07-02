import { test } from "node:test";
import assert from "node:assert/strict";
import { validateGpuIrModule } from "../dist/index.js";
import { reductionModule, singleKernelModule, tensorParameter, vectorAddModule } from "./helpers/ir-fixtures.mjs";

function capabilityIds(diagnostics) {
  return diagnostics.flatMap((diagnostic) =>
    diagnostic.evidence.filter((row) => row.startsWith("target.capability=")).map((row) => row.slice("target.capability=".length)),
  );
}

test("hand-authored vector add IR validates", () => {
  assert.deepEqual(validateGpuIrModule(vectorAddModule()), []);
});

test("reduction IR validates structurally", () => {
  assert.deepEqual(validateGpuIrModule(reductionModule()), []);
});

test("conditional and counted loop IR validates", () => {
  const module = singleKernelModule({
    body: {
      operations: [
        { kind: "thread-index", result: "i", space: "global", dimension: 0 },
        { kind: "const", result: "start", dtype: "int32", value: 0 },
        { kind: "const", result: "limit", dtype: "int32", value: 16 },
        {
          kind: "loop",
          counter: "k",
          lowerBound: "start",
          upperBound: "limit",
          body: {
            operations: [
              { kind: "load", result: "value", tensor: "a", indices: ["k"], dtype: "float32" },
              { kind: "binary", result: "positive", operator: "gt", left: "value", right: "start", dtype: "bool" },
              {
                kind: "if",
                condition: "positive",
                then: { operations: [{ kind: "store", tensor: "out", indices: ["k"], value: "value" }] },
              },
            ],
          },
        },
      ],
    },
  });
  assert.deepEqual(validateGpuIrModule(module), []);
});

test("unknown tensor dtype rejects", () => {
  const module = singleKernelModule({
    parameters: [tensorParameter("a", "input", { dtype: "float128" }), tensorParameter("out", "output")],
    body: { operations: [] },
    effects: [],
  });
  const diagnostics = validateGpuIrModule(module);
  assert.ok(diagnostics.some((diagnostic) => diagnostic.code === "GPU_INVALID_IR"));
  assert.ok(capabilityIds(diagnostics).includes("gpu.ir.tensor.dtype"));
});

test("rank and shape mismatch rejects", () => {
  const module = singleKernelModule({
    parameters: [
      tensorParameter("a", "input", { rank: 2, shape: [{ kind: "symbol", name: "N" }] }),
      tensorParameter("out", "output"),
    ],
    body: { operations: [] },
    effects: [],
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.tensor.shape"));
});

test("non-positive literal dimension rejects", () => {
  const module = singleKernelModule({
    parameters: [
      tensorParameter("a", "input", { shape: [{ kind: "literal", value: 0 }] }),
      tensorParameter("out", "output"),
    ],
    body: { operations: [] },
    effects: [],
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.tensor.shape"));
});

test("mixed device domains reject before backend lowering", () => {
  const module = singleKernelModule({
    parameters: [
      tensorParameter("a", "input", { device: "cuda" }),
      tensorParameter("b", "input", { device: "cpu" }),
      tensorParameter("out", "output", { device: "cuda" }),
    ],
    body: { operations: [] },
    effects: [],
  });
  const diagnostics = validateGpuIrModule(module);
  assert.ok(diagnostics.some((diagnostic) => diagnostic.code === "GPU_DEVICE_MISMATCH"));
  assert.ok(capabilityIds(diagnostics).includes("gpu.device.mixed"));
});

test("undefined value reference rejects", () => {
  const module = singleKernelModule({
    body: {
      operations: [{ kind: "store", tensor: "out", indices: ["missing"], value: "alsoMissing" }],
    },
  });
  const diagnostics = validateGpuIrModule(module);
  assert.ok(capabilityIds(diagnostics).includes("gpu.ir.value-ref"));
});

test("store to readonly tensor rejects", () => {
  const module = singleKernelModule({
    body: {
      operations: [
        { kind: "thread-index", result: "i", space: "global", dimension: 0 },
        { kind: "load", result: "value", tensor: "a", indices: ["i"], dtype: "float32" },
        { kind: "store", tensor: "a", indices: ["i"], value: "value" },
      ],
    },
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.store.mutability"));
});

test("input-role tensors must be readonly", () => {
  const module = singleKernelModule({
    parameters: [tensorParameter("a", "input", { mutability: "mutable" }), tensorParameter("out", "output")],
    body: { operations: [] },
    effects: [],
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.tensor.mutability"));
});

test("index arity must match tensor rank", () => {
  const module = singleKernelModule({
    body: {
      operations: [
        { kind: "thread-index", result: "i", space: "global", dimension: 0 },
        { kind: "load", result: "value", tensor: "a", indices: ["i", "i"], dtype: "float32" },
      ],
    },
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.index-arity"));
});

test("load dtype must match tensor element dtype", () => {
  const module = singleKernelModule({
    body: {
      operations: [
        { kind: "thread-index", result: "i", space: "global", dimension: 0 },
        { kind: "load", result: "value", tensor: "a", indices: ["i"], dtype: "int32" },
      ],
    },
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.load.dtype"));
});

test("duplicate value definitions reject", () => {
  const module = singleKernelModule({
    body: {
      operations: [
        { kind: "thread-index", result: "i", space: "global", dimension: 0 },
        { kind: "thread-index", result: "i", space: "global", dimension: 0 },
      ],
    },
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.value-def"));
});

test("launch plan must have one to three grid dimensions", () => {
  const module = singleKernelModule({
    launch: { grid: [], streamPolicy: "default", devicePolicy: "single-device" },
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.launch.grid"));
});

test("launch plan symbols must be known", () => {
  const module = singleKernelModule({
    launch: {
      grid: [{ kind: "symbol", name: "UNKNOWN_DIM" }],
      streamPolicy: "default",
      devicePolicy: "single-device",
    },
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.launch.symbol"));
});

test("backend meta parameters in launch expressions must be declared", () => {
  const module = singleKernelModule({
    launch: {
      grid: [{ kind: "backend-meta", name: "BLOCK" }],
      streamPolicy: "default",
      devicePolicy: "single-device",
    },
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.launch.meta"));
});

test("effects must reference mutable tensors for writes", () => {
  const module = singleKernelModule({
    effects: [{ kind: "write", parameter: "a" }],
    body: { operations: [] },
  });
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.effect"));
});

test("duplicate kernel names reject", () => {
  const base = vectorAddModule();
  const module = { ...base, kernels: [base.kernels[0], base.kernels[0]] };
  assert.ok(capabilityIds(validateGpuIrModule(module)).includes("gpu.ir.kernel-name"));
});
