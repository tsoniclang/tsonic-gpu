import { test } from "node:test";
import assert from "node:assert/strict";
import {
  createFakeGpuBackend,
  matchGpuModuleAgainstCapabilities,
  requiredCapabilitiesForKernel,
} from "../dist/index.js";
import { reductionModule, singleKernelModule, tensorParameter, vectorAddModule } from "./helpers/ir-fixtures.mjs";

function missingCapabilityIds(diagnostics) {
  return diagnostics
    .filter((diagnostic) => diagnostic.code === "GPU_BACKEND_CAPABILITY_MISSING")
    .flatMap((diagnostic) =>
      diagnostic.evidence.filter((row) => row.startsWith("target.capability=")).map((row) => row.slice("target.capability=".length)),
    );
}

test("required capabilities are derived from the IR", () => {
  const [kernel] = vectorAddModule().kernels;
  const required = requiredCapabilitiesForKernel(kernel);
  assert.ok(required.includes("gpu.dtype.float32"));
  assert.ok(required.includes("gpu.device.cuda"));
  assert.ok(required.includes("gpu.layout.contiguous"));
  assert.ok(required.includes("gpu.thread-index.global"));
  assert.ok(required.includes("gpu.op.binary.add"));
  assert.ok(required.includes("gpu.memory.load"));
  assert.ok(required.includes("gpu.memory.store"));
  assert.ok(required.includes("gpu.memory.masked"));
  assert.ok(required.includes("gpu.shape.symbolic"));
  assert.deepEqual(required, [...required].sort());
});

test("fake backend accepts the vector add module", () => {
  assert.deepEqual(createFakeGpuBackend().validate(vectorAddModule()), []);
});

test("reduction is rejected by a backend without reduction capability", () => {
  const diagnostics = createFakeGpuBackend().validate(reductionModule());
  assert.ok(missingCapabilityIds(diagnostics).includes("gpu.reduce.sum.float32"));
  assert.ok(diagnostics.every((diagnostic) => diagnostic.evidence.includes("gpu.backend=fake")));
});

test("atomics are rejected when the backend lacks atomic capability", () => {
  const module = singleKernelModule({
    body: {
      operations: [
        { kind: "thread-index", result: "i", space: "global", dimension: 0 },
        { kind: "load", result: "value", tensor: "a", indices: ["i"], dtype: "float32" },
        { kind: "atomic", operator: "add", tensor: "out", indices: ["i"], value: "value", dtype: "float32" },
      ],
    },
  });
  const diagnostics = createFakeGpuBackend().validate(module);
  assert.ok(missingCapabilityIds(diagnostics).includes("gpu.atomic.add.float32"));
});

test("barriers are rejected when the backend lacks barrier capability", () => {
  const module = singleKernelModule({
    body: { operations: [{ kind: "barrier", scope: "block" }] },
  });
  const diagnostics = createFakeGpuBackend().validate(module);
  assert.ok(missingCapabilityIds(diagnostics).includes("gpu.barrier.block"));
});

test("unsupported dtype is diagnosed before lowering", () => {
  const module = singleKernelModule({
    parameters: [
      tensorParameter("a", "input", { dtype: "float64" }),
      tensorParameter("out", "output", { dtype: "float64" }),
    ],
    body: { operations: [] },
    effects: [],
  });
  const diagnostics = createFakeGpuBackend().validate(module);
  assert.ok(missingCapabilityIds(diagnostics).includes("gpu.dtype.float64"));
});

test("tensor rank above the backend maximum is diagnosed", () => {
  const module = singleKernelModule({
    parameters: [
      tensorParameter("a", "input", {
        rank: 3,
        shape: [
          { kind: "symbol", name: "N" },
          { kind: "symbol", name: "N" },
          { kind: "symbol", name: "N" },
        ],
      }),
      tensorParameter("out", "output"),
    ],
    body: { operations: [] },
    effects: [],
  });
  const diagnostics = createFakeGpuBackend().validate(module);
  assert.ok(missingCapabilityIds(diagnostics).includes("gpu.tensor.rank.3"));
});

test("reductions are accepted by a backend that reports reduction capability", () => {
  const module = reductionModule();
  const [kernel] = module.kernels;
  const withReduce = {
    backendId: "reducing",
    maxTensorRank: 2,
    capabilityIds: requiredCapabilitiesForKernel(kernel),
  };
  assert.deepEqual(matchGpuModuleAgainstCapabilities(module, withReduce), []);
  const withoutReduce = {
    ...withReduce,
    capabilityIds: withReduce.capabilityIds.filter((id) => !id.startsWith("gpu.reduce.")),
  };
  const diagnostics = matchGpuModuleAgainstCapabilities(module, withoutReduce);
  assert.ok(missingCapabilityIds(diagnostics).includes("gpu.reduce.sum.float32"));
});

test("capability matching is honest about custom capability sets", () => {
  const module = vectorAddModule();
  const withoutMasked = {
    backendId: "narrow",
    maxTensorRank: 2,
    capabilityIds: requiredCapabilitiesForKernel(module.kernels[0]).filter((id) => id !== "gpu.memory.masked"),
  };
  const diagnostics = matchGpuModuleAgainstCapabilities(module, withoutMasked);
  assert.equal(diagnostics.length, 1);
  assert.ok(missingCapabilityIds(diagnostics).includes("gpu.memory.masked"));
  assert.ok(diagnostics[0].message.includes("narrow"));
});
