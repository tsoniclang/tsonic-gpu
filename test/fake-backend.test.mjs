import { test } from "node:test";
import assert from "node:assert/strict";
import { createFakeGpuBackend, fakeGpuBackendId } from "../dist/index.js";
import { reductionModule, singleKernelModule, vectorAddModule } from "./helpers/ir-fixtures.mjs";

const hostContext = { hostTargetId: "python" };

test("fake backend describes a deterministic capability set", () => {
  const backend = createFakeGpuBackend();
  const capabilities = backend.describeCapabilities();
  assert.equal(capabilities.backendId, fakeGpuBackendId);
  assert.equal(capabilities.maxTensorRank, 2);
  assert.deepEqual(capabilities.capabilityIds, createFakeGpuBackend().describeCapabilities().capabilityIds);
  const reductionRows = capabilities.capabilityIds.filter((id) => id.startsWith("gpu.reduce.") || id.startsWith("gpu.atomic.") || id.startsWith("gpu.barrier."));
  assert.deepEqual(reductionRows, [], "the fake backend must stay narrow so rejection tests stay meaningful");
});

test("fake backend lowers a valid module into structured artifacts", () => {
  const artifacts = createFakeGpuBackend().lower(vectorAddModule(), hostContext);
  assert.equal(artifacts.modules.length, 1);
  const [module] = artifacts.modules;
  assert.equal(module.path, "kernels/add.gpu-fake.json");
  assert.equal(module.language, "gpu-fake-module");
  const record = JSON.parse(module.text);
  assert.equal(record.backend, fakeGpuBackendId);
  assert.equal(record.hostTarget, "python");
  assert.equal(record.kernel, "add");
  assert.equal(record.parameters.length, 3);
  assert.equal(record.launch.gridDimensions, 1);
  assert.equal(record.operationCount, 7);
  assert.deepEqual(artifacts.dependencies, []);
  assert.deepEqual(artifacts.launchWrappers, [{ hostFunctionName: "add", kernelName: "add", metaParameters: [] }]);
});

test("fake backend artifact emission is deterministic and name-sorted", () => {
  const base = vectorAddModule();
  const second = { ...base.kernels[0], name: "aardvark" };
  const module = { ...base, kernels: [base.kernels[0], second] };
  const first = createFakeGpuBackend().lower(module, hostContext);
  const again = createFakeGpuBackend().lower(module, hostContext);
  assert.deepEqual(first, again);
  assert.deepEqual(
    first.modules.map((artifact) => artifact.path),
    ["kernels/aardvark.gpu-fake.json", "kernels/add.gpu-fake.json"],
  );
});

test("fake backend refuses to lower a module that fails validation", () => {
  const backend = createFakeGpuBackend();
  assert.throws(() => backend.lower(reductionModule(), hostContext), /cannot lower module/u);
  const structurallyInvalid = singleKernelModule({
    body: { operations: [{ kind: "store", tensor: "nope", indices: [], value: "missing" }] },
  });
  assert.throws(() => backend.lower(structurallyInvalid, hostContext), /cannot lower module/u);
});
