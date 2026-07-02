import { test } from "node:test";
import assert from "node:assert/strict";
import { createTargetRegistry } from "@tsonic/target-api";
import {
  createFakeGpuBackend,
  createGpuTargetPack,
  gpuTargetId,
  gpuTargetSemanticsExtensionId,
} from "../dist/index.js";

const validOptions = { backendId: "fake", hostTargetId: "python" };

function providerContext(pack, target) {
  return {
    project: { entryPoint: "src/index.ts", targets: [target] },
    target,
    targetPack: pack,
    selectedPackages: [],
    selectedSurfaces: [],
  };
}

test("GPU target pack registers with the target registry", () => {
  const registry = createTargetRegistry([createGpuTargetPack()]);
  const pack = registry.get(gpuTargetId);
  assert.notEqual(pack, undefined);
  assert.equal(pack.id, "gpu");
  assert.equal(pack.displayName, "GPU");
  assert.notEqual(pack.provider, undefined);
});

test("provider composes the GPU target semantics extension", () => {
  const pack = createGpuTargetPack();
  const target = { id: gpuTargetId, options: validOptions };
  const extensions = pack.provider.createExtensions(providerContext(pack, target));
  assert.equal(extensions.length, 1);
  assert.equal(extensions[0].identity.id, gpuTargetSemanticsExtensionId);
  assert.equal(extensions[0].composition.target, gpuTargetId);
});

test("missing backend selection fails at every pack entry point", () => {
  const pack = createGpuTargetPack({ backends: [createFakeGpuBackend()] });
  const target = { id: gpuTargetId, options: { hostTargetId: "python" } };
  assert.throws(() => pack.provider.createExtensions(providerContext(pack, target)), /'backendId' is required/u);
  assert.throws(() => pack.createBackend({ project: { entryPoint: "src/index.ts", targets: [target] }, target }), /'backendId' is required/u);
  assert.throws(() => pack.createToolchain({ project: { entryPoint: "src/index.ts", targets: [target] }, target }), /'backendId' is required/u);
});

test("unknown backend id fails backend creation", () => {
  const pack = createGpuTargetPack({ backends: [createFakeGpuBackend()] });
  const target = { id: gpuTargetId, options: { backendId: "unregistered-backend", hostTargetId: "python" } };
  assert.throws(
    () => pack.createBackend({ project: { entryPoint: "src/index.ts", targets: [target] }, target }),
    /GPU backend 'unregistered-backend' is not registered/u,
  );
});

test("registered fake backend produces a target backend", () => {
  const pack = createGpuTargetPack({ backends: [createFakeGpuBackend()] });
  const target = { id: gpuTargetId, options: validOptions };
  const backend = pack.createBackend({ project: { entryPoint: "src/index.ts", targets: [target] }, target });
  assert.equal(typeof backend.compile, "function");
});

test("unknown target options are rejected", () => {
  const pack = createGpuTargetPack();
  const target = { id: gpuTargetId, options: { ...validOptions, tritonBlockSize: 64 } };
  assert.throws(() => pack.provider.createExtensions(providerContext(pack, target)), /not supported/u);
});

test("duplicate backend registration is rejected", async () => {
  const { createGpuBackendRegistry } = await import("../dist/index.js");
  assert.throws(
    () => createGpuBackendRegistry([createFakeGpuBackend(), createFakeGpuBackend()]),
    /registered more than once/u,
  );
});
