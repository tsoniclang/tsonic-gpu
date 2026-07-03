import { test } from "node:test";
import assert from "node:assert/strict";
import { createTargetRegistry } from "@tsonic/target-api";
import {
  createFakeGpuBackend,
  createFakeGpuHostIntegration,
  createTsonicPlugin,
  gpuTargetPluginId,
  matchGpuModuleAgainstCapabilities,
  readTsonicPluginManifest,
  validateGpuIrModule,
} from "../dist/index.js";
import { artifactText, compileGpu } from "./helpers/gpu-session.mjs";

// A second backend and a second host, defined only in tests: the GPU core
// must compose them exactly like any product backend/host, proving it is
// neither Triton-shaped nor tied to any single host target.

function echoBackend() {
  const capabilities = { ...createFakeGpuBackend().describeCapabilities(), backendId: "echo" };
  return {
    id: "echo",
    describeCapabilities: () => capabilities,
    validate: (module) => [...validateGpuIrModule(module), ...matchGpuModuleAgainstCapabilities(module, capabilities)],
    lower(module, context) {
      const kernels = [...module.kernels].sort((left, right) => left.name.localeCompare(right.name, "en"));
      return {
        modules: kernels.map((kernel) => ({
          path: `echo/${kernel.name}.txt`,
          language: "echo",
          text: `${context.hostTargetId}:${kernel.name}\n`,
        })),
        dependencies: [],
        launchWrappers: kernels.map((kernel) => ({ hostFunctionName: kernel.name, kernelName: kernel.name, metaParameters: [] })),
      };
    },
  };
}

function fakeBackendPlugin() {
  return { kind: "gpu-backend", id: "@fake/backend", backendId: "fake", createBackend: createFakeGpuBackend };
}

function echoBackendPlugin() {
  return { kind: "gpu-backend", id: "@fake/echo-backend", backendId: "echo", createBackend: echoBackend };
}

function hostPlugin(id, hostTargetId) {
  return { kind: "gpu-host", id, hostTargetId, createHostIntegration: () => createFakeGpuHostIntegration(hostTargetId) };
}

const scaleSource = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const scale = kernel(function scale(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i] * 2.0;
});
`;

test("the tsonic manifest follows the core plugin contract", () => {
  assert.deepEqual(readTsonicPluginManifest(), { kind: "plugin", contractVersion: 1, entry: "." });
});

test("package.json resolves through package exports for host discovery", async () => {
  const { createRequire } = await import("node:module");
  const requireFromHere = createRequire(import.meta.url);
  const resolved = requireFromHere.resolve("@tsonic/target-gpu/package.json");
  assert.ok(resolved.endsWith("package.json"));
});

test("createTsonicPlugin registers as an installed target plugin", () => {
  const plugin = createTsonicPlugin();
  assert.equal(plugin.kind, "target");
  assert.equal(plugin.id, gpuTargetPluginId);
  assert.equal(plugin.targetId, "gpu");
  const registry = createTargetRegistry([plugin.createTargetPack()]);
  assert.equal(registry.get("gpu").id, "gpu");
});

test("plugin-composed backends and hosts drive selection through target options", () => {
  const plugin = createTsonicPlugin({
    plugins: [fakeBackendPlugin(), echoBackendPlugin(), hostPlugin("@fake/python-host", "python"), hostPlugin("@fake/other-host", "otherhost")],
  });
  const pack = plugin.createTargetPack();
  const target = { id: "gpu", options: { backendId: "echo", hostTargetId: "otherhost" } };
  const { result } = compileGpu({ files: { "index.ts": scaleSource }, target, pack });
  assert.deepEqual(result.diagnostics, []);
  assert.equal(artifactText(result, "echo/scale.txt"), "otherhost:scale\n");
});

test("the same composed pack serves a second backend and host pair", () => {
  const plugin = createTsonicPlugin({
    plugins: [fakeBackendPlugin(), echoBackendPlugin(), hostPlugin("@fake/python-host", "python"), hostPlugin("@fake/other-host", "otherhost")],
  });
  const target = { id: "gpu", options: { backendId: "fake", hostTargetId: "python" } };
  const { result } = compileGpu({ files: { "index.ts": scaleSource }, target, pack: plugin.createTargetPack() });
  assert.deepEqual(result.diagnostics, []);
  const record = JSON.parse(artifactText(result, "kernels/scale.gpu-fake.json"));
  assert.equal(record.backend, "fake");
});

test("plugin entries with unknown kinds fail closed", () => {
  assert.throws(
    () => createTsonicPlugin({ plugins: [{ kind: "python-host", id: "@bad/plugin" }] }),
    /kind 'python-host'; the GPU target composes 'gpu-backend' and 'gpu-host' plugins/u,
  );
  assert.throws(() => createTsonicPlugin({ plugins: [42] }), /non-object plugin entry/u);
  assert.throws(() => createTsonicPlugin({ plugins: [{ kind: "gpu-backend" }] }), /non-empty package id/u);
});

test("declared ids must agree with created instances", () => {
  assert.throws(
    () =>
      createTsonicPlugin({
        plugins: [{ kind: "gpu-backend", id: "@bad/backend", backendId: "mismatched", createBackend: createFakeGpuBackend }],
      }),
    /declares backendId 'mismatched' but created a backend with id 'fake'/u,
  );
  assert.throws(
    () =>
      createTsonicPlugin({
        plugins: [
          { kind: "gpu-host", id: "@bad/host", hostTargetId: "python", createHostIntegration: () => createFakeGpuHostIntegration("otherhost") },
        ],
      }),
    /declares hostTargetId 'python' but created an integration for 'otherhost'/u,
  );
});

test("duplicate plugin, backend, and host ids fail closed", () => {
  assert.throws(
    () => createTsonicPlugin({ plugins: [fakeBackendPlugin(), fakeBackendPlugin()] }),
    /composed more than once/u,
  );
  assert.throws(
    () =>
      createTsonicPlugin({
        plugins: [fakeBackendPlugin(), { ...fakeBackendPlugin(), id: "@fake/backend-2" }],
      }),
    /backend id 'fake' is provided by both/u,
  );
  assert.throws(
    () => createTsonicPlugin({ plugins: [hostPlugin("@fake/h1", "python"), hostPlugin("@fake/h2", "python")] }),
    /host target 'python' is provided by both/u,
  );
});

test("a composed pack without a selected backend or host still fails closed", () => {
  const plugin = createTsonicPlugin({ plugins: [echoBackendPlugin(), hostPlugin("@fake/other-host", "otherhost")] });
  const pack = plugin.createTargetPack();
  const target = { id: "gpu", options: { backendId: "fake", hostTargetId: "otherhost" } };
  assert.throws(
    () => pack.createBackend({ project: { entryPoint: "src/index.ts", targets: [target] }, target }),
    /GPU backend 'fake' is not registered/u,
  );
  const hostTarget = { id: "gpu", options: { backendId: "echo", hostTargetId: "python" } };
  const { result } = compileGpu({ files: { "index.ts": scaleSource }, target: hostTarget, pack });
  assert.deepEqual(result.artifacts, []);
  assert.equal(result.diagnostics[0].code, "GPU_HOST_INTEGRATION_MISSING");
});
