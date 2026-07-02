import { test } from "node:test";
import assert from "node:assert/strict";
import { createFakeGpuHostIntegration, createGpuHostRegistry } from "../dist/index.js";
import { artifactText, capabilityIds, compileGpu } from "./helpers/gpu-session.mjs";

const kernelSource = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const scale = kernel(function scale(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i] * 2.0;
});
`;

test("backend artifacts are packaged by the selected host integration", () => {
  const { result } = compileGpu({ files: { "index.ts": kernelSource } });
  assert.deepEqual(result.diagnostics, []);
  const launchPlan = JSON.parse(artifactText(result, "gpu/launch-plan.json"));
  assert.equal(launchPlan.hostTarget, "python");
  assert.equal(launchPlan.module, "gpu_kernels");
  assert.deepEqual(launchPlan.launchWrappers, [{ hostFunctionName: "scale", kernelName: "scale", metaParameters: [] }]);
});

test("missing host integration reports a deterministic error", () => {
  const { result } = compileGpu({ files: { "index.ts": kernelSource }, hosts: [] });
  assert.deepEqual(result.artifacts, []);
  assert.equal(result.diagnostics.length, 1);
  const diagnostic = result.diagnostics[0];
  assert.equal(diagnostic.code, "GPU_HOST_INTEGRATION_MISSING");
  assert.ok(capabilityIds(result.diagnostics).includes("gpu.host.integration"));
  assert.ok(diagnostic.evidence.includes("gpu.hostTarget=python"));
  assert.ok(diagnostic.evidence.includes("gpu.hosts.registered=(none)"));
});

test("host integration selection follows the explicit hostTargetId option", () => {
  const target = { id: "gpu", options: { backendId: "fake", hostTargetId: "otherhost" } };
  const { result } = compileGpu({
    files: { "index.ts": kernelSource },
    target,
    hosts: [createFakeGpuHostIntegration("otherhost")],
  });
  assert.deepEqual(result.diagnostics, []);
  const launchPlan = JSON.parse(artifactText(result, "gpu/launch-plan.json"));
  assert.equal(launchPlan.hostTarget, "otherhost");
});

test("every emitted artifact path is decided by the host integration", () => {
  const prefixed = {
    hostTargetId: "python",
    packageArtifacts(request) {
      return {
        artifacts: request.modules.map((module) => ({
          kind: "source",
          language: module.language,
          path: `hostchoice/${module.path}`,
          text: module.text,
        })),
        diagnostics: [],
      };
    },
  };
  const { result } = compileGpu({ files: { "index.ts": kernelSource }, hosts: [prefixed] });
  assert.deepEqual(result.diagnostics, []);
  assert.ok(result.artifacts.length > 0);
  assert.ok(result.artifacts.every((artifact) => artifact.path.startsWith("hostchoice/")));
});

test("host packaging diagnostics fail the compile without artifacts", () => {
  const failing = {
    hostTargetId: "python",
    packageArtifacts() {
      return {
        artifacts: [],
        diagnostics: [
          {
            code: "GPU_HOST_PACKAGING_REJECTED",
            category: "error",
            source: "fake-host",
            message: "The host integration rejected the artifact request.",
            evidence: ["target.capability=gpu.host.packaging"],
          },
        ],
      };
    },
  };
  const { result } = compileGpu({ files: { "index.ts": kernelSource }, hosts: [failing] });
  assert.deepEqual(result.artifacts, []);
  assert.equal(result.diagnostics.length, 1);
  assert.equal(result.diagnostics[0].code, "GPU_HOST_PACKAGING_REJECTED");
});

test("duplicate host integrations are rejected", () => {
  assert.throws(
    () => createGpuHostRegistry([createFakeGpuHostIntegration("python"), createFakeGpuHostIntegration("python")]),
    /registered more than once/u,
  );
});
