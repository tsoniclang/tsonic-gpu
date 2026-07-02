import { test } from "node:test";
import assert from "node:assert/strict";
import {
  readGpuBackendId,
  readGpuBackendPackageName,
  readGpuHostTargetId,
  readGpuTypescriptCompatibilityMode,
  validateGpuTargetOptions,
} from "../dist/index.js";

function target(options) {
  return { id: "gpu", options };
}

test("valid options pass validation", () => {
  validateGpuTargetOptions(target({ backendId: "fake", hostTargetId: "python" }));
  validateGpuTargetOptions(
    target({
      backendId: "triton-like",
      backendPackageName: "@acme/gpu-backend",
      hostTargetId: "python",
      typescriptCompatibility: "strict-native",
    }),
  );
});

test("backendId is required and validated", () => {
  assert.throws(() => readGpuBackendId(target(undefined)), /'backendId' is required/u);
  assert.throws(() => readGpuBackendId(target({ hostTargetId: "python" })), /'backendId' is required/u);
  assert.throws(() => readGpuBackendId(target({ backendId: "" })), /non-empty string/u);
  assert.throws(() => readGpuBackendId(target({ backendId: 42 })), /non-empty string/u);
  assert.throws(() => readGpuBackendId(target({ backendId: "Not_Valid" })), /must match/u);
  assert.equal(readGpuBackendId(target({ backendId: "fake" })), "fake");
});

test("hostTargetId is required and validated", () => {
  assert.throws(() => readGpuHostTargetId(target({ backendId: "fake" })), /'hostTargetId' is required/u);
  assert.throws(() => readGpuHostTargetId(target({ hostTargetId: "PYTHON" })), /must match/u);
  assert.equal(readGpuHostTargetId(target({ hostTargetId: "python" })), "python");
});

test("backendPackageName is optional but must be a non-empty string", () => {
  assert.equal(readGpuBackendPackageName(target({})), undefined);
  assert.equal(readGpuBackendPackageName(target({ backendPackageName: "@acme/backend" })), "@acme/backend");
  assert.throws(() => readGpuBackendPackageName(target({ backendPackageName: "" })), /non-empty string/u);
});

test("typescriptCompatibility only supports strict-native", () => {
  assert.equal(readGpuTypescriptCompatibilityMode(target({})), "strict-native");
  assert.equal(readGpuTypescriptCompatibilityMode(target({ typescriptCompatibility: "strict-native" })), "strict-native");
  assert.throws(() => readGpuTypescriptCompatibilityMode(target({ typescriptCompatibility: "compat" })), /strict-native/u);
});

test("unknown option keys are rejected", () => {
  assert.throws(
    () => validateGpuTargetOptions(target({ backendId: "fake", hostTargetId: "python", blockSize: 64 })),
    /'options.blockSize' is not supported/u,
  );
});
