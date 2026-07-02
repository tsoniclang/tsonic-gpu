import type { TargetSelection } from "@tsonic/target-api";

const supportedGpuTargetOptionKeys = Object.freeze([
  "backendId",
  "backendPackageName",
  "hostTargetId",
  "typescriptCompatibility",
]);

const gpuIdentifierPattern = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/u;

export function validateGpuTargetOptions(target: TargetSelection): void {
  const options = target.options;
  if (options !== undefined) {
    const allowedKeys = new Set(supportedGpuTargetOptionKeys);
    for (const key of Object.keys(options)) {
      if (!allowedKeys.has(key)) {
        throw new Error(`GPU target option 'options.${key}' is not supported.`);
      }
    }
  }
  readGpuBackendId(target);
  readGpuBackendPackageName(target);
  readGpuHostTargetId(target);
  readGpuTypescriptCompatibilityMode(target);
}

export function readGpuBackendId(target: TargetSelection): string {
  const value = readOptionalStringOption(target, "backendId");
  if (value === undefined) {
    throw new Error("GPU target option 'backendId' is required; GPU backends are selected explicitly and are never inferred.");
  }
  if (!gpuIdentifierPattern.test(value)) {
    throw new Error(`GPU target option 'backendId' must match ${gpuIdentifierPattern.source}.`);
  }
  return value;
}

export function readGpuBackendPackageName(target: TargetSelection): string | undefined {
  return readOptionalStringOption(target, "backendPackageName");
}

export function readGpuHostTargetId(target: TargetSelection): string {
  const value = readOptionalStringOption(target, "hostTargetId");
  if (value === undefined) {
    throw new Error("GPU target option 'hostTargetId' is required; GPU kernels always compile for an explicit host target.");
  }
  if (!gpuIdentifierPattern.test(value)) {
    throw new Error(`GPU target option 'hostTargetId' must match ${gpuIdentifierPattern.source}.`);
  }
  return value;
}

export function readGpuTypescriptCompatibilityMode(target: TargetSelection): "strict-native" {
  const value = target.options?.typescriptCompatibility;
  if (value === undefined || value === "strict-native") {
    return "strict-native";
  }
  throw new Error("GPU target option 'typescriptCompatibility' only supports 'strict-native'; GPU kernels have no compatibility lane.");
}

function readOptionalStringOption(target: TargetSelection, key: string): string | undefined {
  const value = target.options?.[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`GPU target option '${key}' must be a non-empty string.`);
  }
  return value;
}
