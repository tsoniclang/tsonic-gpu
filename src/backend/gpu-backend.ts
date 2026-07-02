import type { TargetBackend, TargetCompileInput, TargetCompileResult } from "@tsonic/target-api";
import type { GpuBackendPlugin } from "../backends/backend-contract.js";
import { planGpuArtifacts } from "./planner/gpu-planner.js";

export function createGpuBackend(plugin: GpuBackendPlugin): TargetBackend {
  return {
    compile(input: TargetCompileInput): TargetCompileResult {
      return planGpuArtifacts(input, plugin);
    },
  };
}
