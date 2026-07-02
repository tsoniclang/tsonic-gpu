import type { TargetBackend, TargetCompileInput, TargetCompileResult } from "@tsonic/target-api";
import type { GpuBackendPlugin } from "../backends/backend-contract.js";
import type { GpuHostRegistry } from "../hosts/host-registry.js";
import { planGpuArtifacts } from "./planner/gpu-planner.js";

export function createGpuBackend(plugin: GpuBackendPlugin, hosts?: GpuHostRegistry): TargetBackend {
  return {
    compile(input: TargetCompileInput): TargetCompileResult {
      return planGpuArtifacts(input, plugin, hosts);
    },
  };
}
