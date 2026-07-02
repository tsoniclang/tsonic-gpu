import type { TargetArtifact } from "@tsonic/target-api";
import type { GpuHostArtifactRequest, GpuHostIntegration, GpuHostPackagingResult } from "../host-contract.js";

// The fake host integration exists so GPU core tests can prove the host
// artifact contract without any real host target installed. It accepts the
// backend's requested module paths verbatim and records dependencies and
// launch wrappers as one structured configuration artifact. The host target
// id is always the caller's choice; the GPU core has no default host.
export function createFakeGpuHostIntegration(hostTargetId: string): GpuHostIntegration {
  return {
    hostTargetId,
    packageArtifacts(request: GpuHostArtifactRequest): GpuHostPackagingResult {
      const artifacts: TargetArtifact[] = [
        ...request.modules.map((module) => ({
          kind: "source" as const,
          language: module.language,
          path: module.path,
          text: module.text,
        })),
        {
          kind: "configuration" as const,
          path: "gpu/launch-plan.json",
          text: `${JSON.stringify(
            {
              backend: request.backendId,
              hostTarget: request.hostTargetId,
              module: request.moduleName,
              dependencies: request.dependencies,
              launchWrappers: request.launchWrappers,
            },
            null,
            2,
          )}\n`,
        },
      ];
      return { artifacts, diagnostics: [] };
    },
  };
}
