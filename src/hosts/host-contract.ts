import type { TargetArtifact, TargetDiagnostic } from "@tsonic/target-api";
import type {
  GpuBackendDependency,
  GpuBackendModuleArtifact,
  GpuLaunchWrapperRequest,
} from "../backends/backend-contract.js";

// The GPU core never decides host project layout. It hands the selected host
// integration a complete artifact request; the integration owns every path.

export interface GpuHostArtifactRequest {
  readonly backendId: string;
  readonly hostTargetId: string;
  readonly moduleName: string;
  readonly modules: readonly GpuBackendModuleArtifact[];
  readonly dependencies: readonly GpuBackendDependency[];
  readonly launchWrappers: readonly GpuLaunchWrapperRequest[];
}

export interface GpuHostPackagingResult {
  readonly artifacts: readonly TargetArtifact[];
  readonly diagnostics: readonly TargetDiagnostic[];
}

export interface GpuHostIntegration {
  readonly hostTargetId: string;
  packageArtifacts(request: GpuHostArtifactRequest): GpuHostPackagingResult;
}
