import type { TargetDiagnostic } from "@tsonic/target-api";
import type { GpuBackendCapabilitySet } from "../capabilities/capability-set.js";
import type { GpuIrModule } from "../ir/ir.js";

export interface GpuHostLoweringContext {
  readonly hostTargetId: string;
}

export interface GpuBackendModuleArtifact {
  readonly path: string;
  readonly language: string;
  readonly text: string;
}

export interface GpuBackendDependency {
  readonly ecosystem: string;
  readonly name: string;
  readonly versionConstraint?: string;
}

export interface GpuLaunchWrapperRequest {
  readonly hostFunctionName: string;
  readonly kernelName: string;
  readonly metaParameters: readonly string[];
}

export interface GpuBackendArtifacts {
  readonly modules: readonly GpuBackendModuleArtifact[];
  readonly dependencies: readonly GpuBackendDependency[];
  readonly launchWrappers: readonly GpuLaunchWrapperRequest[];
}

export interface GpuBackendPlugin {
  readonly id: string;
  describeCapabilities(): GpuBackendCapabilitySet;
  validate(module: GpuIrModule): readonly TargetDiagnostic[];
  lower(module: GpuIrModule, context: GpuHostLoweringContext): GpuBackendArtifacts;
}
