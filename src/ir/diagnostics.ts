import type { TargetDiagnostic } from "@tsonic/target-api";
import type { GpuSourceSpan } from "./ir.js";

export interface GpuIrDiagnosticInput {
  readonly code: string;
  readonly capabilityId: string;
  readonly message: string;
  readonly moduleName: string;
  readonly kernelName?: string;
  readonly span?: GpuSourceSpan;
  readonly extraEvidence?: readonly string[];
}

export function gpuIrDiagnostic(input: GpuIrDiagnosticInput): TargetDiagnostic {
  const evidence = [
    `target.capability=${input.capabilityId}`,
    `gpu.module=${input.moduleName}`,
    ...(input.kernelName === undefined ? [] : [`gpu.kernel=${input.kernelName}`]),
    ...(input.extraEvidence ?? []),
  ];
  return {
    code: input.code,
    category: "error",
    source: "tsonic-gpu",
    message: input.message,
    ...(input.span === undefined ? {} : { sourceSpan: input.span }),
    evidence,
  };
}
