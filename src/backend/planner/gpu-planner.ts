import type { TargetArtifact, TargetCompileInput, TargetCompileResult, TargetDiagnostic } from "@tsonic/target-api";
import type { GpuBackendPlugin } from "../../backends/backend-contract.js";
import type { GpuIrFunction, GpuIrModule } from "../../ir/ir.js";
import { readGpuHostTargetId } from "../../options/gpu-target-options.js";
import { gpuKernelDeclarationFactKey } from "../../source/gpu-facts/keys.js";
import { extractGpuKernel } from "../extraction/extract-kernel.js";

export const gpuIrModuleName = "gpu_kernels";

// The GPU target contributes nothing for ordinary host code: host statements
// belong to the selected host target. Only explicit kernel marker facts enter
// GPU compilation, and every kernel either extracts fully or fails closed.
export function planGpuArtifacts(input: TargetCompileInput, backend: GpuBackendPlugin): TargetCompileResult {
  const diagnostics: TargetDiagnostic[] = [];
  const kernels: GpuIrFunction[] = [];
  const ast = input.ast;
  for (const sourceFile of input.sourceFiles) {
    for (const statement of ast.statements(sourceFile)) {
      if (statement === undefined) {
        continue;
      }
      const kernelFact = input.facts.getFact(statement, gpuKernelDeclarationFactKey);
      if (kernelFact === undefined) {
        continue;
      }
      const extraction = extractGpuKernel({ input, sourceFile, statement, fact: kernelFact });
      diagnostics.push(...extraction.diagnostics);
      if (extraction.kernel !== undefined) {
        kernels.push(extraction.kernel);
      }
    }
  }
  if (diagnostics.length > 0) {
    return { artifacts: [], diagnostics };
  }
  if (kernels.length === 0) {
    return { artifacts: [], diagnostics: [] };
  }

  const module: GpuIrModule = { name: gpuIrModuleName, kernels };
  const backendDiagnostics = backend.validate(module);
  if (backendDiagnostics.length > 0) {
    return { artifacts: [], diagnostics: [...backendDiagnostics] };
  }

  const hostTargetId = readGpuHostTargetId(input.target);
  const lowered = backend.lower(module, { hostTargetId });
  const artifacts: TargetArtifact[] = [
    ...lowered.modules.map((moduleArtifact) => ({
      kind: "source" as const,
      language: moduleArtifact.language,
      path: moduleArtifact.path,
      text: moduleArtifact.text,
    })),
    {
      kind: "configuration" as const,
      path: "gpu/launch-plan.json",
      text: `${JSON.stringify(
        {
          backend: backend.id,
          hostTarget: hostTargetId,
          dependencies: lowered.dependencies,
          launchWrappers: lowered.launchWrappers,
        },
        null,
        2,
      )}\n`,
    },
  ];
  return { artifacts, diagnostics: [] };
}
