import type { TargetCompileInput, TargetCompileResult, TargetDiagnostic } from "@tsonic/target-api";
import type { GpuBackendPlugin } from "../../backends/backend-contract.js";
import { gpuKernelDeclarationFactKey } from "../../source/gpu-facts/keys.js";
import { kernelExtractionUnavailableDiagnostic } from "./diagnostics.js";

// The GPU target contributes nothing for ordinary host code: host statements
// belong to the selected host target. Only explicit kernel marker facts enter
// GPU compilation, and every kernel fails closed until extraction exists.
export function planGpuArtifacts(input: TargetCompileInput, backend: GpuBackendPlugin): TargetCompileResult {
  const diagnostics: TargetDiagnostic[] = [];
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
      diagnostics.push(
        kernelExtractionUnavailableDiagnostic({ ast, sourceFile, node: statement }, kernelFact.kernelName, backend.id),
      );
    }
  }
  if (diagnostics.length > 0) {
    return { artifacts: [], diagnostics };
  }
  return { artifacts: [], diagnostics: [] };
}
