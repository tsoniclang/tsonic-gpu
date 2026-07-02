import type { TargetCompileInput, TargetCompileResult, TargetDiagnostic } from "@tsonic/target-api";
import type { GpuBackendPlugin } from "../../backends/backend-contract.js";
import { createGpuHostRegistry, type GpuHostRegistry } from "../../hosts/host-registry.js";
import type { GpuIrFunction, GpuIrModule } from "../../ir/ir.js";
import { readGpuHostTargetId } from "../../options/gpu-target-options.js";
import { gpuKernelDeclarationFactKey } from "../../source/gpu-facts/keys.js";
import { extractGpuKernel } from "../extraction/extract-kernel.js";

export const gpuIrModuleName = "gpu_kernels";

const emptyHostRegistry = createGpuHostRegistry([]);

// The GPU target contributes nothing for ordinary host code: host statements
// belong to the selected host target. Only explicit kernel marker facts enter
// GPU compilation, and every kernel either extracts fully or fails closed.
export function planGpuArtifacts(
  input: TargetCompileInput,
  backend: GpuBackendPlugin,
  hosts: GpuHostRegistry = emptyHostRegistry,
): TargetCompileResult {
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
  const hostIntegration = hosts.get(hostTargetId);
  if (hostIntegration === undefined) {
    return {
      artifacts: [],
      diagnostics: [
        {
          code: "GPU_HOST_INTEGRATION_MISSING",
          category: "error",
          source: "tsonic-gpu",
          message: `No GPU host integration is registered for host target '${hostTargetId}'. The GPU core does not decide host project layout; register a host integration for '${hostTargetId}'.`,
          evidence: [
            "target.capability=gpu.host.integration",
            `gpu.hostTarget=${hostTargetId}`,
            `gpu.backend=${backend.id}`,
            `gpu.hosts.registered=${hosts.ids().length === 0 ? "(none)" : hosts.ids().join(",")}`,
          ],
        },
      ],
    };
  }

  const lowered = backend.lower(module, { hostTargetId });
  const packaged = hostIntegration.packageArtifacts({
    backendId: backend.id,
    hostTargetId,
    moduleName: module.name,
    modules: lowered.modules,
    dependencies: lowered.dependencies,
    launchWrappers: lowered.launchWrappers,
  });
  if (packaged.diagnostics.length > 0) {
    return { artifacts: [], diagnostics: [...packaged.diagnostics] };
  }
  return { artifacts: packaged.artifacts, diagnostics: [] };
}
