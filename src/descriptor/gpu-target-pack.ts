import type {
  TargetBackend,
  TargetBackendContext,
  TargetPack,
  TargetProviderContext,
  TargetToolchain,
  TargetToolchainContext,
} from "@tsonic/target-api";
import type { CompilerExtension } from "@tsonic/tsts";
import { createGpuBackend } from "../backend/gpu-backend.js";
import type { GpuBackendPlugin } from "../backends/backend-contract.js";
import { createGpuBackendRegistry } from "../backends/backend-registry.js";
import { readGpuBackendId, validateGpuTargetOptions } from "../options/gpu-target-options.js";
import { createGpuTargetSemanticsExtension } from "../source/gpu-target-semantics/index.js";
import { createGpuToolchain } from "../toolchain/gpu-toolchain.js";
import { gpuTargetId } from "./target-id.js";

export { gpuTargetId } from "./target-id.js";

export interface GpuTargetPackConfig {
  readonly backends?: readonly GpuBackendPlugin[];
}

// Backend plugins are handed in explicitly by the caller that wires the
// registry. The pack never infers a backend from imports, file names, or
// installed packages; an unselected or unregistered backend is an error.
export function createGpuTargetPack(config: GpuTargetPackConfig = {}): TargetPack {
  const backendRegistry = createGpuBackendRegistry(config.backends ?? []);
  return {
    id: gpuTargetId,
    displayName: "GPU",
    provider: {
      id: "gpu-provider",
      displayName: "GPU target provider",
      createExtensions(context: TargetProviderContext): readonly CompilerExtension[] {
        validateGpuTargetOptions(context.target);
        return [createGpuTargetSemanticsExtension(context)];
      },
    },
    createBackend(context: TargetBackendContext): TargetBackend {
      validateGpuTargetOptions(context.target);
      const backendId = readGpuBackendId(context.target);
      const plugin = backendRegistry.get(backendId);
      if (plugin === undefined) {
        throw new Error(
          `GPU backend '${backendId}' is not registered with the GPU target pack. Registered backends: ${
            backendRegistry.ids().length === 0 ? "(none)" : backendRegistry.ids().join(", ")
          }.`,
        );
      }
      return createGpuBackend(plugin);
    },
    createToolchain(context: TargetToolchainContext): TargetToolchain {
      validateGpuTargetOptions(context.target);
      return createGpuToolchain(context);
    },
  };
}
