import { tsonicCoreSourceExtensionId } from "@tsonic/source-core";
import type { CompilerExtension } from "@tsonic/tsts";
import type { TargetProviderContext } from "@tsonic/target-api";
import { gpuTargetId } from "../../descriptor/target-id.js";
import { gpuExtensionId } from "../gpu-facts/keys.js";

export const gpuTargetSemanticsExtensionId = "tsonic.gpu.target-semantics";

// Kernel marker recognition lands with extraction (G3). Until then the
// extension only declares the GPU target's identity and composition so the
// host composes it after core source semantics.
export function createGpuTargetSemanticsExtension(_context: TargetProviderContext): CompilerExtension {
  return {
    identity: {
      id: gpuTargetSemanticsExtensionId,
      version: "0.0.1",
      capabilityNamespace: gpuExtensionId,
    },
    dependencies: {
      dependsOn: [tsonicCoreSourceExtensionId],
      runsAfter: [tsonicCoreSourceExtensionId],
    },
    composition: { kind: "target", target: gpuTargetId },
  };
}
