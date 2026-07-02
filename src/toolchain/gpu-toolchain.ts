import type {
  TargetToolchain,
  TargetToolchainContext,
  TargetToolchainInput,
  TargetToolchainResult,
} from "@tsonic/target-api";

// The GPU target is source-to-source: backend plugins emit artifacts and the
// selected host target owns packaging and any external toolchain handoff.
export function createGpuToolchain(_context: TargetToolchainContext): TargetToolchain {
  return {
    prepare(input: TargetToolchainInput): TargetToolchainResult {
      return {
        diagnostics: [],
        producedArtifacts: input.compileResult.artifacts.map((artifact) => artifact.path),
      };
    },
  };
}
