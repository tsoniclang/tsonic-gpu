export { createGpuTargetPack, gpuTargetId, type GpuTargetPackConfig } from "./descriptor/gpu-target-pack.js";
export {
  createTsonicPlugin,
  gpuTargetPluginId,
  readTsonicPluginManifest,
  resolveGpuPluginComposition,
  type GpuBackendTsonicPlugin,
  type GpuHostTsonicPlugin,
  type GpuPluginComposition,
  type GpuTsonicPlugin,
  type TsonicGpuPluginManifest,
} from "./plugin.js";
export {
  readGpuBackendId,
  readGpuBackendPackageName,
  readGpuHostTargetId,
  readGpuTypescriptCompatibilityMode,
  validateGpuTargetOptions,
} from "./options/gpu-target-options.js";
export { createGpuBackend } from "./backend/gpu-backend.js";
export { gpuIrModuleName, planGpuArtifacts } from "./backend/planner/gpu-planner.js";
export {
  extractGpuKernel,
  type GpuKernelExtractionRequest,
  type GpuKernelExtractionResult,
} from "./backend/extraction/extract-kernel.js";
export {
  gpuSourceSpanForNode,
  missingGpuFactDiagnostic,
  unsupportedGpuConstructDiagnostic,
  type GpuDiagnosticInput,
} from "./backend/planner/diagnostics.js";
export {
  type GpuBackendArtifacts,
  type GpuBackendDependency,
  type GpuBackendModuleArtifact,
  type GpuBackendPlugin,
  type GpuHostLoweringContext,
  type GpuLaunchWrapperRequest,
} from "./backends/backend-contract.js";
export { createGpuBackendRegistry, type GpuBackendRegistry } from "./backends/backend-registry.js";
export { createFakeGpuBackend, fakeGpuBackendId } from "./backends/fake/fake-backend.js";
export {
  type GpuHostArtifactRequest,
  type GpuHostIntegration,
  type GpuHostPackagingResult,
} from "./hosts/host-contract.js";
export { createGpuHostRegistry, type GpuHostRegistry } from "./hosts/host-registry.js";
export { createFakeGpuHostIntegration } from "./hosts/fake/fake-host.js";
export {
  gpuAtomicCapability,
  gpuBarrierBlockCapability,
  gpuBinaryOperatorCapability,
  gpuControlIfCapability,
  gpuControlLoopCapability,
  gpuDeviceCapability,
  gpuDtypeCapability,
  gpuLaunchMetaCapability,
  gpuLayoutCapability,
  gpuMutableLocalCapability,
  gpuMathIntrinsicCapability,
  gpuMemoryLoadCapability,
  gpuMemoryMaskedCapability,
  gpuMemoryStoreCapability,
  gpuReduceCapability,
  gpuShapeSymbolicCapability,
  gpuTensorRankCapability,
  gpuThreadIndexCapability,
  gpuUnaryOperatorCapability,
  type GpuBackendCapabilitySet,
} from "./capabilities/capability-set.js";
export { matchGpuModuleAgainstCapabilities, requiredCapabilitiesForKernel } from "./capabilities/match.js";
export { gpuIrDiagnostic, type GpuIrDiagnosticInput } from "./ir/diagnostics.js";
export { type GpuAliasingConstraint, type GpuEffect } from "./ir/effects.js";
export {
  type GpuAtomicOperator,
  type GpuBinaryOperator,
  type GpuIrBlock,
  type GpuIrFunction,
  type GpuIrModule,
  type GpuIrOperation,
  type GpuKernelParameter,
  type GpuMathIntrinsic,
  type GpuReduceOperator,
  type GpuScalarParameterRole,
  type GpuSourceSpan,
  type GpuTensorParameterRole,
  type GpuThreadIndexSpace,
  type GpuUnaryOperator,
} from "./ir/ir.js";
export { type GpuDevicePolicy, type GpuLaunchPlan, type GpuStreamPolicy } from "./ir/launch.js";
export { gpuScalarTypeIds, isGpuScalarType, type GpuScalarType } from "./ir/scalar-types.js";
export {
  gpuShapeBackendMetaNames,
  gpuShapeSymbolNames,
  type GpuShapeConstraint,
  type GpuShapeExpr,
} from "./ir/shape.js";
export {
  gpuDeviceDomains,
  isGpuDeviceDomain,
  type GpuAliasing,
  type GpuDeviceDomain,
  type GpuDeviceRef,
  type GpuLayout,
  type GpuMemorySpace,
  type GpuMutability,
  type GpuTensorType,
} from "./ir/tensor.js";
export { validateGpuIrModule } from "./ir/validate.js";
export {
  gpuExtensionId,
  gpuIntrinsicCallFactKey,
  gpuKernelDeclarationFactKey,
  gpuScalarParameterFactKey,
  gpuTensorAccessCallFactKey,
  gpuTensorParameterFactKey,
  type GpuIntrinsicCallFact,
  type GpuIntrinsicDescriptor,
  type GpuKernelDeclarationFact,
  type GpuScalarParameterFact,
  type GpuTensorAccessCallFact,
  type GpuTensorParameterFact,
} from "./source/gpu-facts/keys.js";
export {
  createGpuLangBindingExtension,
  gpuLangIntrinsicRows,
  gpuLangKernelExportId,
  gpuLangModuleDefinition,
  gpuLangModuleOwnership,
  gpuLangModuleSpecifier,
  type GpuLangIntrinsicRow,
} from "./source/gpu-lang/index.js";
export {
  collectGpuTensorTypeRows,
  createGpuProviderPackage,
  createGpuProviderPackageBindingProvider,
  isGpuTensorTypeContributor,
  type GpuProviderModuleDefinition,
  type GpuProviderPackageDefinition,
  type GpuProviderPackageImplementation,
  type GpuTensorTypeContributor,
  type GpuTensorTypeRow,
} from "./source/provider-packages/index.js";
export {
  createGpuTargetSemanticsExtension,
  gpuTargetSemanticsExtensionId,
  recordGpuFactsBeforeFinalization,
} from "./source/gpu-target-semantics/index.js";
export { createGpuCompileInputFromSession, type GpuCompileInputOptions } from "./session/compile-input.js";
export { createGpuToolchain } from "./toolchain/gpu-toolchain.js";
