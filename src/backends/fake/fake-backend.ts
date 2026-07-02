import type { TargetDiagnostic } from "@tsonic/target-api";
import {
  gpuBinaryOperatorCapability,
  gpuControlIfCapability,
  gpuControlLoopCapability,
  gpuDeviceCapability,
  gpuDtypeCapability,
  gpuLayoutCapability,
  gpuMathIntrinsicCapability,
  gpuMemoryLoadCapability,
  gpuMemoryMaskedCapability,
  gpuMemoryStoreCapability,
  gpuShapeSymbolicCapability,
  gpuThreadIndexCapability,
  gpuUnaryOperatorCapability,
  type GpuBackendCapabilitySet,
} from "../../capabilities/capability-set.js";
import { matchGpuModuleAgainstCapabilities } from "../../capabilities/match.js";
import type { GpuIrFunction, GpuIrModule } from "../../ir/ir.js";
import { validateGpuIrModule } from "../../ir/validate.js";
import type {
  GpuBackendArtifacts,
  GpuBackendPlugin,
  GpuHostLoweringContext,
} from "../backend-contract.js";

export const fakeGpuBackendId = "fake";

// The fake backend exists so GPU core tests can prove capability negotiation
// and artifact contracts without any real GPU backend or hardware installed.
// It is deliberately narrow: no reductions, atomics, barriers, or wide dtypes.
const fakeCapabilities: GpuBackendCapabilitySet = Object.freeze({
  backendId: fakeGpuBackendId,
  maxTensorRank: 2,
  capabilityIds: Object.freeze([
    gpuDtypeCapability("bool"),
    gpuDtypeCapability("int32"),
    gpuDtypeCapability("float32"),
    gpuDeviceCapability("cuda"),
    gpuLayoutCapability({ kind: "contiguous" }),
    gpuShapeSymbolicCapability,
    gpuThreadIndexCapability("global"),
    gpuThreadIndexCapability("local"),
    gpuThreadIndexCapability("block"),
    gpuBinaryOperatorCapability("add"),
    gpuBinaryOperatorCapability("sub"),
    gpuBinaryOperatorCapability("mul"),
    gpuBinaryOperatorCapability("div"),
    gpuBinaryOperatorCapability("min"),
    gpuBinaryOperatorCapability("max"),
    gpuBinaryOperatorCapability("eq"),
    gpuBinaryOperatorCapability("ne"),
    gpuBinaryOperatorCapability("lt"),
    gpuBinaryOperatorCapability("le"),
    gpuBinaryOperatorCapability("gt"),
    gpuBinaryOperatorCapability("ge"),
    gpuUnaryOperatorCapability("neg"),
    gpuUnaryOperatorCapability("abs"),
    gpuMemoryLoadCapability,
    gpuMemoryStoreCapability,
    gpuMemoryMaskedCapability,
    gpuControlIfCapability,
    gpuControlLoopCapability,
    gpuMathIntrinsicCapability("sqrt"),
    gpuMathIntrinsicCapability("exp"),
    gpuMathIntrinsicCapability("tanh"),
  ]),
});

export function createFakeGpuBackend(): GpuBackendPlugin {
  return {
    id: fakeGpuBackendId,
    describeCapabilities(): GpuBackendCapabilitySet {
      return fakeCapabilities;
    },
    validate(module: GpuIrModule): readonly TargetDiagnostic[] {
      return [...validateGpuIrModule(module), ...matchGpuModuleAgainstCapabilities(module, fakeCapabilities)];
    },
    lower(module: GpuIrModule, context: GpuHostLoweringContext): GpuBackendArtifacts {
      const diagnostics = this.validate(module);
      if (diagnostics.length > 0) {
        throw new Error(
          `The fake GPU backend cannot lower module '${module.name}': validation reported ${diagnostics.length} diagnostic(s).`,
        );
      }
      const kernels = [...module.kernels].sort((left, right) => left.name.localeCompare(right.name, "en"));
      return {
        modules: kernels.map((kernel) => ({
          path: `kernels/${kernel.name}.gpu-fake.json`,
          language: "gpu-fake-module",
          text: fakeKernelModuleText(module.name, kernel, context),
        })),
        dependencies: [],
        launchWrappers: kernels.map((kernel) => ({
          hostFunctionName: kernel.name,
          kernelName: kernel.name,
          metaParameters: kernel.launch.metaParameters ?? [],
        })),
      };
    },
  };
}

function fakeKernelModuleText(moduleName: string, kernel: GpuIrFunction, context: GpuHostLoweringContext): string {
  const record = {
    backend: fakeGpuBackendId,
    hostTarget: context.hostTargetId,
    module: moduleName,
    kernel: kernel.name,
    parameters: kernel.parameters.map((parameter) =>
      parameter.kind === "tensor"
        ? {
            name: parameter.name,
            kind: parameter.kind,
            role: parameter.role,
            dtype: parameter.tensor.elementType,
            rank: parameter.tensor.rank,
            device: parameter.tensor.device.domain,
            mutability: parameter.tensor.mutability,
          }
        : {
            name: parameter.name,
            kind: parameter.kind,
            role: parameter.role,
            dtype: parameter.scalarType,
          },
    ),
    launch: {
      gridDimensions: kernel.launch.grid.length,
      blockDimensions: kernel.launch.block?.length ?? null,
      metaParameters: kernel.launch.metaParameters ?? [],
      streamPolicy: kernel.launch.streamPolicy,
      devicePolicy: kernel.launch.devicePolicy,
    },
    effects: kernel.effects,
    operationCount: kernel.body.operations.length,
  };
  return `${JSON.stringify(record, null, 2)}\n`;
}
