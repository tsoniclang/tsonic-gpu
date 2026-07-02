import type { TargetDiagnostic } from "@tsonic/target-api";
import { gpuIrDiagnostic } from "../ir/diagnostics.js";
import type { GpuIrBlock, GpuIrFunction, GpuIrModule } from "../ir/ir.js";
import { gpuShapeSymbolNames } from "../ir/shape.js";
import {
  gpuAtomicCapability,
  gpuBarrierBlockCapability,
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
  gpuReduceCapability,
  gpuShapeSymbolicCapability,
  gpuThreadIndexCapability,
  gpuUnaryOperatorCapability,
  type GpuBackendCapabilitySet,
} from "./capability-set.js";

export function requiredCapabilitiesForKernel(kernel: GpuIrFunction): readonly string[] {
  const required = new Set<string>();
  for (const parameter of kernel.parameters) {
    if (parameter.kind === "tensor") {
      required.add(gpuDtypeCapability(parameter.tensor.elementType));
      required.add(gpuLayoutCapability(parameter.tensor.layout));
      required.add(gpuDeviceCapability(parameter.tensor.device.domain));
      if (parameter.tensor.shape.some((dimension) => gpuShapeSymbolNames(dimension).length > 0)) {
        required.add(gpuShapeSymbolicCapability);
      }
    } else {
      required.add(gpuDtypeCapability(parameter.scalarType));
    }
  }
  collectBlockCapabilities(kernel.body, required);
  return [...required].sort();
}

function collectBlockCapabilities(block: GpuIrBlock, required: Set<string>): void {
  for (const operation of block.operations) {
    switch (operation.kind) {
      case "const":
        required.add(gpuDtypeCapability(operation.dtype));
        break;
      case "thread-index":
        required.add(gpuThreadIndexCapability(operation.space));
        break;
      case "binary":
        required.add(gpuBinaryOperatorCapability(operation.operator));
        required.add(gpuDtypeCapability(operation.dtype));
        break;
      case "unary":
        required.add(gpuUnaryOperatorCapability(operation.operator));
        required.add(gpuDtypeCapability(operation.dtype));
        break;
      case "load":
        required.add(gpuMemoryLoadCapability);
        required.add(gpuDtypeCapability(operation.dtype));
        if (operation.mask !== undefined) {
          required.add(gpuMemoryMaskedCapability);
        }
        break;
      case "store":
        required.add(gpuMemoryStoreCapability);
        if (operation.mask !== undefined) {
          required.add(gpuMemoryMaskedCapability);
        }
        break;
      case "if":
        required.add(gpuControlIfCapability);
        collectBlockCapabilities(operation.then, required);
        if (operation.else !== undefined) {
          collectBlockCapabilities(operation.else, required);
        }
        break;
      case "loop":
        required.add(gpuControlLoopCapability);
        collectBlockCapabilities(operation.body, required);
        break;
      case "reduce":
        required.add(gpuReduceCapability(operation.operator, operation.dtype));
        break;
      case "intrinsic":
        required.add(gpuMathIntrinsicCapability(operation.name));
        required.add(gpuDtypeCapability(operation.dtype));
        break;
      case "barrier":
        required.add(gpuBarrierBlockCapability);
        break;
      case "atomic":
        required.add(gpuAtomicCapability(operation.operator, operation.dtype));
        break;
      case "return":
        break;
    }
  }
}

export function matchGpuModuleAgainstCapabilities(
  module: GpuIrModule,
  capabilities: GpuBackendCapabilitySet,
): readonly TargetDiagnostic[] {
  const diagnostics: TargetDiagnostic[] = [];
  const supported = new Set(capabilities.capabilityIds);
  for (const kernel of module.kernels) {
    for (const capabilityId of requiredCapabilitiesForKernel(kernel)) {
      if (!supported.has(capabilityId)) {
        diagnostics.push(
          gpuIrDiagnostic({
            code: "GPU_BACKEND_CAPABILITY_MISSING",
            capabilityId,
            message: `GPU backend '${capabilities.backendId}' does not support capability '${capabilityId}' required by kernel '${kernel.name}'.`,
            moduleName: module.name,
            kernelName: kernel.name,
            ...(kernel.span === undefined ? {} : { span: kernel.span }),
            extraEvidence: [`gpu.backend=${capabilities.backendId}`],
          }),
        );
      }
    }
    for (const parameter of kernel.parameters) {
      if (parameter.kind === "tensor" && parameter.tensor.rank > capabilities.maxTensorRank) {
        diagnostics.push(
          gpuIrDiagnostic({
            code: "GPU_BACKEND_CAPABILITY_MISSING",
            capabilityId: `gpu.tensor.rank.${parameter.tensor.rank}`,
            message: `GPU backend '${capabilities.backendId}' supports tensor rank up to ${capabilities.maxTensorRank}; kernel '${kernel.name}' parameter '${parameter.name}' has rank ${parameter.tensor.rank}.`,
            moduleName: module.name,
            kernelName: kernel.name,
            ...(kernel.span === undefined ? {} : { span: kernel.span }),
            extraEvidence: [`gpu.backend=${capabilities.backendId}`],
          }),
        );
      }
    }
  }
  return diagnostics;
}
