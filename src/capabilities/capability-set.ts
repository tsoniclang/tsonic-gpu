import type {
  GpuAtomicOperator,
  GpuBinaryOperator,
  GpuMathIntrinsic,
  GpuReduceOperator,
  GpuThreadIndexSpace,
  GpuUnaryOperator,
} from "../ir/ir.js";
import type { GpuScalarType } from "../ir/scalar-types.js";
import type { GpuDeviceDomain, GpuLayout } from "../ir/tensor.js";

export interface GpuBackendCapabilitySet {
  readonly backendId: string;
  readonly maxTensorRank: number;
  readonly capabilityIds: readonly string[];
}

export function gpuDtypeCapability(dtype: GpuScalarType): string {
  return `gpu.dtype.${dtype}`;
}

export function gpuDeviceCapability(domain: GpuDeviceDomain): string {
  return `gpu.device.${domain}`;
}

export function gpuLayoutCapability(layout: GpuLayout): string {
  return layout.kind === "backend-specific" ? `gpu.layout.backend.${layout.layoutId}` : `gpu.layout.${layout.kind}`;
}

export function gpuBinaryOperatorCapability(operator: GpuBinaryOperator): string {
  return `gpu.op.binary.${operator}`;
}

export function gpuUnaryOperatorCapability(operator: GpuUnaryOperator): string {
  return `gpu.op.unary.${operator}`;
}

export function gpuThreadIndexCapability(space: GpuThreadIndexSpace): string {
  return `gpu.thread-index.${space}`;
}

export function gpuReduceCapability(operator: GpuReduceOperator, dtype: GpuScalarType): string {
  return `gpu.reduce.${operator}.${dtype}`;
}

export function gpuAtomicCapability(operator: GpuAtomicOperator, dtype: GpuScalarType): string {
  return `gpu.atomic.${operator}.${dtype}`;
}

export function gpuMathIntrinsicCapability(name: GpuMathIntrinsic): string {
  return `gpu.math.${name}`;
}

export const gpuMemoryLoadCapability = "gpu.memory.load";
export const gpuMemoryStoreCapability = "gpu.memory.store";
export const gpuMemoryMaskedCapability = "gpu.memory.masked";
export const gpuControlIfCapability = "gpu.control.if";
export const gpuControlLoopCapability = "gpu.control.loop";
export const gpuMutableLocalCapability = "gpu.local.mutable";
export const gpuLaunchMetaCapability = "gpu.launch.meta";
export const gpuBarrierBlockCapability = "gpu.barrier.block";
export const gpuShapeSymbolicCapability = "gpu.shape.symbolic";

export function gpuTensorRankCapability(rank: number): string {
  return `gpu.tensor.rank.${rank}`;
}
