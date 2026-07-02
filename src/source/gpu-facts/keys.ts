import { defineExtensionFactKey } from "@tsonic/tsts";
import type { GpuMathIntrinsic, GpuThreadIndexSpace } from "../../ir/ir.js";
import type { GpuScalarType } from "../../ir/scalar-types.js";
import type { GpuDeviceDomain } from "../../ir/tensor.js";

export const gpuExtensionId = "tsonic.gpu";

export interface GpuKernelDeclarationFact {
  readonly kernelName: string;
}

export const gpuKernelDeclarationFactKey = defineExtensionFactKey<GpuKernelDeclarationFact>({
  extensionId: gpuExtensionId,
  name: "kernelDeclaration",
  equals: (left, right) => left.kernelName === right.kernelName,
});

export interface GpuTensorParameterFact {
  readonly elementType: GpuScalarType;
  readonly rank: number;
  readonly device: GpuDeviceDomain;
}

export const gpuTensorParameterFactKey = defineExtensionFactKey<GpuTensorParameterFact>({
  extensionId: gpuExtensionId,
  name: "tensorParameter",
  equals: (left, right) =>
    left.elementType === right.elementType && left.rank === right.rank && left.device === right.device,
});

export interface GpuScalarParameterFact {
  readonly scalarType: GpuScalarType;
}

export const gpuScalarParameterFactKey = defineExtensionFactKey<GpuScalarParameterFact>({
  extensionId: gpuExtensionId,
  name: "scalarParameter",
  equals: (left, right) => left.scalarType === right.scalarType,
});

export type GpuIntrinsicDescriptor =
  | { readonly kind: "thread-index"; readonly space: GpuThreadIndexSpace }
  | { readonly kind: "math"; readonly name: GpuMathIntrinsic; readonly dtype: GpuScalarType };

export interface GpuIntrinsicCallFact {
  readonly intrinsic: GpuIntrinsicDescriptor;
}

export const gpuIntrinsicCallFactKey = defineExtensionFactKey<GpuIntrinsicCallFact>({
  extensionId: gpuExtensionId,
  name: "intrinsicCall",
  equals: (left, right) => {
    if (left.intrinsic.kind === "thread-index" && right.intrinsic.kind === "thread-index") {
      return left.intrinsic.space === right.intrinsic.space;
    }
    if (left.intrinsic.kind === "math" && right.intrinsic.kind === "math") {
      return left.intrinsic.name === right.intrinsic.name && left.intrinsic.dtype === right.intrinsic.dtype;
    }
    return false;
  },
});
