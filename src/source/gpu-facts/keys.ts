import { defineExtensionFactKey } from "@tsonic/tsts";
import type { GpuMathIntrinsic, GpuReduceOperator, GpuThreadIndexSpace } from "../../ir/ir.js";
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
  // Dimension symbol names declared through the tensor type's generic type
  // arguments; sharing a symbol across parameters states a shape-equality
  // constraint (for example A[M,K] x B[K,N] -> C[M,N]).
  readonly shape?: readonly string[];
}

export const gpuTensorParameterFactKey = defineExtensionFactKey<GpuTensorParameterFact>({
  extensionId: gpuExtensionId,
  name: "tensorParameter",
  equals: (left, right) =>
    left.elementType === right.elementType &&
    left.rank === right.rank &&
    left.device === right.device &&
    (left.shape ?? []).length === (right.shape ?? []).length &&
    (left.shape ?? []).every((symbol, index) => symbol === (right.shape ?? [])[index]),
});

export interface GpuTensorAccessCallFact {
  readonly access: "load" | "store";
}

export const gpuTensorAccessCallFactKey = defineExtensionFactKey<GpuTensorAccessCallFact>({
  extensionId: gpuExtensionId,
  name: "tensorAccessCall",
  equals: (left, right) => left.access === right.access,
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
  | { readonly kind: "math"; readonly name: GpuMathIntrinsic; readonly dtype: GpuScalarType }
  | { readonly kind: "block-reduce"; readonly operator: GpuReduceOperator; readonly dtype: GpuScalarType }
  | { readonly kind: "shape-dim" }
  | { readonly kind: "meta-parameter" };

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
    if (left.intrinsic.kind === "block-reduce" && right.intrinsic.kind === "block-reduce") {
      return left.intrinsic.operator === right.intrinsic.operator && left.intrinsic.dtype === right.intrinsic.dtype;
    }
    if (left.intrinsic.kind === "shape-dim" && right.intrinsic.kind === "shape-dim") {
      return true;
    }
    if (left.intrinsic.kind === "meta-parameter" && right.intrinsic.kind === "meta-parameter") {
      return true;
    }
    return false;
  },
});
