import { defineExtensionFactKey } from "@tsonic/tsts";

export const gpuExtensionId = "tsonic.gpu";

export interface GpuKernelDeclarationFact {
  readonly kernelName: string;
}

export const gpuKernelDeclarationFactKey = defineExtensionFactKey<GpuKernelDeclarationFact>({
  extensionId: gpuExtensionId,
  name: "kernelDeclaration",
  equals: (left, right) => left.kernelName === right.kernelName,
});
