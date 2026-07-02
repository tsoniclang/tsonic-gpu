import type { CompilerExtension } from "@tsonic/tsts";
import type { GpuIntrinsicDescriptor } from "../gpu-facts/keys.js";
import {
  createGpuProviderPackageBindingProvider,
  type GpuProviderModuleDefinition,
  type GpuProviderPackageDefinition,
} from "../provider-packages/index.js";

// The GPU language surface: the explicit kernel marker and the device
// intrinsic namespace. This module is owned by the GPU target itself; user
// code imports it explicitly, and nothing becomes a kernel without it.

export const gpuLangModuleSpecifier = "@tsonic/gpu/lang.js";
export const gpuLangProviderModuleId = "tsonic.gpu.lang";
export const gpuLangKernelExportId = `${gpuLangModuleSpecifier}::kernel`;
export const gpuLangGpuExportId = `${gpuLangModuleSpecifier}::gpu`;

export interface GpuLangIntrinsicRow {
  readonly memberId: string;
  readonly intrinsic: GpuIntrinsicDescriptor;
}

const int32Type = { kind: "source-primitive", name: "int32" } as const;
const float32Type = { kind: "source-primitive", name: "float32" } as const;
const numberType = { kind: "number" } as const;

function gpuMemberId(memberName: string): string {
  return `${gpuLangGpuExportId}.${memberName}`;
}

function threadIndexMember(memberName: string) {
  return {
    id: gpuMemberId(memberName),
    name: memberName,
    kind: "method",
    static: true,
    signatures: [
      {
        id: `${gpuMemberId(memberName)}(dimension)`,
        name: memberName,
        parameters: [{ name: "dimension", type: numberType }],
        returnType: int32Type,
      },
    ],
  } as const;
}

function mathMember(memberName: string) {
  return {
    id: gpuMemberId(memberName),
    name: memberName,
    kind: "method",
    static: true,
    signatures: [
      {
        id: `${gpuMemberId(memberName)}(value)`,
        name: memberName,
        parameters: [{ name: "value", type: float32Type }],
        returnType: float32Type,
      },
    ],
  } as const;
}

export function gpuLangModuleDefinition(): GpuProviderModuleDefinition {
  return {
    moduleSpecifier: gpuLangModuleSpecifier,
    providerModuleId: gpuLangProviderModuleId,
    exports: [
      {
        id: gpuLangKernelExportId,
        name: "kernel",
        kind: "function",
        signatures: [
          {
            id: `${gpuLangKernelExportId}(fn)`,
            name: "kernel",
            typeParameters: [{ name: "T" }],
            parameters: [{ name: "fn", type: { kind: "type-parameter", name: "T" } }],
            returnType: { kind: "type-parameter", name: "T" },
          },
        ],
      },
      {
        id: gpuLangGpuExportId,
        name: "gpu",
        kind: "class",
        members: [
          threadIndexMember("globalId"),
          threadIndexMember("localId"),
          threadIndexMember("blockId"),
          mathMember("sqrt"),
          mathMember("exp"),
          mathMember("tanh"),
        ],
      },
    ],
  };
}

export function gpuLangIntrinsicRows(): readonly GpuLangIntrinsicRow[] {
  return [
    { memberId: gpuMemberId("globalId"), intrinsic: { kind: "thread-index", space: "global" } },
    { memberId: gpuMemberId("localId"), intrinsic: { kind: "thread-index", space: "local" } },
    { memberId: gpuMemberId("blockId"), intrinsic: { kind: "thread-index", space: "block" } },
    { memberId: gpuMemberId("sqrt"), intrinsic: { kind: "math", name: "sqrt", dtype: "float32" } },
    { memberId: gpuMemberId("exp"), intrinsic: { kind: "math", name: "exp", dtype: "float32" } },
    { memberId: gpuMemberId("tanh"), intrinsic: { kind: "math", name: "tanh", dtype: "float32" } },
  ];
}

const gpuLangPackageDefinition: GpuProviderPackageDefinition = {
  id: "gpu-lang",
  displayName: "GPU language surface",
  version: "0.0.1",
  modules: [gpuLangModuleDefinition()],
};

export function gpuLangModuleOwnership(): readonly { readonly specifierPrefix: string }[] {
  return [{ specifierPrefix: gpuLangModuleSpecifier }];
}

export function createGpuLangBindingExtension(): CompilerExtension {
  return {
    identity: {
      id: "tsonic.gpu.lang",
      version: "0.0.1",
      capabilityNamespace: "tsonic.gpu.lang",
    },
    initialize(context): void {
      context.registerTargetBindingProvider(createGpuProviderPackageBindingProvider(gpuLangPackageDefinition));
    },
  };
}
