import { TstsProviderContractVersion } from "@tsonic/tsts";
import type {
  CompilerExtension,
  ProviderDeclarationModel,
  ProviderExportDeclaration,
  ProviderModuleResolution,
  TargetBindingProvider,
} from "@tsonic/tsts";
import type { TargetProviderPackageImplementation } from "@tsonic/target-api";
import { gpuTargetId } from "../../descriptor/target-id.js";
import type { GpuScalarType } from "../../ir/scalar-types.js";
import type { GpuDeviceDomain } from "../../ir/tensor.js";

// Generic GPU provider-package model. Concrete module specifiers, export
// names, and tensor type identities live only in package definitions (product
// packages or test fakes), never in generic mapping code. This is how any
// host tensor library plugs into the GPU core without the core naming it.

export interface GpuProviderModuleDefinition {
  readonly moduleSpecifier: string;
  readonly providerModuleId: string;
  readonly exports: readonly ProviderExportDeclaration[];
}

export interface GpuTensorTypeRow {
  readonly exportId: string;
  readonly elementType: GpuScalarType;
  readonly rank: number;
  readonly device: GpuDeviceDomain;
  // Type-argument positions whose named type parameters become dimension
  // symbols; sharing a type parameter across parameters shares the symbol.
  // The list length must equal the rank.
  readonly shapeSymbolArguments?: readonly number[];
  // Provider member ids for element access: loadMember reads one element
  // (t.at(...indices)), storeMember writes one (t.set(...indices, value)).
  readonly loadMember?: string;
  readonly storeMember?: string;
}

export interface GpuProviderPackageDefinition {
  readonly id: string;
  readonly displayName: string;
  readonly version: string;
  readonly modules: readonly GpuProviderModuleDefinition[];
  readonly tensorTypes?: readonly GpuTensorTypeRow[];
}

export interface GpuTensorTypeContributor {
  gpuTensorTypes(): readonly GpuTensorTypeRow[];
}

export type GpuProviderPackageImplementation = TargetProviderPackageImplementation & GpuTensorTypeContributor;

export function createGpuProviderPackage(definition: GpuProviderPackageDefinition): GpuProviderPackageImplementation {
  return {
    id: definition.id,
    displayName: definition.displayName,
    moduleOwnership: definition.modules.map((module) => ({ specifierPrefix: module.moduleSpecifier })),
    createExtensions(): readonly CompilerExtension[] {
      return [createGpuProviderPackageBindingExtension(definition)];
    },
    gpuTensorTypes(): readonly GpuTensorTypeRow[] {
      return definition.tensorTypes ?? [];
    },
  };
}

export function isGpuTensorTypeContributor(value: object): value is GpuTensorTypeContributor {
  return typeof (value as { gpuTensorTypes?: unknown }).gpuTensorTypes === "function";
}

export function collectGpuTensorTypeRows(selectedPackages: readonly object[]): readonly GpuTensorTypeRow[] {
  const rows: GpuTensorTypeRow[] = [];
  for (const selectedPackage of selectedPackages) {
    if (isGpuTensorTypeContributor(selectedPackage)) {
      rows.push(...selectedPackage.gpuTensorTypes());
    }
  }
  return rows;
}

function createGpuProviderPackageBindingExtension(definition: GpuProviderPackageDefinition): CompilerExtension {
  return {
    identity: {
      id: `tsonic.gpu.provider-package.${definition.id}`,
      version: definition.version,
      capabilityNamespace: `tsonic.gpu.provider-package.${definition.id}`,
    },
    initialize(context): void {
      context.registerTargetBindingProvider(createGpuProviderPackageBindingProvider(definition));
    },
  };
}

export function createGpuProviderPackageBindingProvider(definition: GpuProviderPackageDefinition): TargetBindingProvider {
  const modulesBySpecifier = new Map(definition.modules.map((module) => [module.moduleSpecifier, module]));
  return {
    identity: {
      id: `tsonic.gpu.provider-package.${definition.id}.binding`,
      version: definition.version,
      target: gpuTargetId,
      extensionContractVersion: TstsProviderContractVersion,
      providerKind: "binding",
    },
    ownsModule(specifier: string) {
      return modulesBySpecifier.has(specifier) ? { kind: "owned" as const } : { kind: "unowned" as const };
    },
    resolveModule(specifier: string) {
      const module = modulesBySpecifier.get(specifier);
      if (module === undefined) {
        return {
          extensionId: `tsonic.gpu.provider-package.${definition.id}`,
          extensionCode: "GPU_PROVIDER_MODULE_NOT_OWNED",
          numericCode: 0,
          category: "error" as const,
          message: `GPU provider package '${definition.id}' does not own module '${specifier}'.`,
        };
      }
      return {
        kind: "virtual" as const,
        moduleSpecifier: module.moduleSpecifier,
        virtualFileName: `tsts-provider://tsonic-gpu/${definition.id}/${encodeURIComponent(module.moduleSpecifier)}.d.ts`,
        providerModuleId: module.providerModuleId,
        packageName: module.moduleSpecifier,
        packageVersion: definition.version,
      };
    },
    getDeclarationModel(resolution: ProviderModuleResolution): ProviderDeclarationModel {
      const module = modulesBySpecifier.get(resolution.moduleSpecifier);
      if (module === undefined) {
        return { moduleSpecifier: resolution.moduleSpecifier, providerModuleId: resolution.providerModuleId, exports: [] };
      }
      return {
        moduleSpecifier: module.moduleSpecifier,
        providerModuleId: module.providerModuleId,
        exports: module.exports,
      };
    },
    getTargetIdentity(): undefined {
      return undefined;
    },
  };
}
