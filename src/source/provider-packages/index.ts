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
import { isGpuScalarType, type GpuScalarType } from "../../ir/scalar-types.js";
import { isGpuDeviceDomain, type GpuDeviceDomain } from "../../ir/tensor.js";

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

// Definitions are configuration; every inconsistency fails at package
// creation instead of surfacing later as a missing or wrong fact.
function validateGpuProviderPackageDefinition(definition: GpuProviderPackageDefinition): void {
  const packageError = (message: string): Error => new Error(`GPU provider package '${definition.id}': ${message}`);
  const moduleSpecifiers = new Set<string>();
  const exportsById = new Map<string, (typeof definition.modules)[number]["exports"][number]>();
  for (const module of definition.modules) {
    if (module.moduleSpecifier.length === 0) {
      throw packageError("module specifiers must be non-empty strings.");
    }
    if (moduleSpecifiers.has(module.moduleSpecifier)) {
      throw packageError(`module '${module.moduleSpecifier}' is declared more than once.`);
    }
    moduleSpecifiers.add(module.moduleSpecifier);
    for (const exportDeclaration of module.exports) {
      if (exportsById.has(exportDeclaration.id)) {
        throw packageError(`export '${exportDeclaration.id}' is declared more than once.`);
      }
      exportsById.set(exportDeclaration.id, exportDeclaration);
    }
  }
  const tensorRowExports = new Set<string>();
  for (const row of definition.tensorTypes ?? []) {
    const rowError = (message: string): Error => packageError(`tensor row '${row.exportId}' ${message}`);
    const exportDeclaration = exportsById.get(row.exportId);
    if (exportDeclaration === undefined) {
      throw rowError("references an export this package does not declare.");
    }
    if (tensorRowExports.has(row.exportId)) {
      throw rowError("is declared more than once.");
    }
    tensorRowExports.add(row.exportId);
    if (!isGpuScalarType(row.elementType)) {
      throw rowError(`has unknown element dtype '${String(row.elementType)}'.`);
    }
    if (!isGpuDeviceDomain(row.device)) {
      throw rowError(`has unknown device domain '${String(row.device)}'.`);
    }
    if (!Number.isInteger(row.rank) || row.rank < 1) {
      throw rowError(`needs an integer rank of at least 1, found '${String(row.rank)}'.`);
    }
    if (row.shapeSymbolArguments !== undefined) {
      if (row.shapeSymbolArguments.length !== row.rank) {
        throw rowError(`declares ${row.shapeSymbolArguments.length} shape symbol arguments for rank ${row.rank}.`);
      }
      const typeParameterCount = (exportDeclaration.typeParameters ?? []).length;
      const positions = new Set<number>();
      for (const position of row.shapeSymbolArguments) {
        if (!Number.isInteger(position) || position < 0) {
          throw rowError(`has an invalid shape symbol argument position '${String(position)}'.`);
        }
        if (positions.has(position)) {
          throw rowError(`repeats shape symbol argument position ${position}.`);
        }
        if (position >= typeParameterCount) {
          throw rowError(
            `declares shape symbol argument position ${position}, but export '${row.exportId}' declares ${typeParameterCount} type parameter(s).`,
          );
        }
        positions.add(position);
      }
    }
    const memberIds = new Set((exportDeclaration.members ?? []).map((member) => member.id));
    for (const [label, memberId] of [
      ["loadMember", row.loadMember],
      ["storeMember", row.storeMember],
    ] as const) {
      if (memberId === undefined) {
        continue;
      }
      if (memberId.length === 0) {
        throw rowError(`declares an empty ${label} id.`);
      }
      if (!memberIds.has(memberId)) {
        throw rowError(`declares ${label} '${memberId}', which is not a member of export '${row.exportId}'.`);
      }
    }
  }
}

export function createGpuProviderPackage(definition: GpuProviderPackageDefinition): GpuProviderPackageImplementation {
  validateGpuProviderPackageDefinition(definition);
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
