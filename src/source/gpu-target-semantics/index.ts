import { ExtensionLifecycleEvent, providerVirtualDeclarationFactKey, sourcePrimitiveFactKey } from "@tsonic/tsts";
import { tsonicCoreSourceExtensionId } from "@tsonic/source-core";
import type { CompilerExtension, ExtensionLifecycleContext, Node, ProviderDeclarationIdentity } from "@tsonic/tsts";
import type { TargetProviderContext } from "@tsonic/target-api";
import {
  KindCallExpression,
  KindFunctionExpression,
  KindTypeReference,
  KindVariableDeclaration,
  KindVariableStatement,
  Node_Expression,
  Node_Initializer,
  Node_Type,
  TypeReferenceNode_TypeName,
  VariableStatement_DeclarationList,
} from "../../common/source-ast.js";
import { gpuTargetId } from "../../descriptor/target-id.js";
import type { GpuScalarType } from "../../ir/scalar-types.js";
import { isGpuScalarType } from "../../ir/scalar-types.js";
import {
  gpuExtensionId,
  gpuIntrinsicCallFactKey,
  gpuKernelDeclarationFactKey,
  gpuScalarParameterFactKey,
  gpuTensorParameterFactKey,
} from "../gpu-facts/keys.js";
import { gpuLangIntrinsicRows, gpuLangKernelExportId, type GpuLangIntrinsicRow } from "../gpu-lang/index.js";
import { collectGpuTensorTypeRows, type GpuTensorTypeRow } from "../provider-packages/index.js";
import { validateGpuTargetOptions } from "../../options/gpu-target-options.js";

export const gpuTargetSemanticsExtensionId = "tsonic.gpu.target-semantics";

// Records the finalized GPU facts the backend consumes: kernel markers on
// declarations, tensor/scalar facts on kernel parameters, and intrinsic facts
// on device calls. Identities come from provider declaration facts, never
// from source spellings.
export function createGpuTargetSemanticsExtension(context: TargetProviderContext): CompilerExtension {
  validateGpuTargetOptions(context.target);
  const tensorRows = collectGpuTensorTypeRows(context.selectedPackages);
  const intrinsicRows = gpuLangIntrinsicRows();
  return {
    identity: {
      id: gpuTargetSemanticsExtensionId,
      version: "0.0.1",
      capabilityNamespace: gpuExtensionId,
    },
    dependencies: {
      dependsOn: [tsonicCoreSourceExtensionId],
      runsAfter: [tsonicCoreSourceExtensionId],
    },
    composition: { kind: "target", target: gpuTargetId },
    initialize(extensionContext): void {
      extensionContext.registerLifecycleHook(ExtensionLifecycleEvent.beforeSemanticsFinalized, (_request, lifecycleContext) => {
        recordGpuFactsBeforeFinalization(lifecycleContext, tensorRows, intrinsicRows);
      });
    },
  };
}

export function recordGpuFactsBeforeFinalization(
  lifecycle: ExtensionLifecycleContext,
  tensorRows: readonly GpuTensorTypeRow[],
  intrinsicRows: readonly GpuLangIntrinsicRow[],
): void {
  const { ast } = lifecycle.compiler;
  for (const sourceFile of lifecycle.compiler.getSourceFiles()) {
    if (sourceFile === undefined || ast.getFileName(sourceFile).endsWith(".d.ts")) {
      continue;
    }
    for (const statement of ast.statements(sourceFile)) {
      if (statement === undefined || ast.kindName(statement) !== KindVariableStatement) {
        continue;
      }
      recordKernelStatementFacts(lifecycle, statement, tensorRows, intrinsicRows);
    }
  }
}

function recordKernelStatementFacts(
  lifecycle: ExtensionLifecycleContext,
  statement: Node,
  tensorRows: readonly GpuTensorTypeRow[],
  intrinsicRows: readonly GpuLangIntrinsicRow[],
): void {
  const { ast } = lifecycle.compiler;
  const declarationList = VariableStatement_DeclarationList(statement);
  if (declarationList === undefined) {
    return;
  }
  ast.forEachChild(declarationList, (declaration) => {
    if (declaration === undefined || ast.kindName(declaration) !== KindVariableDeclaration) {
      return;
    }
    const initializer = Node_Initializer(declaration);
    if (initializer === undefined || ast.kindName(initializer) !== KindCallExpression) {
      return;
    }
    const callee = Node_Expression(initializer);
    const identity = callee === undefined ? undefined : providerIdentityFor(lifecycle, callee);
    if (identity === undefined || identity.exportId !== gpuLangKernelExportId || identity.memberId !== undefined) {
      return;
    }
    const [kernelArgument] = ast.arguments(initializer);
    const kernelFunction =
      kernelArgument !== undefined && ast.kindName(kernelArgument) === KindFunctionExpression ? kernelArgument : undefined;
    const kernelName = kernelNameFor(lifecycle, kernelFunction, declaration);
    lifecycle.host.facts.set(statement, gpuKernelDeclarationFactKey, { kernelName }, [
      { message: "explicit gpu kernel marker" },
    ]);
    if (kernelFunction === undefined) {
      return;
    }
    recordParameterFacts(lifecycle, kernelFunction, tensorRows);
    recordIntrinsicFacts(lifecycle, kernelFunction, intrinsicRows);
  });
}

function kernelNameFor(lifecycle: ExtensionLifecycleContext, kernelFunction: Node | undefined, declaration: Node): string {
  const { ast } = lifecycle.compiler;
  const functionName = kernelFunction === undefined ? undefined : ast.name(kernelFunction);
  if (functionName !== undefined) {
    const text = ast.text(functionName);
    if (text.length > 0) {
      return text;
    }
  }
  const declarationName = ast.name(declaration);
  return declarationName === undefined ? "" : ast.text(declarationName);
}

function recordParameterFacts(
  lifecycle: ExtensionLifecycleContext,
  kernelFunction: Node,
  tensorRows: readonly GpuTensorTypeRow[],
): void {
  const { ast } = lifecycle.compiler;
  for (const parameter of ast.parameters(kernelFunction)) {
    if (parameter === undefined) {
      continue;
    }
    const typeNode = Node_Type(parameter);
    if (typeNode === undefined) {
      continue;
    }
    if (ast.kindName(typeNode) === KindTypeReference) {
      const typeName = TypeReferenceNode_TypeName(typeNode) ?? typeNode;
      const identity = providerIdentityFor(lifecycle, typeName);
      const tensorRow =
        identity === undefined || identity.memberId !== undefined
          ? undefined
          : tensorRows.find((row) => row.exportId === identity.exportId);
      if (tensorRow !== undefined) {
        lifecycle.host.facts.set(
          parameter,
          gpuTensorParameterFactKey,
          { elementType: tensorRow.elementType, rank: tensorRow.rank, device: tensorRow.device },
          [{ message: "gpu tensor parameter fact from provider tensor row" }],
        );
        continue;
      }
    }
    const primitive = lifecycle.host.facts.get(typeNode, sourcePrimitiveFactKey);
    const scalarType = primitive === undefined ? undefined : gpuScalarTypeForSourcePrimitive(primitive.kind);
    if (scalarType !== undefined) {
      lifecycle.host.facts.set(parameter, gpuScalarParameterFactKey, { scalarType }, [
        { message: "gpu scalar parameter fact from source primitive" },
      ]);
    }
  }
}

function recordIntrinsicFacts(
  lifecycle: ExtensionLifecycleContext,
  kernelFunction: Node,
  intrinsicRows: readonly GpuLangIntrinsicRow[],
): void {
  const { ast } = lifecycle.compiler;
  const visit = (node: Node): void => {
    if (ast.kindName(node) === KindCallExpression) {
      const callee = Node_Expression(node);
      const identity = callee === undefined ? undefined : providerIdentityFor(lifecycle, callee);
      const row =
        identity?.memberId === undefined ? undefined : intrinsicRows.find((candidate) => candidate.memberId === identity.memberId);
      if (row !== undefined) {
        lifecycle.host.facts.set(node, gpuIntrinsicCallFactKey, { intrinsic: row.intrinsic }, [
          { message: "gpu intrinsic call fact" },
        ]);
      }
    }
    ast.forEachChild(node, (child) => {
      if (child !== undefined) {
        visit(child);
      }
    });
  };
  const body = ast.body(kernelFunction);
  if (body !== undefined) {
    visit(body);
  }
}

function gpuScalarTypeForSourcePrimitive(kind: string): GpuScalarType | undefined {
  return isGpuScalarType(kind) ? kind : undefined;
}

function providerIdentityFor(lifecycle: ExtensionLifecycleContext, reference: Node): ProviderDeclarationIdentity | undefined {
  const { checker } = lifecycle.compiler;
  const facts = lifecycle.host.facts;
  const symbol = checker.getResolvedSymbolOrNil(reference) ?? checker.getSymbolAtLocation(reference);
  if (symbol === undefined) {
    return undefined;
  }
  for (const candidate of [symbol, safeAliasedSymbol(checker, symbol)]) {
    if (candidate === undefined) {
      continue;
    }
    for (const declaration of checker.getSymbolDeclarations(candidate)) {
      if (declaration === undefined) {
        continue;
      }
      const fact = facts.get(declaration, providerVirtualDeclarationFactKey);
      if (fact !== undefined) {
        return fact as ProviderDeclarationIdentity;
      }
    }
  }
  return undefined;
}

type GpuCompilerChecker = ExtensionLifecycleContext["compiler"]["checker"];
type GpuCheckerSymbol = NonNullable<ReturnType<GpuCompilerChecker["getSymbolAtLocation"]>>;

function safeAliasedSymbol(checker: GpuCompilerChecker, symbol: GpuCheckerSymbol) {
  try {
    return checker.getAliasedSymbol(symbol);
  } catch {
    return undefined;
  }
}
