import { createExtensionConsumerQueries, runtimeCarrierFactKey } from "@tsonic/tsts";
import type { CompilerSession, ExtensionHost, Node, SourceFile } from "@tsonic/tsts";
import type {
  TargetCarrierResolution,
  TargetCompilationPaths,
  TargetCompileInput,
  TargetRuntimeReference,
  TargetSelection,
  TsonicProjectConfig,
} from "@tsonic/target-api";

export interface GpuCompileInputOptions {
  readonly session: CompilerSession;
  readonly extensionHost: ExtensionHost;
  readonly project: TsonicProjectConfig;
  readonly target: TargetSelection;
  readonly runtimeReferences?: readonly TargetRuntimeReference[];
  readonly paths?: TargetCompilationPaths;
  readonly consumerName?: string;
}

// Session-to-compile-input bridge for GPU-owned integration tests and
// tooling. When the Tsonic host drives the GPU target it supplies its own
// TargetCompileInput; this bridge only wires the queries the GPU backend
// actually consumes, from public TSTS APIs.
export function createGpuCompileInputFromSession(options: GpuCompileInputOptions): TargetCompileInput {
  const { session, extensionHost } = options;
  const ast = session.ast;
  const checker = session.checker;
  const sourceFiles = session
    .getSourceFiles()
    .filter((sourceFile): sourceFile is SourceFile => sourceFile !== undefined && !ast.getFileName(sourceFile).endsWith(".d.ts"));
  const facts = createExtensionConsumerQueries(extensionHost, options.consumerName ?? "tsonic-gpu-backend");

  const resolveRuntimeCarrier = (subject: object | undefined): TargetCarrierResolution => {
    if (subject === undefined) {
      return { kind: "missing", reason: "No subject provided for carrier resolution.", evidence: [] };
    }
    const fact = extensionHost.facts.get(subject, runtimeCarrierFactKey) ?? extensionHost.factResolver.resolve(subject, runtimeCarrierFactKey);
    return fact === undefined
      ? { kind: "missing", reason: "No finalized GPU runtime carrier fact.", evidence: [] }
      : { kind: "resolved", carrier: fact.carrier, evidence: [] };
  };

  const analysis = {
    getSymbolName: (subject: object | undefined) => (subject === undefined ? undefined : checker.getSymbolName(subject as never)),
    getProjectSourceReferenceForNode: (node: Node | undefined) => {
      if (node === undefined) {
        return undefined;
      }
      const symbol = checker.getResolvedSymbolOrNil(node) ?? checker.getSymbolAtLocation(node);
      if (symbol === undefined) {
        return undefined;
      }
      const declaration = checker.getSymbolValueDeclaration(symbol) ?? checker.getPrimarySymbolDeclaration(symbol);
      if (declaration === undefined) {
        return undefined;
      }
      const declarationSourceFile = ast.getSourceFile(declaration);
      if (declarationSourceFile === undefined || ast.getFileName(declarationSourceFile).endsWith(".d.ts")) {
        return undefined;
      }
      return { symbol, declaration, sourceFile: declarationSourceFile };
    },
  };

  const targetFacts = {
    resolveRuntimeCarrier,
    resolveRuntimeCarrierForNode: (node: object | undefined) => resolveRuntimeCarrier(node),
    getTargetBinding: () => undefined,
    getTargetBindingForReference: () => undefined,
    resolveCallReturnRuntimeCarrier: (node: object | undefined) => resolveRuntimeCarrier(node),
    resolveCallParameterRuntimeCarriers: () => ({
      kind: "missing" as const,
      reason: "Call parameter carrier resolution is not provided by the GPU test bridge.",
      evidence: [],
    }),
    resolveDeclarationReturnCarrier: (node: object | undefined) => resolveRuntimeCarrier(node),
  };

  const paths: TargetCompilationPaths = options.paths ?? {
    projectFilePath: "tsonic.json",
    projectRoot: ".",
    outputRoot: "out",
    targetOutputRoot: "out/gpu",
  };

  const input = {
    program: session.program,
    ast,
    types: session.types,
    sourceFiles,
    facts,
    analysis,
    targetFacts,
    project: options.project,
    target: options.target,
    runtimeReferences: options.runtimeReferences ?? [],
    paths,
  };
  return input as unknown as TargetCompileInput;
}
