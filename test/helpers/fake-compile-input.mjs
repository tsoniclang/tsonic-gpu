// Fake TargetCompileInput pieces for backend tests. Only the members the
// planner actually reads are provided; anything else missing keeps the tests
// honest about what the GPU backend consumes.

import { gpuKernelDeclarationFactKey } from "../../dist/index.js";

export function fakeSourceFile({ fileName = "src/index.ts", text = "", statements = [] } = {}) {
  return { fileName, text, statements };
}

export function fakeStatement({ pos = 0, end = 0, kindName = "ExpressionStatement", gpuKernelFact = undefined } = {}) {
  return { pos, end, kindName, gpuKernelFact };
}

export function fakeAstReader() {
  return {
    statements: (sourceFile) => sourceFile.statements ?? [],
    kindName: (node) => node.kindName,
    pos: (node) => node.pos,
    end: (node) => node.end,
    getFileName: (sourceFile) => sourceFile.fileName,
    getSourceFile: (node) => node.sourceFile ?? node,
    getSourceText: (sourceFile) => sourceFile.text ?? "",
    forEachChild: () => {},
    hasModifierKind: () => false,
    name: () => undefined,
    parameters: () => [],
    arguments: () => [],
    body: () => undefined,
    text: (node) => node.text ?? "",
  };
}

export function fakeCompileInput({
  sourceFiles = [],
  target = { id: "gpu", options: { backendId: "fake", hostTargetId: "python" } },
  runtimeReferences = [],
} = {}) {
  return {
    program: {},
    ast: fakeAstReader(),
    types: {},
    sourceFiles,
    facts: {
      getFact: (subject, key) => (key === gpuKernelDeclarationFactKey ? subject.gpuKernelFact : undefined),
      getRuntimeCarrierFact: () => undefined,
      getSelectedTargetCall: () => undefined,
    },
    analysis: {
      getSymbolName: () => undefined,
      getProjectSourceReferenceForNode: () => undefined,
    },
    targetFacts: {},
    project: { entryPoint: "src/index.ts", targets: [target] },
    target,
    runtimeReferences,
    paths: {
      projectFilePath: "tsonic.json",
      projectRoot: ".",
      outputRoot: "out",
      targetOutputRoot: "out/gpu",
    },
  };
}
