// End-to-end test harness: compiles in-memory TypeScript through TSTS with
// the GPU target extensions, then plans GPU artifacts via the backend.
// Uses only public @tsonic packages — no @tsonic/host.
import { createCompilerSessionFromFiles, formatDiagnostics } from "@tsonic/tsts";
import { createTsonicCoreSourceExtension } from "@tsonic/source-core";
import {
  createFakeGpuBackend,
  createGpuCompileInputFromSession,
  createGpuProviderPackage,
  createGpuTargetPack,
} from "../../dist/index.js";

export function acmeTensorPackage() {
  const float32Indexer = {
    id: "@acme/tensor::Float32Tensor.indexer",
    name: "indexer",
    kind: "indexer",
    signatures: [
      {
        id: "@acme/tensor::Float32Tensor.indexer(index)",
        parameters: [{ name: "index", type: { kind: "number" } }],
        returnType: { kind: "source-primitive", name: "float32" },
      },
    ],
  };
  return createGpuProviderPackage({
    id: "acme-tensor",
    displayName: "Acme tensors",
    version: "1.0.0",
    modules: [
      {
        moduleSpecifier: "@acme/tensor",
        providerModuleId: "acme.tensor",
        exports: [
          {
            id: "@acme/tensor::Float32Tensor",
            name: "Float32Tensor",
            kind: "class",
            members: [float32Indexer],
          },
          {
            id: "@acme/tensor::Int32Tensor",
            name: "Int32Tensor",
            kind: "class",
            members: [
              {
                id: "@acme/tensor::Int32Tensor.indexer",
                name: "indexer",
                kind: "indexer",
                signatures: [
                  {
                    id: "@acme/tensor::Int32Tensor.indexer(index)",
                    parameters: [{ name: "index", type: { kind: "number" } }],
                    returnType: { kind: "source-primitive", name: "int32" },
                  },
                ],
              },
            ],
          },
          {
            id: "@acme/tensor::Float64Tensor",
            name: "Float64Tensor",
            kind: "class",
            members: [
              {
                id: "@acme/tensor::Float64Tensor.indexer",
                name: "indexer",
                kind: "indexer",
                signatures: [
                  {
                    id: "@acme/tensor::Float64Tensor.indexer(index)",
                    parameters: [{ name: "index", type: { kind: "number" } }],
                    returnType: { kind: "source-primitive", name: "float64" },
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
    tensorTypes: [
      { exportId: "@acme/tensor::Float32Tensor", elementType: "float32", rank: 1, device: "cuda" },
      { exportId: "@acme/tensor::Int32Tensor", elementType: "int32", rank: 1, device: "cuda" },
      { exportId: "@acme/tensor::Float64Tensor", elementType: "float64", rank: 1, device: "cuda" },
    ],
  });
}

export const defaultGpuTarget = Object.freeze({
  id: "gpu",
  options: Object.freeze({ backendId: "fake", hostTargetId: "python" }),
});

export function createGpuSession({ files, target = defaultGpuTarget, packages = [], entryPoint = "index.ts" } = {}) {
  const pack = createGpuTargetPack({ backends: [createFakeGpuBackend()] });
  const project = { entryPoint, targets: [target] };
  const providerContext = {
    project,
    target,
    targetPack: pack,
    selectedSurfaces: [],
    selectedPackages: packages,
  };
  const fileMap = new Map(Object.entries(files).map(([name, text]) => [`/src/${name}`, text]));
  const session = createCompilerSessionFromFiles({
    currentDirectory: "/src",
    files: fileMap,
    compilerOptions: {
      module: "esnext",
      moduleResolution: "bundler",
      strictNullChecks: true,
      target: "es2022",
    },
    extensionHostOptions: {
      activeTarget: "gpu",
      extensions: [
        createTsonicCoreSourceExtension(),
        ...pack.provider.createExtensions(providerContext),
        ...packages.flatMap((providerPackage) => providerPackage.createExtensions?.({ ...providerContext, package: providerPackage }) ?? []),
      ],
    },
  });
  return { session, pack, project, target, providerContext };
}

export function checkGpuSession(harness, fileNames) {
  const { session } = harness;
  const checked =
    fileNames ??
    [...session.getSourceFiles()]
      .filter((sourceFile) => sourceFile !== undefined)
      .map((sourceFile) => session.ast.getFileName(sourceFile))
      .filter((fileName) => fileName.startsWith("/src/"));
  for (const fileName of checked) {
    const diagnostics = formatDiagnostics(session.ensureChecked(session.getSourceFile(fileName)));
    if (diagnostics !== "") {
      throw new Error(`TypeScript diagnostics for ${fileName}:\n${diagnostics}`);
    }
  }
  return session.finalizeExtensions();
}

export function compileGpu({ files, target = defaultGpuTarget, packages = [], entryPoint = "index.ts" }) {
  const resolvedPackages = packages.length === 0 ? [acmeTensorPackage()] : packages;
  const harness = createGpuSession({ files, target, packages: resolvedPackages, entryPoint });
  const extensionHost = checkGpuSession(harness);
  const input = createGpuCompileInputFromSession({
    session: harness.session,
    extensionHost,
    project: harness.project,
    target,
  });
  const backend = harness.pack.createBackend({ project: harness.project, target });
  return { result: backend.compile(input), extensionHost, harness };
}

export function artifactText(result, path) {
  const artifact = result.artifacts.find((candidate) => candidate.path === path);
  if (artifact === undefined) {
    throw new Error(`Missing artifact '${path}'. Present: ${result.artifacts.map((candidate) => candidate.path).join(", ")}`);
  }
  return artifact.text;
}

export function capabilityIds(diagnostics) {
  return diagnostics.flatMap((diagnostic) =>
    (diagnostic.evidence ?? [])
      .filter((row) => row.startsWith("target.capability="))
      .map((row) => row.slice("target.capability=".length)),
  );
}
