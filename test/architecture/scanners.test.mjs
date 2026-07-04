import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const sourceRoot = join(repositoryRoot, "src");

function collectFiles(root, extension) {
  const results = [];
  const visit = (directory) => {
    for (const entry of readdirSync(directory)) {
      const fullPath = join(directory, entry);
      if (statSync(fullPath).isDirectory()) {
        visit(fullPath);
        continue;
      }
      if (fullPath.endsWith(extension)) {
        results.push(fullPath);
      }
    }
  };
  visit(root);
  return results;
}

const sourceFiles = collectFiles(sourceRoot, ".ts").map((path) => ({
  path,
  text: readFileSync(path, "utf8"),
}));

test("no internal TSTS imports", () => {
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, /from "@tsonic\/tsts\/.+"/u, `${path} imports a deep tsts path`);
    assert.doesNotMatch(text, /dist\/src\/internal/u, `${path} references tsts internals`);
  }
});

test("no concrete GPU backend or host library names in GPU core", () => {
  const banned = /triton|pytorch|\btorch\b|numpy|python|pyproject/iu;
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, banned, `${path} references a concrete backend or host library`);
  }
});

test("no cross-target references in GPU core", () => {
  const banned = /csharp|roslyn|dotnet|\bcargo\b|rustc/iu;
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, banned, `${path} references another target family`);
  }
});

test("no source-name recasing in GPU core", () => {
  // Provider/library/source-visible names are never auto-recased; kernel and
  // parameter names flow into facts and IR verbatim.
  const banned = /toUpperCase\(|toLowerCase\(|camelcase|snakecase|pascalcase/iu;
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, banned, `${path} recases names`);
  }
});

test("no CPU-recovery semantics in GPU core", () => {
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, /fallback/iu, `${path} mentions a recovery lane; GPU compilation must fail closed`);
  }
});

test("no product dependency on analysis files", () => {
  for (const { path, text } of sourceFiles) {
    assert.ok(!text.includes(".analysis/") && !text.includes('".analysis"'), `${path} references .analysis`);
  }
});

test("no runtime reflection or dynamic evaluation in GPU core", () => {
  const banned = /Reflect\.|\beval\(|new Function\(/u;
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, banned, `${path} uses runtime reflection or dynamic evaluation`);
  }
});

test("no embedded JS engine or runtime interpretation dependencies", () => {
  const packageJson = readFileSync(join(repositoryRoot, "package.json"), "utf8");
  const banned = /quickjs|rquickjs|boa_engine|deno_core|"v8"/iu;
  assert.doesNotMatch(packageJson, banned);
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, banned, `${path} references an embedded JS engine`);
  }
});

test("no fallback source emission: backend diagnostics never coexist with artifacts", () => {
  const plannerText = readFileSync(join(sourceRoot, "backend/planner/gpu-planner.ts"), "utf8");
  assert.match(plannerText, /if \(diagnostics\.length > 0\) \{\s*return \{ artifacts: \[\], diagnostics \};/u);
});

test("no product backend is bundled by default", async () => {
  const { createGpuTargetPack } = await import("../../dist/index.js");
  const pack = createGpuTargetPack();
  const target = { id: "gpu", options: { backendId: "fake", hostTargetId: "python" } };
  assert.throws(
    () => pack.createBackend({ project: { entryPoint: "src/index.ts", targets: [target] }, target }),
    /not registered/u,
    "backends must be handed in explicitly; none may ship pre-registered",
  );
});

test("the GPU pack declares no surfaces or bundled provider packages", async () => {
  const { createGpuTargetPack } = await import("../../dist/index.js");
  const pack = createGpuTargetPack();
  assert.deepEqual(pack.surfaces ?? [], []);
  assert.deepEqual(pack.packages ?? [], []);
});
