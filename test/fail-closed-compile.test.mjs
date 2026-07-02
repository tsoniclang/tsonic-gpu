import { test } from "node:test";
import assert from "node:assert/strict";
import { createFakeGpuBackend, planGpuArtifacts } from "../dist/index.js";
import { fakeCompileInput, fakeSourceFile, fakeStatement } from "./helpers/fake-compile-input.mjs";

test("non-kernel host code is ignored by the GPU target", () => {
  const input = fakeCompileInput({
    sourceFiles: [
      fakeSourceFile({
        fileName: "src/host.ts",
        text: "export const greeting = \"hello\";\n",
        statements: [fakeStatement({ pos: 0, end: 32, kindName: "KindVariableStatement" })],
      }),
    ],
  });
  const result = planGpuArtifacts(input, createFakeGpuBackend());
  assert.deepEqual(result.artifacts, []);
  assert.deepEqual(result.diagnostics, []);
});

test("explicit kernel marker fails closed until extraction exists", () => {
  const text = "export const add = kernel(function add() {});\n";
  const input = fakeCompileInput({
    sourceFiles: [
      fakeSourceFile({
        fileName: "src/kernels.ts",
        text,
        statements: [
          fakeStatement({
            pos: 0,
            end: text.length - 1,
            kindName: "KindVariableStatement",
            gpuKernelFact: { kernelName: "add" },
          }),
        ],
      }),
    ],
  });
  const result = planGpuArtifacts(input, createFakeGpuBackend());
  assert.equal(result.artifacts.length, 0);
  assert.equal(result.diagnostics.length, 1);
  const diagnostic = result.diagnostics[0];
  assert.equal(diagnostic.code, "GPU_KERNEL_EXTRACTION_UNAVAILABLE");
  assert.equal(diagnostic.category, "error");
  assert.equal(diagnostic.source, "tsonic-gpu");
  assert.ok(diagnostic.evidence.includes("target.capability=gpu.kernel.extraction"));
  assert.ok(diagnostic.evidence.includes("gpu.kernel=add"));
  assert.ok(diagnostic.evidence.includes("gpu.backend=fake"));
  assert.deepEqual(diagnostic.sourceSpan, {
    fileName: "src/kernels.ts",
    line: 1,
    column: 1,
    endLine: 1,
    endColumn: text.length,
  });
});

test("source spans convert UTF-8 byte offsets across multi-byte text and newlines", () => {
  const line1 = "const café = 1;"; // 'é' is 2 UTF-8 bytes
  const line2 = "export const k = kernel(function k() {});";
  const text = `${line1}\n${line2}\n`;
  const line1Bytes = 16; // 15 chars + 1 extra byte for 'é'
  const pos = line1Bytes + 1; // start of line 2
  const end = pos + line2.length;
  const input = fakeCompileInput({
    sourceFiles: [
      fakeSourceFile({
        fileName: "src/unicode.ts",
        text,
        statements: [
          fakeStatement({ pos, end, kindName: "KindVariableStatement", gpuKernelFact: { kernelName: "k" } }),
        ],
      }),
    ],
  });
  const result = planGpuArtifacts(input, createFakeGpuBackend());
  assert.equal(result.diagnostics.length, 1);
  assert.deepEqual(result.diagnostics[0].sourceSpan, {
    fileName: "src/unicode.ts",
    line: 2,
    column: 1,
    endLine: 2,
    endColumn: line2.length + 1,
  });
});

test("multiple kernels each fail closed with their own diagnostic", () => {
  const text = "a;\nb;\n";
  const input = fakeCompileInput({
    sourceFiles: [
      fakeSourceFile({
        fileName: "src/kernels.ts",
        text,
        statements: [
          fakeStatement({ pos: 0, end: 2, kindName: "KindVariableStatement", gpuKernelFact: { kernelName: "one" } }),
          fakeStatement({ pos: 3, end: 5, kindName: "KindVariableStatement", gpuKernelFact: { kernelName: "two" } }),
        ],
      }),
    ],
  });
  const result = planGpuArtifacts(input, createFakeGpuBackend());
  assert.equal(result.artifacts.length, 0);
  assert.equal(result.diagnostics.length, 2);
  assert.ok(result.diagnostics.every((diagnostic) => diagnostic.code === "GPU_KERNEL_EXTRACTION_UNAVAILABLE"));
});
