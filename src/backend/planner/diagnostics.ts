import type { AstReader, Node, SourceFile } from "@tsonic/tsts";
import type { TargetDiagnostic, TargetDiagnosticSourceSpan } from "@tsonic/target-api";

export interface GpuDiagnosticInput {
  readonly ast: AstReader;
  readonly sourceFile: SourceFile;
  readonly node: Node;
}

export function unsupportedGpuConstructDiagnostic(
  input: GpuDiagnosticInput,
  capabilityId: string,
  message: string,
): TargetDiagnostic {
  return baseGpuDiagnostic(input, "GPU_UNSUPPORTED_KERNEL_OPERATION", capabilityId, message);
}

export function missingGpuFactDiagnostic(
  input: GpuDiagnosticInput,
  capabilityId: string,
  message: string,
): TargetDiagnostic {
  return baseGpuDiagnostic(input, "GPU_MISSING_TARGET_FACT", capabilityId, message);
}

export function kernelExtractionUnavailableDiagnostic(
  input: GpuDiagnosticInput,
  kernelName: string,
  backendId: string,
): TargetDiagnostic {
  const base = baseGpuDiagnostic(
    input,
    "GPU_KERNEL_EXTRACTION_UNAVAILABLE",
    "gpu.kernel.extraction",
    `GPU kernel '${kernelName}' cannot be compiled yet: kernel extraction is not implemented. The GPU target fails closed instead of guessing.`,
  );
  return {
    ...base,
    evidence: [...(base.evidence ?? []), `gpu.kernel=${kernelName}`, `gpu.backend=${backendId}`],
  };
}

function baseGpuDiagnostic(
  input: GpuDiagnosticInput,
  code: string,
  capabilityId: string,
  message: string,
): TargetDiagnostic {
  const { ast, sourceFile, node } = input;
  const fileName = ast.getFileName(sourceFile);
  const text = ast.getSourceText(sourceFile);
  const pos = ast.pos(node);
  const end = ast.end(node);
  const sourceSpan = structuredSourceSpan(fileName, text, pos, end);
  const evidence = [
    `target.capability=${capabilityId}`,
    ...(fileName.length === 0 ? [] : [`source.module=${fileName}`, `source.file=${fileName}`]),
    `source.byteSpan=${pos}-${end}`,
    ...(sourceSpan === undefined
      ? []
      : [`source.span=${sourceSpan.line}:${sourceSpan.column}-${sourceSpan.endLine}:${sourceSpan.endColumn}`]),
  ];
  return {
    code,
    category: "error",
    source: "tsonic-gpu",
    message: `${message} Node kind: ${ast.kindName(node)}.`,
    ...(sourceSpan === undefined ? {} : { sourceSpan }),
    evidence,
  };
}

function structuredSourceSpan(
  fileName: string,
  text: string,
  pos: number,
  end: number,
): TargetDiagnosticSourceSpan | undefined {
  if (fileName.length === 0 || pos < 0 || end < pos) {
    return undefined;
  }
  const start = sourceLocationFromByteOffset(text, pos);
  const stop = sourceLocationFromByteOffset(text, end);
  if (start === undefined || stop === undefined) {
    return undefined;
  }
  return {
    fileName,
    line: start.line,
    column: start.column,
    endLine: stop.line,
    endColumn: stop.column,
  };
}

interface SourceLocation {
  readonly line: number;
  readonly column: number;
}

// AST positions are UTF-8 byte offsets; convert by scanning code points so
// multi-byte characters, CRLF, and Unicode line separators stay accurate.
function sourceLocationFromByteOffset(text: string, byteOffset: number): SourceLocation | undefined {
  let bytes = 0;
  let line = 1;
  let column = 1;
  let index = 0;
  while (index < text.length) {
    if (bytes === byteOffset) {
      return { line, column };
    }
    if (bytes > byteOffset) {
      return undefined;
    }
    const codePoint = text.codePointAt(index);
    if (codePoint === undefined) {
      return undefined;
    }
    let charLength = codePoint > 0xffff ? 2 : 1;
    bytes += utf8ByteLengthOfCodePoint(codePoint);
    if (codePoint === 0x0d) {
      if (text.charCodeAt(index + 1) === 0x0a) {
        bytes += 1;
        charLength += 1;
      }
      line += 1;
      column = 1;
    } else if (codePoint === 0x0a || codePoint === 0x2028 || codePoint === 0x2029) {
      line += 1;
      column = 1;
    } else {
      column += 1;
    }
    index += charLength;
  }
  return bytes === byteOffset ? { line, column } : undefined;
}

function utf8ByteLengthOfCodePoint(codePoint: number): number {
  if (codePoint <= 0x7f) {
    return 1;
  }
  if (codePoint <= 0x7ff) {
    return 2;
  }
  if (codePoint <= 0xffff) {
    return 3;
  }
  return 4;
}
