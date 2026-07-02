import type { TargetDiagnostic } from "@tsonic/target-api";
import { gpuIrDiagnostic } from "./diagnostics.js";
import type { GpuIrBlock, GpuIrFunction, GpuIrModule, GpuKernelParameter, GpuSourceSpan } from "./ir.js";
import { isGpuScalarType } from "./scalar-types.js";
import { gpuShapeBackendMetaNames, gpuShapeSymbolNames, type GpuShapeExpr } from "./shape.js";
import { isGpuDeviceDomain, type GpuTensorType } from "./tensor.js";

export function validateGpuIrModule(module: GpuIrModule): readonly TargetDiagnostic[] {
  const diagnostics: TargetDiagnostic[] = [];
  const report = (
    code: string,
    capabilityId: string,
    message: string,
    kernelName?: string,
    span?: GpuSourceSpan,
  ): void => {
    diagnostics.push(
      gpuIrDiagnostic({
        code,
        capabilityId,
        message,
        moduleName: module.name,
        ...(kernelName === undefined ? {} : { kernelName }),
        ...(span === undefined ? {} : { span }),
      }),
    );
  };

  if (module.name.length === 0) {
    report("GPU_INVALID_IR", "gpu.ir.module-name", "GPU IR module name must be a non-empty string.");
  }
  const kernelNames = new Set<string>();
  for (const kernel of module.kernels) {
    if (kernel.name.length === 0) {
      report("GPU_INVALID_IR", "gpu.ir.kernel-name", "GPU IR kernel name must be a non-empty string.");
      continue;
    }
    if (kernelNames.has(kernel.name)) {
      report("GPU_INVALID_IR", "gpu.ir.kernel-name", `GPU IR kernel name '${kernel.name}' is declared more than once.`, kernel.name, kernel.span);
      continue;
    }
    kernelNames.add(kernel.name);
    validateKernel(kernel, report);
  }
  return diagnostics;
}

type ReportFn = (code: string, capabilityId: string, message: string, kernelName?: string, span?: GpuSourceSpan) => void;

function validateKernel(kernel: GpuIrFunction, report: ReportFn): void {
  const kernelReport: ReportFn = (code, capabilityId, message, _kernelName, span) => {
    report(code, capabilityId, message, kernel.name, span ?? kernel.span);
  };

  const tensorParameters = new Map<string, Extract<GpuKernelParameter, { kind: "tensor" }>>();
  const definedValues = new Set<string>();
  const parameterNames = new Set<string>();

  for (const parameter of kernel.parameters) {
    if (parameter.name.length === 0) {
      kernelReport("GPU_INVALID_IR", "gpu.ir.parameter-name", "GPU kernel parameter names must be non-empty strings.");
      continue;
    }
    if (parameterNames.has(parameter.name)) {
      kernelReport("GPU_INVALID_IR", "gpu.ir.parameter-name", `GPU kernel parameter '${parameter.name}' is declared more than once.`);
      continue;
    }
    parameterNames.add(parameter.name);
    if (parameter.kind === "tensor") {
      validateTensorParameter(parameter.name, parameter.role, parameter.tensor, kernelReport);
      tensorParameters.set(parameter.name, parameter);
      for (const dimension of parameter.tensor.shape) {
        for (const symbol of gpuShapeSymbolNames(dimension)) {
          definedValues.add(symbol);
        }
      }
    } else {
      if (!isGpuScalarType(parameter.scalarType)) {
        kernelReport(
          "GPU_INVALID_IR",
          "gpu.ir.scalar.dtype",
          `GPU kernel parameter '${parameter.name}' has unknown scalar dtype '${String(parameter.scalarType)}'.`,
        );
      }
      definedValues.add(parameter.name);
    }
  }

  validateDeviceConsistency(kernel, tensorParameters, kernelReport);

  const metaParameterNames = new Set(kernel.launch.metaParameters ?? []);
  for (const metaParameter of metaParameterNames) {
    definedValues.add(metaParameter);
  }
  validateLaunchPlan(kernel, definedValues, metaParameterNames, kernelReport);
  validateEffects(kernel, tensorParameters, parameterNames, kernelReport);
  validateBlock(kernel.body, definedValues, tensorParameters, kernelReport, new Map(), 0);
}

function validateTensorParameter(
  name: string,
  role: "input" | "output" | "inout",
  tensor: GpuTensorType,
  report: ReportFn,
): void {
  if (!isGpuScalarType(tensor.elementType)) {
    report("GPU_INVALID_IR", "gpu.ir.tensor.dtype", `GPU tensor parameter '${name}' has unknown element dtype '${String(tensor.elementType)}'.`);
  }
  if (!Number.isInteger(tensor.rank) || tensor.rank < 0) {
    report("GPU_INVALID_IR", "gpu.ir.tensor.rank", `GPU tensor parameter '${name}' has invalid rank '${String(tensor.rank)}'.`);
    return;
  }
  if (tensor.shape.length !== tensor.rank) {
    report(
      "GPU_INVALID_IR",
      "gpu.ir.tensor.shape",
      `GPU tensor parameter '${name}' declares rank ${tensor.rank} but its shape has ${tensor.shape.length} dimensions.`,
    );
  }
  for (const dimension of tensor.shape) {
    validateShapeDimension(name, dimension, report);
  }
  if (tensor.strides !== undefined && tensor.strides.length !== tensor.rank) {
    report(
      "GPU_INVALID_IR",
      "gpu.ir.tensor.strides",
      `GPU tensor parameter '${name}' declares rank ${tensor.rank} but its strides have ${tensor.strides.length} entries.`,
    );
  }
  if (!isGpuDeviceDomain(tensor.device.domain)) {
    report("GPU_INVALID_IR", "gpu.ir.tensor.device", `GPU tensor parameter '${name}' has unknown device domain '${String(tensor.device.domain)}'.`);
  }
  if (role === "input" && tensor.mutability !== "readonly") {
    report("GPU_INVALID_IR", "gpu.ir.tensor.mutability", `GPU tensor parameter '${name}' has role 'input' and must be readonly.`);
  }
  if ((role === "output" || role === "inout") && tensor.mutability !== "mutable") {
    report("GPU_INVALID_IR", "gpu.ir.tensor.mutability", `GPU tensor parameter '${name}' has role '${role}' and must be mutable.`);
  }
}

function validateShapeDimension(parameterName: string, dimension: GpuShapeExpr, report: ReportFn): void {
  if (dimension.kind === "literal") {
    if (!Number.isInteger(dimension.value) || dimension.value <= 0) {
      report(
        "GPU_INVALID_IR",
        "gpu.ir.tensor.shape",
        `GPU tensor parameter '${parameterName}' has non-positive literal dimension '${String(dimension.value)}'.`,
      );
    }
    return;
  }
  if (dimension.kind === "product" || dimension.kind === "sum") {
    for (const operand of dimension.operands) {
      validateShapeDimension(parameterName, operand, report);
    }
  }
}

function validateDeviceConsistency(
  kernel: GpuIrFunction,
  tensorParameters: ReadonlyMap<string, Extract<GpuKernelParameter, { kind: "tensor" }>>,
  report: ReportFn,
): void {
  const domains = new Set<string>();
  for (const parameter of tensorParameters.values()) {
    if (isGpuDeviceDomain(parameter.tensor.device.domain)) {
      domains.add(parameter.tensor.device.domain);
    }
  }
  if (domains.size > 1) {
    report(
      "GPU_DEVICE_MISMATCH",
      "gpu.device.mixed",
      `GPU kernel '${kernel.name}' mixes tensor device domains: ${[...domains].sort().join(", ")}. All kernel tensors must share one device domain.`,
    );
  }
}

function validateLaunchPlan(
  kernel: GpuIrFunction,
  definedValues: ReadonlySet<string>,
  metaParameterNames: ReadonlySet<string>,
  report: ReportFn,
): void {
  const { launch } = kernel;
  if (launch.grid.length < 1 || launch.grid.length > 3) {
    report("GPU_INVALID_IR", "gpu.ir.launch.grid", `GPU kernel '${kernel.name}' launch grid must have 1 to 3 dimensions, found ${launch.grid.length}.`);
  }
  if (launch.block !== undefined && (launch.block.length < 1 || launch.block.length > 3)) {
    report("GPU_INVALID_IR", "gpu.ir.launch.block", `GPU kernel '${kernel.name}' launch block must have 1 to 3 dimensions, found ${launch.block.length}.`);
  }
  const launchExpressions = [...launch.grid, ...(launch.block ?? [])];
  for (const expression of launchExpressions) {
    for (const symbol of gpuShapeSymbolNames(expression)) {
      if (!definedValues.has(symbol)) {
        report(
          "GPU_INVALID_IR",
          "gpu.ir.launch.symbol",
          `GPU kernel '${kernel.name}' launch plan references unknown shape symbol '${symbol}'.`,
        );
      }
    }
    for (const metaName of gpuShapeBackendMetaNames(expression)) {
      if (!metaParameterNames.has(metaName)) {
        report(
          "GPU_INVALID_IR",
          "gpu.ir.launch.meta",
          `GPU kernel '${kernel.name}' launch plan references undeclared backend meta parameter '${metaName}'.`,
        );
      }
    }
  }
}

function validateEffects(
  kernel: GpuIrFunction,
  tensorParameters: ReadonlyMap<string, Extract<GpuKernelParameter, { kind: "tensor" }>>,
  parameterNames: ReadonlySet<string>,
  report: ReportFn,
): void {
  for (const effect of kernel.effects) {
    if (effect.kind === "barrier") {
      continue;
    }
    if (!parameterNames.has(effect.parameter)) {
      report("GPU_INVALID_IR", "gpu.ir.effect", `GPU kernel '${kernel.name}' declares a '${effect.kind}' effect on unknown parameter '${effect.parameter}'.`);
      continue;
    }
    if (effect.kind === "write" || effect.kind === "atomic") {
      const tensorParameter = tensorParameters.get(effect.parameter);
      if (tensorParameter === undefined || tensorParameter.tensor.mutability !== "mutable") {
        report(
          "GPU_INVALID_IR",
          "gpu.ir.effect",
          `GPU kernel '${kernel.name}' declares a '${effect.kind}' effect on '${effect.parameter}', which is not a mutable tensor parameter.`,
        );
      }
    }
  }
  for (const constraint of kernel.aliasingConstraints ?? []) {
    for (const parameter of constraint.parameters) {
      if (!tensorParameters.has(parameter)) {
        report("GPU_INVALID_IR", "gpu.ir.aliasing", `GPU kernel '${kernel.name}' declares an aliasing constraint on unknown tensor parameter '${parameter}'.`);
      }
    }
  }
}

function validateBlock(
  block: GpuIrBlock,
  outerDefined: ReadonlySet<string>,
  tensorParameters: ReadonlyMap<string, Extract<GpuKernelParameter, { kind: "tensor" }>>,
  report: ReportFn,
  outerMutables: ReadonlyMap<string, number>,
  loopDepth: number,
): void {
  const defined = new Set(outerDefined);
  const mutables = new Map(outerMutables);

  const requireValue = (name: string, description: string, span?: GpuSourceSpan): void => {
    if (!defined.has(name)) {
      report("GPU_INVALID_IR", "gpu.ir.value-ref", `GPU IR references undefined value '${name}' as ${description}.`, undefined, span);
    }
  };

  const defineResult = (name: string, span?: GpuSourceSpan): void => {
    if (name.length === 0) {
      report("GPU_INVALID_IR", "gpu.ir.value-def", "GPU IR operation results must be non-empty value names.", undefined, span);
      return;
    }
    if (defined.has(name) || tensorParameters.has(name)) {
      report("GPU_INVALID_IR", "gpu.ir.value-def", `GPU IR value '${name}' is defined more than once.`, undefined, span);
      return;
    }
    defined.add(name);
  };

  const requireTensor = (
    name: string,
    span: GpuSourceSpan | undefined,
    mutation: boolean,
  ): Extract<GpuKernelParameter, { kind: "tensor" }> | undefined => {
    const tensorParameter = tensorParameters.get(name);
    if (tensorParameter === undefined) {
      report("GPU_INVALID_IR", "gpu.ir.tensor-ref", `GPU IR references unknown tensor parameter '${name}'.`, undefined, span);
      return undefined;
    }
    if (mutation && tensorParameter.tensor.mutability !== "mutable") {
      report("GPU_INVALID_IR", "gpu.ir.store.mutability", `GPU IR writes to readonly tensor parameter '${name}'.`, undefined, span);
    }
    return tensorParameter;
  };

  const requireIndices = (
    tensorParameter: Extract<GpuKernelParameter, { kind: "tensor" }>,
    indices: readonly string[],
    span?: GpuSourceSpan,
  ): void => {
    if (indices.length !== tensorParameter.tensor.rank) {
      report(
        "GPU_INVALID_IR",
        "gpu.ir.index-arity",
        `GPU IR indexes tensor parameter '${tensorParameter.name}' of rank ${tensorParameter.tensor.rank} with ${indices.length} indices.`,
        undefined,
        span,
      );
    }
    for (const index of indices) {
      requireValue(index, `an index into tensor parameter '${tensorParameter.name}'`, span);
    }
  };

  for (const operation of block.operations) {
    switch (operation.kind) {
      case "const": {
        if (!isGpuScalarType(operation.dtype)) {
          report("GPU_INVALID_IR", "gpu.ir.scalar.dtype", `GPU IR constant has unknown dtype '${String(operation.dtype)}'.`, undefined, operation.span);
        }
        defineResult(operation.result, operation.span);
        break;
      }
      case "thread-index": {
        if (!Number.isInteger(operation.dimension) || operation.dimension < 0 || operation.dimension > 2) {
          report(
            "GPU_INVALID_IR",
            "gpu.ir.thread-index",
            `GPU IR thread-index dimension must be 0, 1, or 2, found '${String(operation.dimension)}'.`,
            undefined,
            operation.span,
          );
        }
        defineResult(operation.result, operation.span);
        break;
      }
      case "binary": {
        requireValue(operation.left, `the left operand of '${operation.operator}'`, operation.span);
        requireValue(operation.right, `the right operand of '${operation.operator}'`, operation.span);
        defineResult(operation.result, operation.span);
        break;
      }
      case "unary": {
        requireValue(operation.operand, `the operand of '${operation.operator}'`, operation.span);
        defineResult(operation.result, operation.span);
        break;
      }
      case "load": {
        const tensorParameter = requireTensor(operation.tensor, operation.span, false);
        if (tensorParameter !== undefined) {
          requireIndices(tensorParameter, operation.indices, operation.span);
          if (operation.dtype !== tensorParameter.tensor.elementType) {
            report(
              "GPU_INVALID_IR",
              "gpu.ir.load.dtype",
              `GPU IR loads '${operation.dtype}' from tensor parameter '${operation.tensor}' whose element dtype is '${tensorParameter.tensor.elementType}'.`,
              undefined,
              operation.span,
            );
          }
        }
        if (operation.mask !== undefined) {
          requireValue(operation.mask, "a load mask", operation.span);
        }
        defineResult(operation.result, operation.span);
        break;
      }
      case "store": {
        const tensorParameter = requireTensor(operation.tensor, operation.span, true);
        if (tensorParameter !== undefined) {
          requireIndices(tensorParameter, operation.indices, operation.span);
        }
        requireValue(operation.value, `the value stored to tensor parameter '${operation.tensor}'`, operation.span);
        if (operation.mask !== undefined) {
          requireValue(operation.mask, "a store mask", operation.span);
        }
        break;
      }
      case "local": {
        if (!isGpuScalarType(operation.dtype)) {
          report("GPU_INVALID_IR", "gpu.ir.scalar.dtype", `GPU IR local has unknown dtype '${String(operation.dtype)}'.`, undefined, operation.span);
        }
        requireValue(operation.initial, `the initial value of local '${operation.result}'`, operation.span);
        defineResult(operation.result, operation.span);
        mutables.set(operation.result, loopDepth);
        break;
      }
      case "assign": {
        const declarationDepth = mutables.get(operation.target);
        if (declarationDepth === undefined) {
          report(
            "GPU_INVALID_IR",
            "gpu.ir.assign",
            `GPU IR assigns to '${operation.target}', which is not a mutable local.`,
            undefined,
            operation.span,
          );
        } else if (loopDepth <= declarationDepth) {
          report(
            "GPU_INVALID_IR",
            "gpu.ir.assign",
            `GPU IR assigns to local '${operation.target}' outside a loop nested deeper than its declaration; mutable locals are loop-carried accumulators.`,
            undefined,
            operation.span,
          );
        }
        requireValue(operation.value, `the value assigned to '${operation.target}'`, operation.span);
        break;
      }
      case "if": {
        requireValue(operation.condition, "a conditional guard", operation.span);
        validateBlock(operation.then, defined, tensorParameters, report, mutables, loopDepth);
        if (operation.else !== undefined) {
          validateBlock(operation.else, defined, tensorParameters, report, mutables, loopDepth);
        }
        break;
      }
      case "loop": {
        requireValue(operation.lowerBound, "a loop lower bound", operation.span);
        requireValue(operation.upperBound, "a loop upper bound", operation.span);
        if (operation.step !== undefined) {
          requireValue(operation.step, "a loop step", operation.span);
        }
        if (defined.has(operation.counter) || tensorParameters.has(operation.counter)) {
          report("GPU_INVALID_IR", "gpu.ir.value-def", `GPU IR loop counter '${operation.counter}' shadows an existing value.`, undefined, operation.span);
          validateBlock(operation.body, defined, tensorParameters, report, mutables, loopDepth + 1);
          break;
        }
        const bodyScope = new Set(defined);
        bodyScope.add(operation.counter);
        validateBlock(operation.body, bodyScope, tensorParameters, report, mutables, loopDepth + 1);
        break;
      }
      case "reduce": {
        if (!defined.has(operation.operand) && !tensorParameters.has(operation.operand)) {
          report(
            "GPU_INVALID_IR",
            "gpu.ir.value-ref",
            `GPU IR reduce references '${operation.operand}', which is neither a defined value nor a tensor parameter.`,
            undefined,
            operation.span,
          );
        }
        defineResult(operation.result, operation.span);
        break;
      }
      case "intrinsic": {
        for (const operand of operation.operands) {
          requireValue(operand, `an operand of intrinsic '${operation.name}'`, operation.span);
        }
        defineResult(operation.result, operation.span);
        break;
      }
      case "barrier": {
        break;
      }
      case "atomic": {
        const tensorParameter = requireTensor(operation.tensor, operation.span, true);
        if (tensorParameter !== undefined) {
          requireIndices(tensorParameter, operation.indices, operation.span);
        }
        requireValue(operation.value, `the value of atomic '${operation.operator}'`, operation.span);
        if (operation.result !== undefined) {
          defineResult(operation.result, operation.span);
        }
        break;
      }
      case "return": {
        break;
      }
    }
  }
}
