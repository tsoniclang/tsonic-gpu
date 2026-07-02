import type { AstReader, Node, SourceFile } from "@tsonic/tsts";
import type { TargetCompileInput, TargetDiagnostic } from "@tsonic/target-api";
import {
  BinaryExpression_Left,
  BinaryExpression_OperatorToken,
  BinaryExpression_Right,
  ElementAccessExpression_ArgumentExpression,
  ForStatement_Condition,
  ForStatement_Incrementor,
  ForStatement_Initializer,
  IfStatement_ElseStatement,
  IfStatement_ThenStatement,
  IterationStatement_Statement,
  KindBinaryExpression,
  KindBlock,
  KindCallExpression,
  KindDoStatement,
  KindElementAccessExpression,
  KindExpressionStatement,
  KindFalseKeyword,
  KindForStatement,
  KindFunctionExpression,
  KindIdentifier,
  KindIfStatement,
  KindNumericLiteral,
  KindParenthesizedExpression,
  KindPostfixUnaryExpression,
  KindPrefixUnaryExpression,
  KindReturnStatement,
  KindTrueKeyword,
  KindVariableDeclaration,
  KindVariableStatement,
  KindWhileStatement,
  Node_Expression,
  Node_Initializer,
  PostfixUnaryExpression_Operand,
  PrefixUnaryExpression_Operand,
  VariableStatement_DeclarationList,
} from "../../common/source-ast.js";
import type {
  GpuBinaryOperator,
  GpuIrBlock,
  GpuIrFunction,
  GpuIrOperation,
  GpuKernelParameter,
  GpuSourceSpan,
} from "../../ir/ir.js";
import type { GpuScalarType } from "../../ir/scalar-types.js";
import type { GpuEffect } from "../../ir/effects.js";
import type { GpuShapeExpr } from "../../ir/shape.js";
import {
  gpuIntrinsicCallFactKey,
  gpuKernelDeclarationFactKey,
  gpuScalarParameterFactKey,
  gpuTensorParameterFactKey,
  type GpuKernelDeclarationFact,
  type GpuTensorParameterFact,
} from "../../source/gpu-facts/keys.js";
import {
  gpuSourceSpanForNode,
  missingGpuFactDiagnostic,
  unsupportedGpuConstructDiagnostic,
} from "../planner/diagnostics.js";

export interface GpuKernelExtractionRequest {
  readonly input: TargetCompileInput;
  readonly sourceFile: SourceFile;
  readonly statement: Node;
  readonly fact: GpuKernelDeclarationFact;
}

export interface GpuKernelExtractionResult {
  readonly kernel?: GpuIrFunction;
  readonly diagnostics: readonly TargetDiagnostic[];
}

interface TensorEntry {
  readonly name: string;
  readonly fact: GpuTensorParameterFact;
}

interface ExtractionContext {
  readonly input: TargetCompileInput;
  readonly ast: AstReader;
  readonly sourceFile: SourceFile;
  readonly diagnostics: TargetDiagnostic[];
  readonly tensors: Map<string, TensorEntry>;
  tempCounter: number;
}

interface ScalarValue {
  readonly id: string;
  readonly dtype: GpuScalarType;
}

// Bindings map source names to the IR value that carries them; a plain
// aliasing binding (const j = i) reuses the aliased value id instead of
// emitting a copy operation.
type Scope = Map<string, ScalarValue>;

export function extractGpuKernel(request: GpuKernelExtractionRequest): GpuKernelExtractionResult {
  const context: ExtractionContext = {
    input: request.input,
    ast: request.input.ast,
    sourceFile: request.sourceFile,
    diagnostics: [],
    tensors: new Map(),
    tempCounter: 0,
  };
  const kernel = extractFromStatement(context, request.statement, request.fact);
  if (context.diagnostics.length > 0) {
    return { diagnostics: context.diagnostics };
  }
  return kernel === undefined ? { diagnostics: context.diagnostics } : { kernel, diagnostics: [] };
}

function reject(context: ExtractionContext, node: Node, capabilityId: string, message: string): undefined {
  context.diagnostics.push(
    unsupportedGpuConstructDiagnostic({ ast: context.ast, sourceFile: context.sourceFile, node }, capabilityId, message),
  );
  return undefined;
}

function spanFor(context: ExtractionContext, node: Node): GpuSourceSpan | undefined {
  return gpuSourceSpanForNode(context.ast, context.sourceFile, node);
}

function withSpan<T extends GpuIrOperation>(context: ExtractionContext, node: Node, operation: T): T {
  const span = spanFor(context, node);
  return span === undefined ? operation : { ...operation, span };
}

function nextTemp(context: ExtractionContext): string {
  context.tempCounter += 1;
  return `%${context.tempCounter}`;
}

function extractFromStatement(
  context: ExtractionContext,
  statement: Node,
  fact: GpuKernelDeclarationFact,
): GpuIrFunction | undefined {
  const { ast } = context;
  const kernelFunction = kernelFunctionExpression(context, statement);
  if (kernelFunction === undefined) {
    return reject(
      context,
      statement,
      "gpu.kernel.declaration-form",
      "GPU kernel declarations must assign kernel(function name(...) { ... }) to a const binding.",
    );
  }
  if (fact.kernelName.length === 0) {
    return reject(context, statement, "gpu.kernel.name", "GPU kernels need a non-empty kernel name.");
  }

  const scope: Scope = new Map();
  const parameterOrder: string[] = [];
  for (const parameter of ast.parameters(kernelFunction)) {
    if (parameter === undefined) {
      continue;
    }
    const nameNode = ast.name(parameter);
    const name = nameNode === undefined ? "" : ast.text(nameNode);
    if (name.length === 0) {
      reject(context, parameter, "gpu.kernel.parameter-name", "GPU kernel parameters must be plain named identifiers.");
      continue;
    }
    if (scope.has(name) || context.tensors.has(name)) {
      reject(context, parameter, "gpu.kernel.parameter-name", `GPU kernel parameter '${name}' is declared more than once.`);
      continue;
    }
    const tensorFact = context.input.facts.getFact(parameter, gpuTensorParameterFactKey);
    if (tensorFact !== undefined) {
      context.tensors.set(name, { name, fact: tensorFact });
      parameterOrder.push(name);
      continue;
    }
    const scalarFact = context.input.facts.getFact(parameter, gpuScalarParameterFactKey);
    if (scalarFact !== undefined) {
      scope.set(name, { id: name, dtype: scalarFact.scalarType });
      parameterOrder.push(name);
      continue;
    }
    context.diagnostics.push(
      missingGpuFactDiagnostic(
        { ast, sourceFile: context.sourceFile, node: parameter },
        "gpu.kernel.parameter-type",
        `GPU kernel parameter '${name}' has no tensor or scalar target fact; device parameters must be provider tensors or source primitives.`,
      ),
    );
  }

  const body = ast.body(kernelFunction);
  if (body === undefined || ast.kindName(body) !== KindBlock) {
    return reject(context, kernelFunction, "gpu.kernel.declaration-form", "GPU kernels need a block body.");
  }
  const operations: GpuIrOperation[] = [];
  buildBlockStatements(context, body, scope, operations);
  if (context.diagnostics.length > 0) {
    return undefined;
  }

  const reads = new Set<string>();
  const writes = new Set<string>();
  collectTensorAccess({ operations }, reads, writes);

  const writtenTensors = parameterOrder.filter((name) => writes.has(name));
  if (writtenTensors.length === 0) {
    return reject(context, statement, "gpu.kernel.no-output", `GPU kernel '${fact.kernelName}' writes no tensor output.`);
  }
  const launchTensorName = writtenTensors[0];
  const launchTensor = launchTensorName === undefined ? undefined : context.tensors.get(launchTensorName);
  if (launchTensor === undefined || launchTensor.fact.rank < 1 || launchTensor.fact.rank > 3) {
    return reject(
      context,
      statement,
      "gpu.kernel.launch",
      `GPU kernel '${fact.kernelName}' cannot derive a launch grid from its output tensor.`,
    );
  }

  const parameters: GpuKernelParameter[] = [];
  const effects: GpuEffect[] = [];
  for (const name of parameterOrder) {
    const tensorEntry = context.tensors.get(name);
    if (tensorEntry === undefined) {
      const scalarValue = scope.get(name);
      if (scalarValue !== undefined) {
        parameters.push({ kind: "scalar", name, role: "scalar", scalarType: scalarValue.dtype });
      }
      continue;
    }
    const written = writes.has(name);
    const read = reads.has(name);
    const role = written ? (read ? "inout" : "output") : "input";
    parameters.push({
      kind: "tensor",
      name,
      role,
      tensor: {
        elementType: tensorEntry.fact.elementType,
        rank: tensorEntry.fact.rank,
        shape: tensorShape(name, tensorEntry.fact.rank),
        layout: { kind: "contiguous" },
        device: { domain: tensorEntry.fact.device },
        mutability: written ? "mutable" : "readonly",
        aliasing: "noalias",
      },
    });
    if (read) {
      effects.push({ kind: "read", parameter: name });
    }
    if (written) {
      effects.push({ kind: "write", parameter: name });
    }
  }

  const kernelSpan = spanFor(context, statement);
  return {
    name: fact.kernelName,
    ...(kernelSpan === undefined ? {} : { span: kernelSpan }),
    parameters,
    launch: {
      grid: tensorShape(launchTensor.name, launchTensor.fact.rank),
      streamPolicy: "default",
      devicePolicy: "single-device",
    },
    effects,
    body: { operations },
  };
}

function tensorShape(parameterName: string, rank: number): readonly GpuShapeExpr[] {
  return Array.from({ length: rank }, (_, dimension) => ({ kind: "symbol", name: `${parameterName}_dim${dimension}` }));
}

function kernelFunctionExpression(context: ExtractionContext, statement: Node): Node | undefined {
  const { ast } = context;
  if (ast.kindName(statement) !== KindVariableStatement) {
    return undefined;
  }
  const declarationList = VariableStatement_DeclarationList(statement);
  if (declarationList === undefined) {
    return undefined;
  }
  let kernelFunction: Node | undefined;
  ast.forEachChild(declarationList, (declaration) => {
    if (kernelFunction !== undefined || declaration === undefined || ast.kindName(declaration) !== KindVariableDeclaration) {
      return;
    }
    const initializer = Node_Initializer(declaration);
    if (initializer === undefined || ast.kindName(initializer) !== KindCallExpression) {
      return;
    }
    const [argument] = ast.arguments(initializer);
    if (argument !== undefined && ast.kindName(argument) === KindFunctionExpression) {
      kernelFunction = argument;
    }
  });
  return kernelFunction;
}

function collectTensorAccess(block: GpuIrBlock, reads: Set<string>, writes: Set<string>): void {
  for (const operation of block.operations) {
    switch (operation.kind) {
      case "load":
        reads.add(operation.tensor);
        break;
      case "store":
        writes.add(operation.tensor);
        break;
      case "atomic":
        writes.add(operation.tensor);
        break;
      case "reduce":
        // Extraction only emits reduces over tensor parameters; names that
        // are plain values simply never match a tensor parameter.
        reads.add(operation.operand);
        break;
      case "if":
        collectTensorAccess(operation.then, reads, writes);
        if (operation.else !== undefined) {
          collectTensorAccess(operation.else, reads, writes);
        }
        break;
      case "loop":
        collectTensorAccess(operation.body, reads, writes);
        break;
      default:
        break;
    }
  }
}

function buildBlockStatements(context: ExtractionContext, block: Node, scope: Scope, operations: GpuIrOperation[]): void {
  for (const statement of context.ast.statements(block)) {
    if (statement === undefined) {
      continue;
    }
    buildStatement(context, statement, scope, operations);
  }
}

function buildStatement(context: ExtractionContext, statement: Node, scope: Scope, operations: GpuIrOperation[]): void {
  const { ast } = context;
  const kind = ast.kindName(statement);
  switch (kind) {
    case KindVariableStatement: {
      buildLocalBindings(context, statement, scope, operations);
      return;
    }
    case KindExpressionStatement: {
      buildExpressionStatement(context, statement, scope, operations);
      return;
    }
    case KindIfStatement: {
      buildIfStatement(context, statement, scope, operations);
      return;
    }
    case KindForStatement: {
      buildForStatement(context, statement, scope, operations);
      return;
    }
    case KindWhileStatement:
    case KindDoStatement: {
      reject(
        context,
        statement,
        "gpu.kernel.unbounded-loop",
        "Device code supports only counted for loops with a proven bound.",
      );
      return;
    }
    case KindReturnStatement: {
      const expression = Node_Expression(statement);
      if (expression !== undefined) {
        reject(context, statement, "gpu.kernel.return-value", "GPU kernels return void; returning a value is not supported.");
        return;
      }
      operations.push(withSpan(context, statement, { kind: "return" }));
      return;
    }
    case KindBlock: {
      buildBlockStatements(context, statement, new Map(scope), operations);
      return;
    }
    default: {
      reject(context, statement, "gpu.kernel.statement", "This statement form is not supported in device code.");
      return;
    }
  }
}

function buildLocalBindings(context: ExtractionContext, statement: Node, scope: Scope, operations: GpuIrOperation[]): void {
  const { ast } = context;
  const declarationList = VariableStatement_DeclarationList(statement);
  if (declarationList === undefined) {
    reject(context, statement, "gpu.kernel.statement", "This variable statement form is not supported in device code.");
    return;
  }
  ast.forEachChild(declarationList, (declaration) => {
    if (declaration === undefined || ast.kindName(declaration) !== KindVariableDeclaration) {
      return;
    }
    const nameNode = ast.name(declaration);
    const name = nameNode === undefined ? "" : ast.text(nameNode);
    if (name.length === 0) {
      reject(context, declaration, "gpu.kernel.binding", "Device bindings must be plain named identifiers.");
      return;
    }
    if (scope.has(name) || context.tensors.has(name)) {
      reject(context, declaration, "gpu.kernel.binding", `Device binding '${name}' shadows an existing kernel value.`);
      return;
    }
    const initializer = Node_Initializer(declaration);
    if (initializer === undefined) {
      reject(context, declaration, "gpu.kernel.binding", `Device binding '${name}' needs an initializer.`);
      return;
    }
    const value = buildExpression(context, initializer, scope, operations, name);
    if (value === undefined) {
      return;
    }
    scope.set(name, value);
  });
}

function buildExpressionStatement(
  context: ExtractionContext,
  statement: Node,
  scope: Scope,
  operations: GpuIrOperation[],
): void {
  const { ast } = context;
  const expression = Node_Expression(statement);
  if (expression === undefined) {
    reject(context, statement, "gpu.kernel.statement", "Empty expression statements are not supported in device code.");
    return;
  }
  if (ast.kindName(expression) === KindBinaryExpression) {
    const operatorToken = BinaryExpression_OperatorToken(expression);
    if (operatorToken !== undefined && ast.kindName(operatorToken) === "KindEqualsToken") {
      buildStore(context, expression, scope, operations);
      return;
    }
  }
  if (ast.kindName(expression) === KindCallExpression) {
    buildExpression(context, expression, scope, operations);
    return;
  }
  reject(context, statement, "gpu.kernel.statement", "Only tensor element stores and intrinsic calls can stand alone in device code.");
}

function buildStore(context: ExtractionContext, assignment: Node, scope: Scope, operations: GpuIrOperation[]): void {
  const { ast } = context;
  const target = BinaryExpression_Left(assignment);
  const valueExpression = BinaryExpression_Right(assignment);
  if (target === undefined || valueExpression === undefined) {
    reject(context, assignment, "gpu.kernel.assignment", "This assignment form is not supported in device code.");
    return;
  }
  if (ast.kindName(target) !== KindElementAccessExpression) {
    reject(
      context,
      target,
      "gpu.kernel.mutable-local",
      "Only tensor elements can be assigned in device code; local bindings are immutable in the initial GPU subset.",
    );
    return;
  }
  const access = resolveTensorElementAccess(context, target, scope, operations);
  if (access === undefined) {
    return;
  }
  const value = buildExpression(context, valueExpression, scope, operations);
  if (value === undefined) {
    return;
  }
  if (value.dtype !== access.tensor.fact.elementType) {
    reject(
      context,
      assignment,
      "gpu.kernel.store-dtype",
      `Device code stores '${value.dtype}' into tensor '${access.tensor.name}' whose element dtype is '${access.tensor.fact.elementType}'.`,
    );
    return;
  }
  operations.push(
    withSpan(context, assignment, {
      kind: "store",
      tensor: access.tensor.name,
      indices: [access.index.id],
      value: value.id,
    }),
  );
}

interface TensorElementAccess {
  readonly tensor: TensorEntry;
  readonly index: ScalarValue;
}

function resolveTensorElementAccess(
  context: ExtractionContext,
  access: Node,
  scope: Scope,
  operations: GpuIrOperation[],
): TensorElementAccess | undefined {
  const { ast } = context;
  const target = Node_Expression(access);
  if (target === undefined || ast.kindName(target) !== KindIdentifier) {
    reject(context, access, "gpu.kernel.indexing", "Device code can only index tensor parameters directly.");
    return undefined;
  }
  const tensor = context.tensors.get(ast.text(target));
  if (tensor === undefined) {
    reject(context, target, "gpu.kernel.indexing", `'${ast.text(target)}' is not a tensor parameter of this kernel.`);
    return undefined;
  }
  if (tensor.fact.rank !== 1) {
    reject(
      context,
      access,
      "gpu.kernel.index-arity",
      `Tensor parameter '${tensor.name}' has rank ${tensor.fact.rank}; single-index element access needs rank 1.`,
    );
    return undefined;
  }
  const indexExpression = ElementAccessExpression_ArgumentExpression(access);
  if (indexExpression === undefined) {
    reject(context, access, "gpu.kernel.indexing", "Tensor element access needs an index expression.");
    return undefined;
  }
  const index = buildExpression(context, indexExpression, scope, operations);
  if (index === undefined) {
    return undefined;
  }
  if (index.dtype !== "int32") {
    reject(context, indexExpression, "gpu.kernel.index-dtype", `Tensor indices must be int32 values; found '${index.dtype}'.`);
    return undefined;
  }
  return { tensor, index };
}

function buildIfStatement(context: ExtractionContext, statement: Node, scope: Scope, operations: GpuIrOperation[]): void {
  const condition = Node_Expression(statement);
  if (condition === undefined) {
    reject(context, statement, "gpu.kernel.condition", "Device conditionals need a guard expression.");
    return;
  }
  const guard = buildExpression(context, condition, scope, operations);
  if (guard === undefined) {
    return;
  }
  if (guard.dtype !== "bool") {
    reject(context, condition, "gpu.kernel.condition", `Device conditionals need a bool guard; found '${guard.dtype}'.`);
    return;
  }
  const thenStatement = IfStatement_ThenStatement(statement);
  if (thenStatement === undefined) {
    reject(context, statement, "gpu.kernel.condition", "Device conditionals need a then branch.");
    return;
  }
  const thenOperations: GpuIrOperation[] = [];
  buildBlockOrStatement(context, thenStatement, new Map(scope), thenOperations);
  const elseStatement = IfStatement_ElseStatement(statement);
  let elseBlock: GpuIrBlock | undefined;
  if (elseStatement !== undefined) {
    const elseOperations: GpuIrOperation[] = [];
    buildBlockOrStatement(context, elseStatement, new Map(scope), elseOperations);
    elseBlock = { operations: elseOperations };
  }
  operations.push(
    withSpan(context, statement, {
      kind: "if",
      condition: guard.id,
      then: { operations: thenOperations },
      ...(elseBlock === undefined ? {} : { else: elseBlock }),
    }),
  );
}

function buildBlockOrStatement(context: ExtractionContext, node: Node, scope: Scope, operations: GpuIrOperation[]): void {
  if (context.ast.kindName(node) === KindBlock) {
    buildBlockStatements(context, node, scope, operations);
    return;
  }
  buildStatement(context, node, scope, operations);
}

function buildForStatement(context: ExtractionContext, statement: Node, scope: Scope, operations: GpuIrOperation[]): void {
  const { ast } = context;
  const initializer = ForStatement_Initializer(statement);
  const condition = ForStatement_Condition(statement);
  const incrementor = ForStatement_Incrementor(statement);
  if (condition === undefined) {
    reject(context, statement, "gpu.kernel.unbounded-loop", "Device loops need a proven bound; unbounded loops are not supported.");
    return;
  }
  if (initializer === undefined || incrementor === undefined) {
    reject(context, statement, "gpu.kernel.loop-form", "Device loops must have the form for (let k = start; k < bound; k++).");
    return;
  }

  let counterName = "";
  let lowerBound: ScalarValue | undefined;
  ast.forEachChild(initializer, (declaration) => {
    if (declaration === undefined || ast.kindName(declaration) !== KindVariableDeclaration || counterName.length > 0) {
      return;
    }
    const nameNode = ast.name(declaration);
    counterName = nameNode === undefined ? "" : ast.text(nameNode);
    const start = Node_Initializer(declaration);
    if (start !== undefined) {
      lowerBound = buildExpression(context, start, scope, operations);
    }
  });
  if (counterName.length === 0 || lowerBound === undefined) {
    if (context.diagnostics.length === 0) {
      reject(context, statement, "gpu.kernel.loop-form", "Device loops must declare a single counter with an initializer.");
    }
    return;
  }
  if (lowerBound.dtype !== "int32") {
    reject(context, initializer, "gpu.kernel.loop-form", `Device loop counters must be int32 values; found '${lowerBound.dtype}'.`);
    return;
  }
  if (scope.has(counterName) || context.tensors.has(counterName)) {
    reject(context, initializer, "gpu.kernel.binding", `Device loop counter '${counterName}' shadows an existing kernel value.`);
    return;
  }

  const conditionKind = ast.kindName(condition);
  const conditionLeft = BinaryExpression_Left(condition);
  const conditionOperator = BinaryExpression_OperatorToken(condition);
  const conditionRight = BinaryExpression_Right(condition);
  const conditionIsCounterBound =
    conditionKind === KindBinaryExpression &&
    conditionLeft !== undefined &&
    ast.kindName(conditionLeft) === KindIdentifier &&
    ast.text(conditionLeft) === counterName &&
    conditionOperator !== undefined &&
    ast.kindName(conditionOperator) === "KindLessThanToken" &&
    conditionRight !== undefined;
  if (!conditionIsCounterBound) {
    reject(context, condition, "gpu.kernel.loop-form", "Device loop bounds must compare the counter with '<' against a bound value.");
    return;
  }
  const upperBound = buildExpression(context, conditionRight, scope, operations);
  if (upperBound === undefined) {
    return;
  }
  if (upperBound.dtype !== "int32") {
    reject(context, conditionRight, "gpu.kernel.loop-form", `Device loop bounds must be int32 values; found '${upperBound.dtype}'.`);
    return;
  }

  if (!isCounterIncrement(context, incrementor, counterName)) {
    reject(context, incrementor, "gpu.kernel.loop-form", "Device loops must step their counter with '++'.");
    return;
  }

  const bodyStatement = IterationStatement_Statement(statement);
  if (bodyStatement === undefined) {
    reject(context, statement, "gpu.kernel.loop-form", "Device loops need a body.");
    return;
  }
  const bodyScope = new Map(scope);
  bodyScope.set(counterName, { id: counterName, dtype: "int32" });
  const bodyOperations: GpuIrOperation[] = [];
  buildBlockOrStatement(context, bodyStatement, bodyScope, bodyOperations);
  operations.push(
    withSpan(context, statement, {
      kind: "loop",
      counter: counterName,
      lowerBound: lowerBound.id,
      upperBound: upperBound.id,
      body: { operations: bodyOperations },
    }),
  );
}

function isCounterIncrement(context: ExtractionContext, incrementor: Node, counterName: string): boolean {
  const { ast } = context;
  const kind = ast.kindName(incrementor);
  const operand =
    kind === KindPostfixUnaryExpression
      ? PostfixUnaryExpression_Operand(incrementor)
      : kind === KindPrefixUnaryExpression
        ? PrefixUnaryExpression_Operand(incrementor)
        : undefined;
  if (operand === undefined || ast.kindName(operand) !== KindIdentifier || ast.text(operand) !== counterName) {
    return false;
  }
  const sourceText = ast.getSourceText(context.sourceFile);
  const text = sourceText.slice(ast.pos(incrementor), ast.end(incrementor));
  return text.includes("++");
}

const binaryOperatorByTokenKind: ReadonlyMap<string, { operator: GpuBinaryOperator; family: "arithmetic" | "comparison" | "logical" }> =
  new Map([
    ["KindPlusToken", { operator: "add", family: "arithmetic" }],
    ["KindMinusToken", { operator: "sub", family: "arithmetic" }],
    ["KindAsteriskToken", { operator: "mul", family: "arithmetic" }],
    ["KindSlashToken", { operator: "div", family: "arithmetic" }],
    ["KindPercentToken", { operator: "mod", family: "arithmetic" }],
    ["KindLessThanToken", { operator: "lt", family: "comparison" }],
    ["KindLessThanEqualsToken", { operator: "le", family: "comparison" }],
    ["KindGreaterThanToken", { operator: "gt", family: "comparison" }],
    ["KindGreaterThanEqualsToken", { operator: "ge", family: "comparison" }],
    ["KindEqualsEqualsEqualsToken", { operator: "eq", family: "comparison" }],
    ["KindExclamationEqualsEqualsToken", { operator: "ne", family: "comparison" }],
    ["KindAmpersandAmpersandToken", { operator: "and", family: "logical" }],
    ["KindBarBarToken", { operator: "or", family: "logical" }],
  ]);

function buildExpression(
  context: ExtractionContext,
  expression: Node,
  scope: Scope,
  operations: GpuIrOperation[],
  preferredName?: string,
): ScalarValue | undefined {
  const { ast } = context;
  const kind = ast.kindName(expression);
  switch (kind) {
    case KindNumericLiteral: {
      // The AST normalizes literal text (2.0 becomes 2), so classify the
      // literal from its raw source slice: a decimal point or exponent marks
      // a float32 constant, everything else is int32.
      const sourceSlice = ast.getSourceText(context.sourceFile).slice(ast.pos(expression), ast.end(expression)).trim();
      const literalText = sourceSlice.length > 0 ? sourceSlice : ast.text(expression);
      const dtype: GpuScalarType = /[.eE]/u.test(literalText) ? "float32" : "int32";
      const id = preferredName ?? nextTemp(context);
      operations.push(withSpan(context, expression, { kind: "const", result: id, dtype, value: Number(ast.text(expression)) }));
      return { id, dtype };
    }
    case KindTrueKeyword:
    case KindFalseKeyword: {
      const id = preferredName ?? nextTemp(context);
      operations.push(withSpan(context, expression, { kind: "const", result: id, dtype: "bool", value: kind === KindTrueKeyword }));
      return { id, dtype: "bool" };
    }
    case KindIdentifier: {
      const name = ast.text(expression);
      const value = scope.get(name);
      if (value !== undefined) {
        return value;
      }
      if (context.tensors.has(name)) {
        return reject(
          context,
          expression,
          "gpu.kernel.tensor-value",
          `Tensor parameter '${name}' can only be used through element access in this GPU subset.`,
        );
      }
      return reject(
        context,
        expression,
        "gpu.kernel.host-capture",
        `'${name}' is not defined inside the kernel; device code cannot capture host values.`,
      );
    }
    case KindParenthesizedExpression: {
      const inner = Node_Expression(expression);
      if (inner === undefined) {
        return reject(context, expression, "gpu.kernel.expression", "Empty parenthesized expressions are not supported.");
      }
      return buildExpression(context, inner, scope, operations, preferredName);
    }
    case KindElementAccessExpression: {
      const access = resolveTensorElementAccess(context, expression, scope, operations);
      if (access === undefined) {
        return undefined;
      }
      const id = preferredName ?? nextTemp(context);
      operations.push(
        withSpan(context, expression, {
          kind: "load",
          result: id,
          tensor: access.tensor.name,
          indices: [access.index.id],
          dtype: access.tensor.fact.elementType,
        }),
      );
      return { id, dtype: access.tensor.fact.elementType };
    }
    case KindBinaryExpression: {
      return buildBinaryExpression(context, expression, scope, operations, preferredName);
    }
    case KindPrefixUnaryExpression: {
      return buildPrefixUnaryExpression(context, expression, scope, operations, preferredName);
    }
    case KindCallExpression: {
      return buildCallExpression(context, expression, scope, operations, preferredName);
    }
    default: {
      return reject(context, expression, "gpu.kernel.expression", "This expression form is not supported in device code.");
    }
  }
}

function buildBinaryExpression(
  context: ExtractionContext,
  expression: Node,
  scope: Scope,
  operations: GpuIrOperation[],
  preferredName?: string,
): ScalarValue | undefined {
  const { ast } = context;
  const operatorToken = BinaryExpression_OperatorToken(expression);
  const operatorKind = operatorToken === undefined ? "" : ast.kindName(operatorToken);
  if (operatorKind === "KindEqualsToken") {
    return reject(context, expression, "gpu.kernel.assignment", "Assignments are statements in device code, not expressions.");
  }
  const mapping = binaryOperatorByTokenKind.get(operatorKind);
  if (mapping === undefined) {
    return reject(context, expression, "gpu.kernel.operator", `Operator '${operatorKind}' is not supported in device code.`);
  }
  const leftExpression = BinaryExpression_Left(expression);
  const rightExpression = BinaryExpression_Right(expression);
  if (leftExpression === undefined || rightExpression === undefined) {
    return reject(context, expression, "gpu.kernel.operator", "This operator form is not supported in device code.");
  }
  const left = buildExpression(context, leftExpression, scope, operations);
  const right = buildExpression(context, rightExpression, scope, operations);
  if (left === undefined || right === undefined) {
    return undefined;
  }
  if (left.dtype !== right.dtype) {
    return reject(
      context,
      expression,
      "gpu.kernel.mixed-dtype",
      `Device operators need matching operand dtypes; found '${left.dtype}' and '${right.dtype}'.`,
    );
  }
  if (mapping.family === "arithmetic" && left.dtype === "bool") {
    return reject(context, expression, "gpu.kernel.operator", "Arithmetic operators need numeric operands in device code.");
  }
  if (mapping.family === "logical" && left.dtype !== "bool") {
    return reject(context, expression, "gpu.kernel.operator", "Logical operators need bool operands in device code.");
  }
  const id = preferredName ?? nextTemp(context);
  operations.push(
    withSpan(context, expression, {
      kind: "binary",
      result: id,
      operator: mapping.operator,
      left: left.id,
      right: right.id,
      dtype: left.dtype,
    }),
  );
  return { id, dtype: mapping.family === "arithmetic" ? left.dtype : "bool" };
}

function buildPrefixUnaryExpression(
  context: ExtractionContext,
  expression: Node,
  scope: Scope,
  operations: GpuIrOperation[],
  preferredName?: string,
): ScalarValue | undefined {
  const { ast } = context;
  const operand = PrefixUnaryExpression_Operand(expression);
  if (operand === undefined) {
    return reject(context, expression, "gpu.kernel.operator", "This unary operator form is not supported in device code.");
  }
  const sourceText = ast.getSourceText(context.sourceFile);
  const prefixText = sourceText.slice(ast.pos(expression), ast.pos(operand)).trimStart();
  const value = buildExpression(context, operand, scope, operations);
  if (value === undefined) {
    return undefined;
  }
  if (prefixText.startsWith("-")) {
    if (value.dtype === "bool") {
      return reject(context, expression, "gpu.kernel.operator", "Negation needs a numeric operand in device code.");
    }
    const id = preferredName ?? nextTemp(context);
    operations.push(withSpan(context, expression, { kind: "unary", result: id, operator: "neg", operand: value.id, dtype: value.dtype }));
    return { id, dtype: value.dtype };
  }
  if (prefixText.startsWith("!")) {
    if (value.dtype !== "bool") {
      return reject(context, expression, "gpu.kernel.operator", "Logical not needs a bool operand in device code.");
    }
    const id = preferredName ?? nextTemp(context);
    operations.push(withSpan(context, expression, { kind: "unary", result: id, operator: "not", operand: value.id, dtype: "bool" }));
    return { id, dtype: "bool" };
  }
  return reject(context, expression, "gpu.kernel.operator", "This unary operator is not supported in device code.");
}

function buildCallExpression(
  context: ExtractionContext,
  expression: Node,
  scope: Scope,
  operations: GpuIrOperation[],
  preferredName?: string,
): ScalarValue | undefined {
  const { ast } = context;
  const intrinsicFact = context.input.facts.getFact(expression, gpuIntrinsicCallFactKey);
  if (intrinsicFact === undefined) {
    return reject(
      context,
      expression,
      "gpu.device.host-call",
      "Device code cannot call host values; only GPU intrinsics are callable inside kernels.",
    );
  }
  const callArguments = ast.arguments(expression).filter((argument): argument is Node => argument !== undefined);
  if (intrinsicFact.intrinsic.kind === "thread-index") {
    const [dimensionArgument] = callArguments;
    const dimensionText = dimensionArgument === undefined ? "" : ast.text(dimensionArgument);
    const dimension = Number(dimensionText);
    if (
      callArguments.length !== 1 ||
      dimensionArgument === undefined ||
      ast.kindName(dimensionArgument) !== KindNumericLiteral ||
      !Number.isInteger(dimension) ||
      dimension < 0 ||
      dimension > 2
    ) {
      return reject(
        context,
        expression,
        "gpu.kernel.thread-index",
        "Thread index intrinsics need a literal dimension of 0, 1, or 2.",
      );
    }
    const id = preferredName ?? nextTemp(context);
    operations.push(
      withSpan(context, expression, { kind: "thread-index", result: id, space: intrinsicFact.intrinsic.space, dimension }),
    );
    return { id, dtype: "int32" };
  }
  if (intrinsicFact.intrinsic.kind === "block-reduce") {
    const [tensorArgument] = callArguments;
    if (callArguments.length !== 1 || tensorArgument === undefined || ast.kindName(tensorArgument) !== KindIdentifier) {
      return reject(
        context,
        expression,
        "gpu.kernel.reduce-operand",
        "Block reduce intrinsics take exactly one tensor parameter as their operand.",
      );
    }
    const tensor = context.tensors.get(ast.text(tensorArgument));
    if (tensor === undefined) {
      return reject(
        context,
        tensorArgument,
        "gpu.kernel.reduce-operand",
        `'${ast.text(tensorArgument)}' is not a tensor parameter of this kernel.`,
      );
    }
    if (tensor.fact.elementType !== intrinsicFact.intrinsic.dtype) {
      return reject(
        context,
        tensorArgument,
        "gpu.kernel.mixed-dtype",
        `Block reduce over '${intrinsicFact.intrinsic.dtype}' cannot consume tensor '${tensor.name}' with element dtype '${tensor.fact.elementType}'.`,
      );
    }
    const id = preferredName ?? nextTemp(context);
    operations.push(
      withSpan(context, expression, {
        kind: "reduce",
        result: id,
        operator: intrinsicFact.intrinsic.operator,
        operand: tensor.name,
        scope: "block",
        dtype: intrinsicFact.intrinsic.dtype,
      }),
    );
    return { id, dtype: intrinsicFact.intrinsic.dtype };
  }
  const operands: string[] = [];
  for (const argument of callArguments) {
    const value = buildExpression(context, argument, scope, operations);
    if (value === undefined) {
      return undefined;
    }
    if (value.dtype !== intrinsicFact.intrinsic.dtype) {
      return reject(
        context,
        argument,
        "gpu.kernel.mixed-dtype",
        `Intrinsic '${intrinsicFact.intrinsic.name}' needs '${intrinsicFact.intrinsic.dtype}' operands; found '${value.dtype}'.`,
      );
    }
    operands.push(value.id);
  }
  if (operands.length === 0) {
    return reject(context, expression, "gpu.kernel.expression", "Math intrinsics need at least one operand.");
  }
  const id = preferredName ?? nextTemp(context);
  operations.push(
    withSpan(context, expression, {
      kind: "intrinsic",
      result: id,
      name: intrinsicFact.intrinsic.name,
      operands,
      dtype: intrinsicFact.intrinsic.dtype,
    }),
  );
  return { id, dtype: intrinsicFact.intrinsic.dtype };
}

export { gpuKernelDeclarationFactKey };
