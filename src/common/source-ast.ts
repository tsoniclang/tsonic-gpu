import type { AstReader, Node } from "@tsonic/tsts";

// Kind names compared against ast.kindName(node). Field access is duck-typed
// against the TS-Go AST data shapes; no internal TSTS modules are imported.

export const KindBinaryExpression = "KindBinaryExpression";
export const KindBlock = "KindBlock";
export const KindCallExpression = "KindCallExpression";
export const KindElementAccessExpression = "KindElementAccessExpression";
export const KindExpressionStatement = "KindExpressionStatement";
export const KindFalseKeyword = "KindFalseKeyword";
export const KindForStatement = "KindForStatement";
export const KindFunctionExpression = "KindFunctionExpression";
export const KindIdentifier = "KindIdentifier";
export const KindIfStatement = "KindIfStatement";
export const KindNumericLiteral = "KindNumericLiteral";
export const KindParenthesizedExpression = "KindParenthesizedExpression";
export const KindPostfixUnaryExpression = "KindPostfixUnaryExpression";
export const KindPrefixUnaryExpression = "KindPrefixUnaryExpression";
export const KindPropertyAccessExpression = "KindPropertyAccessExpression";
export const KindReturnStatement = "KindReturnStatement";
export const KindStringLiteral = "KindStringLiteral";
export const KindTrueKeyword = "KindTrueKeyword";
export const KindTypeReference = "KindTypeReference";
export const KindVariableDeclaration = "KindVariableDeclaration";
export const KindVariableDeclarationList = "KindVariableDeclarationList";
export const KindVariableStatement = "KindVariableStatement";
export const KindWhileStatement = "KindWhileStatement";
export const KindDoStatement = "KindDoStatement";

// Operator token kind names.
export const KindPlusToken = "KindPlusToken";
export const KindMinusToken = "KindMinusToken";
export const KindAsteriskToken = "KindAsteriskToken";
export const KindSlashToken = "KindSlashToken";
export const KindPercentToken = "KindPercentToken";
export const KindLessThanToken = "KindLessThanToken";
export const KindLessThanEqualsToken = "KindLessThanEqualsToken";
export const KindGreaterThanToken = "KindGreaterThanToken";
export const KindGreaterThanEqualsToken = "KindGreaterThanEqualsToken";
export const KindEqualsEqualsEqualsToken = "KindEqualsEqualsEqualsToken";
export const KindExclamationEqualsEqualsToken = "KindExclamationEqualsEqualsToken";
export const KindAmpersandAmpersandToken = "KindAmpersandAmpersandToken";
export const KindBarBarToken = "KindBarBarToken";
export const KindEqualsToken = "KindEqualsToken";
export const KindPlusPlusToken = "KindPlusPlusToken";

function nodeField(node: Node | undefined, fieldName: string): Node | undefined {
  if (node === undefined) {
    return undefined;
  }
  const value = (node as unknown as Record<string, unknown>)[fieldName];
  return typeof value === "object" && value !== null ? (value as Node) : undefined;
}

export function Node_Expression(node: Node | undefined): Node | undefined {
  return nodeField(node, "Expression");
}

export function Node_Type(node: Node | undefined): Node | undefined {
  return nodeField(node, "Type");
}

export function Node_Initializer(node: Node | undefined): Node | undefined {
  return nodeField(node, "Initializer");
}

export function BinaryExpression_Left(node: Node | undefined): Node | undefined {
  return nodeField(node, "Left");
}

export function BinaryExpression_Right(node: Node | undefined): Node | undefined {
  return nodeField(node, "Right");
}

export function BinaryExpression_OperatorToken(node: Node | undefined): Node | undefined {
  return nodeField(node, "OperatorToken");
}

export function PrefixUnaryExpression_Operand(node: Node | undefined): Node | undefined {
  return nodeField(node, "Operand");
}

export function PostfixUnaryExpression_Operand(node: Node | undefined): Node | undefined {
  return nodeField(node, "Operand");
}

export function IfStatement_ThenStatement(node: Node | undefined): Node | undefined {
  return nodeField(node, "ThenStatement");
}

export function IfStatement_ElseStatement(node: Node | undefined): Node | undefined {
  return nodeField(node, "ElseStatement");
}

export function ForStatement_Initializer(node: Node | undefined): Node | undefined {
  return nodeField(node, "Initializer");
}

export function ForStatement_Condition(node: Node | undefined): Node | undefined {
  return nodeField(node, "Condition");
}

export function ForStatement_Incrementor(node: Node | undefined): Node | undefined {
  return nodeField(node, "Incrementor");
}

export function IterationStatement_Statement(node: Node | undefined): Node | undefined {
  return nodeField(node, "Statement");
}

export function ElementAccessExpression_ArgumentExpression(node: Node | undefined): Node | undefined {
  return nodeField(node, "ArgumentExpression");
}

export function TypeReferenceNode_TypeName(node: Node | undefined): Node | undefined {
  return nodeField(node, "TypeName");
}

export function VariableStatement_DeclarationList(node: Node | undefined): Node | undefined {
  return nodeField(node, "DeclarationList");
}

export function Node_Flags(node: Node | undefined): number {
  const value = (node as unknown as { readonly Flags?: unknown } | undefined)?.Flags;
  return typeof value === "number" ? value : 0;
}

// TS-Go NodeFlags: NodeFlagsLet = 1 << 0, NodeFlagsConst = 1 << 1, matching
// upstream TypeScript NodeFlags for variable declaration lists.
const nodeFlagsConst = 2;

export function isConstVariableDeclarationList(node: Node | undefined): boolean {
  return (Node_Flags(node) & nodeFlagsConst) !== 0;
}

// Unary expressions expose their operator as a raw numeric Kind, which the
// public AstReader cannot name. Follow the reference target packs: read the
// operator text from the source span between the node and its operand.
export function getPrefixUnaryOperatorText(ast: AstReader, node: Node): string | undefined {
  const operand = PrefixUnaryExpression_Operand(node);
  const sourceText = ast.getSourceText(ast.getSourceFile(node));
  const start = ast.pos(node);
  const end = operand === undefined ? ast.end(node) : ast.pos(operand);
  const prefixText = start < 0 || end < start ? "" : sourceText.slice(start, end).trimStart();
  for (const operator of ["++", "--", "!", "-", "+"]) {
    if (prefixText.startsWith(operator)) {
      return operator;
    }
  }
  return undefined;
}

export function getPostfixUnaryOperatorText(ast: AstReader, node: Node): string | undefined {
  const operand = PostfixUnaryExpression_Operand(node);
  const sourceText = ast.getSourceText(ast.getSourceFile(node));
  const start = operand === undefined ? ast.pos(node) : ast.end(operand);
  const end = ast.end(node);
  const postfixText = start < 0 || end < start ? "" : sourceText.slice(start, end).trimStart();
  for (const operator of ["++", "--"]) {
    if (postfixText.startsWith(operator)) {
      return operator;
    }
  }
  return undefined;
}

// The AST normalizes numeric literal text (2.0 becomes 2), so dtype
// classification must read the literal as written in the source.
export function numericLiteralSourceText(ast: AstReader, node: Node): string {
  const sourceText = ast.getSourceText(ast.getSourceFile(node));
  const slice = sourceText.slice(ast.pos(node), ast.end(node)).trim();
  return slice.length > 0 ? slice : ast.text(node);
}
