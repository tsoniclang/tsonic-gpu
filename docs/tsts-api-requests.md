# TSTS Public API Requirements

Public helpers this target pack requires from `@tsonic/tsts`. Each entry
states the requirement and the centralized local implementation this package
substitutes in `src/common/source-ast.ts`. A public helper supersedes its
local implementation, and every substitution lives in that one module — raw
TS-Go constants and span-text readers are not permitted anywhere else.

## Variable declaration list flags

- Requirement: a public `AstReader` helper such as
  `isConstVariableDeclarationList(node)` (or
  `declarationListFlavor(node): "const" | "let" | "var" | "using"`).
- Local implementation: `isConstVariableDeclarationList` reads the duck-typed
  `Flags` field against `NodeFlagsConst = 2` (TS-Go `NodeFlagsConst = 1 << 1`).

## Unary operator kinds

- Requirement: a public way to name the operator of prefix/postfix unary
  expressions, for example `AstReader.operatorKindName(node)`; the `Operator`
  field is a raw numeric `Kind` that public API cannot name.
- Local implementation: `getPrefixUnaryOperatorText` and
  `getPostfixUnaryOperatorText` read the operator text from the source span
  between the node and its operand, following the reference target packs.

## Numeric literal source text

- Requirement: public access to a numeric literal's text as written (the AST
  normalizes `2.0` to `2`), for example `AstReader.literalSourceText(node)`.
- Local implementation: `numericLiteralSourceText` slices the source span.
