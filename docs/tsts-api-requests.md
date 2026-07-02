# TSTS Public API Requests

Needs this target pack has for `@tsonic/tsts` public helpers. Each entry names
the workaround currently in place; remove the workaround when the helper
lands. Do not add further raw TS-Go constants or span-text readers — extend
this list instead.

## Variable declaration list flags

- Need: a public `AstReader` helper such as `isConstVariableDeclarationList(node)`
  (or `declarationListFlavor(node): "const" | "let" | "var" | "using"`).
- Workaround: `src/common/source-ast.ts` reads the duck-typed `Flags` field
  against `NodeFlagsConst = 2` (matches TS-Go `NodeFlagsConst = 1 << 1`).

## Unary operator kinds

- Need: a public way to name the operator of prefix/postfix unary expressions,
  for example `AstReader.operatorKindName(node)`; the `Operator` field is a raw
  numeric `Kind` that public API cannot name.
- Workaround: `src/common/source-ast.ts` reads the operator text from the
  source span between the node and its operand (`getPrefixUnaryOperatorText`,
  `getPostfixUnaryOperatorText`), following the reference target packs.

## Numeric literal source text

- Need: public access to a numeric literal's text as written (the AST
  normalizes `2.0` to `2`), for example `AstReader.literalSourceText(node)`.
- Workaround: `numericLiteralSourceText` in `src/common/source-ast.ts` slices
  the source span.
