export type GpuShapeExpr =
  | { readonly kind: "literal"; readonly value: number }
  | { readonly kind: "symbol"; readonly name: string }
  | { readonly kind: "product"; readonly operands: readonly GpuShapeExpr[] }
  | { readonly kind: "sum"; readonly operands: readonly GpuShapeExpr[] }
  | { readonly kind: "backend-meta"; readonly name: string };

export type GpuShapeConstraint =
  | { readonly kind: "equal"; readonly left: GpuShapeExpr; readonly right: GpuShapeExpr }
  | { readonly kind: "divisible"; readonly value: GpuShapeExpr; readonly divisor: GpuShapeExpr };

export function gpuShapeSymbolNames(expr: GpuShapeExpr): readonly string[] {
  switch (expr.kind) {
    case "literal":
      return [];
    case "symbol":
      return [expr.name];
    case "backend-meta":
      return [];
    case "product":
    case "sum":
      return expr.operands.flatMap((operand) => gpuShapeSymbolNames(operand));
  }
}

export function gpuShapeBackendMetaNames(expr: GpuShapeExpr): readonly string[] {
  switch (expr.kind) {
    case "literal":
    case "symbol":
      return [];
    case "backend-meta":
      return [expr.name];
    case "product":
    case "sum":
      return expr.operands.flatMap((operand) => gpuShapeBackendMetaNames(operand));
  }
}
