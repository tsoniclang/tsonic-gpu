import type { GpuEffect, GpuAliasingConstraint } from "./effects.js";
import type { GpuLaunchPlan } from "./launch.js";
import type { GpuScalarType } from "./scalar-types.js";
import type { GpuShapeConstraint } from "./shape.js";
import type { GpuTensorType } from "./tensor.js";

export interface GpuSourceSpan {
  readonly fileName: string;
  readonly line: number;
  readonly column: number;
  readonly endLine: number;
  readonly endColumn: number;
}

export type GpuTensorParameterRole = "input" | "output" | "inout";

export type GpuScalarParameterRole = "scalar" | "shape" | "meta";

export type GpuKernelParameter =
  | {
      readonly kind: "tensor";
      readonly name: string;
      readonly role: GpuTensorParameterRole;
      readonly tensor: GpuTensorType;
      readonly span?: GpuSourceSpan;
    }
  | {
      readonly kind: "scalar";
      readonly name: string;
      readonly role: GpuScalarParameterRole;
      readonly scalarType: GpuScalarType;
      readonly span?: GpuSourceSpan;
    };

export type GpuBinaryOperator =
  | "add"
  | "sub"
  | "mul"
  | "div"
  | "mod"
  | "min"
  | "max"
  | "and"
  | "or"
  | "xor"
  | "shift-left"
  | "shift-right"
  | "eq"
  | "ne"
  | "lt"
  | "le"
  | "gt"
  | "ge";

export type GpuUnaryOperator = "neg" | "not" | "abs";

export type GpuReduceOperator = "sum" | "prod" | "min" | "max";

export type GpuAtomicOperator = "add" | "min" | "max" | "exchange";

export type GpuMathIntrinsic =
  | "sqrt"
  | "rsqrt"
  | "exp"
  | "log"
  | "sin"
  | "cos"
  | "tanh"
  | "erf"
  | "floor"
  | "ceil"
  | "round"
  | "pow"
  | "fma";

export type GpuThreadIndexSpace = "global" | "local" | "block";

interface GpuIrOperationBase {
  readonly span?: GpuSourceSpan;
}

export type GpuIrOperation =
  | (GpuIrOperationBase & {
      readonly kind: "const";
      readonly result: string;
      readonly dtype: GpuScalarType;
      readonly value: number | boolean;
    })
  | (GpuIrOperationBase & {
      readonly kind: "thread-index";
      readonly result: string;
      readonly space: GpuThreadIndexSpace;
      readonly dimension: number;
    })
  | (GpuIrOperationBase & {
      readonly kind: "binary";
      readonly result: string;
      readonly operator: GpuBinaryOperator;
      readonly left: string;
      readonly right: string;
      readonly dtype: GpuScalarType;
    })
  | (GpuIrOperationBase & {
      readonly kind: "unary";
      readonly result: string;
      readonly operator: GpuUnaryOperator;
      readonly operand: string;
      readonly dtype: GpuScalarType;
    })
  | (GpuIrOperationBase & {
      readonly kind: "load";
      readonly result: string;
      readonly tensor: string;
      readonly indices: readonly string[];
      readonly dtype: GpuScalarType;
      readonly mask?: string;
    })
  | (GpuIrOperationBase & {
      readonly kind: "store";
      readonly tensor: string;
      readonly indices: readonly string[];
      readonly value: string;
      readonly mask?: string;
    })
  | (GpuIrOperationBase & {
      readonly kind: "if";
      readonly condition: string;
      readonly then: GpuIrBlock;
      readonly else?: GpuIrBlock;
    })
  | (GpuIrOperationBase & {
      readonly kind: "loop";
      readonly counter: string;
      readonly lowerBound: string;
      readonly upperBound: string;
      readonly step?: string;
      readonly body: GpuIrBlock;
    })
  | (GpuIrOperationBase & {
      readonly kind: "reduce";
      readonly result: string;
      readonly operator: GpuReduceOperator;
      readonly operand: string;
      readonly scope: "block";
      readonly dtype: GpuScalarType;
    })
  | (GpuIrOperationBase & {
      readonly kind: "intrinsic";
      readonly result: string;
      readonly name: GpuMathIntrinsic;
      readonly operands: readonly string[];
      readonly dtype: GpuScalarType;
    })
  | (GpuIrOperationBase & {
      readonly kind: "barrier";
      readonly scope: "block";
    })
  | (GpuIrOperationBase & {
      readonly kind: "atomic";
      readonly result?: string;
      readonly operator: GpuAtomicOperator;
      readonly tensor: string;
      readonly indices: readonly string[];
      readonly value: string;
      readonly dtype: GpuScalarType;
    })
  | (GpuIrOperationBase & {
      readonly kind: "return";
    });

export interface GpuIrBlock {
  readonly operations: readonly GpuIrOperation[];
}

export interface GpuIrFunction {
  readonly name: string;
  readonly span?: GpuSourceSpan;
  readonly parameters: readonly GpuKernelParameter[];
  readonly launch: GpuLaunchPlan;
  readonly effects: readonly GpuEffect[];
  readonly aliasingConstraints?: readonly GpuAliasingConstraint[];
  readonly shapeConstraints?: readonly GpuShapeConstraint[];
  readonly body: GpuIrBlock;
}

export interface GpuIrModule {
  readonly name: string;
  readonly kernels: readonly GpuIrFunction[];
}
