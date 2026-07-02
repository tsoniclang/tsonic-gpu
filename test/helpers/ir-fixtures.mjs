// Hand-authored GPU IR fixtures shared across G2 tests.

export function tensorParameter(name, role, { dtype = "float32", rank = 1, shape, device = "cuda", mutability } = {}) {
  const resolvedMutability = mutability ?? (role === "input" ? "readonly" : "mutable");
  const resolvedShape = shape ?? Array.from({ length: rank }, () => ({ kind: "symbol", name: "N" }));
  return {
    kind: "tensor",
    name,
    role,
    tensor: {
      elementType: dtype,
      rank,
      shape: resolvedShape,
      layout: { kind: "contiguous" },
      device: { domain: device },
      mutability: resolvedMutability,
      aliasing: "noalias",
    },
  };
}

export function vectorAddModule() {
  return {
    name: "vector_add_module",
    kernels: [
      {
        name: "add",
        parameters: [
          tensorParameter("a", "input"),
          tensorParameter("b", "input"),
          tensorParameter("out", "output"),
        ],
        launch: {
          grid: [{ kind: "symbol", name: "N" }],
          streamPolicy: "default",
          devicePolicy: "single-device",
        },
        effects: [
          { kind: "read", parameter: "a" },
          { kind: "read", parameter: "b" },
          { kind: "write", parameter: "out" },
        ],
        body: {
          operations: [
            { kind: "thread-index", result: "i", space: "global", dimension: 0 },
            { kind: "binary", result: "inBounds", operator: "lt", left: "i", right: "N", dtype: "bool" },
            { kind: "load", result: "av", tensor: "a", indices: ["i"], dtype: "float32", mask: "inBounds" },
            { kind: "load", result: "bv", tensor: "b", indices: ["i"], dtype: "float32", mask: "inBounds" },
            { kind: "binary", result: "sum", operator: "add", left: "av", right: "bv", dtype: "float32" },
            { kind: "store", tensor: "out", indices: ["i"], value: "sum", mask: "inBounds" },
            { kind: "return" },
          ],
        },
      },
    ],
  };
}

export function singleKernelModule(overrides = {}) {
  const base = vectorAddModule();
  const kernel = { ...base.kernels[0], ...overrides };
  return { ...base, kernels: [kernel] };
}

export function reductionModule() {
  return {
    name: "reduction_module",
    kernels: [
      {
        name: "sum",
        parameters: [tensorParameter("values", "input"), tensorParameter("out", "output", { shape: [{ kind: "literal", value: 1 }] })],
        launch: {
          grid: [{ kind: "symbol", name: "N" }],
          streamPolicy: "default",
          devicePolicy: "single-device",
        },
        effects: [
          { kind: "read", parameter: "values" },
          { kind: "write", parameter: "out" },
        ],
        body: {
          operations: [
            { kind: "reduce", result: "partial", operator: "sum", operand: "values", scope: "block", dtype: "float32" },
            { kind: "thread-index", result: "lane", space: "local", dimension: 0 },
            { kind: "const", result: "zero", dtype: "int32", value: 0 },
            { kind: "binary", result: "isLeader", operator: "eq", left: "lane", right: "zero", dtype: "bool" },
            { kind: "const", result: "outIndex", dtype: "int32", value: 0 },
            {
              kind: "if",
              condition: "isLeader",
              then: { operations: [{ kind: "store", tensor: "out", indices: ["outIndex"], value: "partial" }] },
            },
          ],
        },
      },
    ],
  };
}
