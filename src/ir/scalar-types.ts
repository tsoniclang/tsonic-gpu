export const gpuScalarTypeIds = Object.freeze([
  "bool",
  "int8",
  "uint8",
  "int16",
  "uint16",
  "int32",
  "uint32",
  "int64",
  "uint64",
  "float16",
  "bfloat16",
  "float32",
  "float64",
] as const);

export type GpuScalarType = (typeof gpuScalarTypeIds)[number];

export function isGpuScalarType(value: unknown): value is GpuScalarType {
  return typeof value === "string" && (gpuScalarTypeIds as readonly string[]).includes(value);
}
