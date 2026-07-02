import type { GpuScalarType } from "./scalar-types.js";
import type { GpuShapeExpr } from "./shape.js";

export const gpuDeviceDomains = Object.freeze(["cpu", "cuda", "rocm", "metal", "backend-specific"] as const);

export type GpuDeviceDomain = (typeof gpuDeviceDomains)[number];

export function isGpuDeviceDomain(value: unknown): value is GpuDeviceDomain {
  return typeof value === "string" && (gpuDeviceDomains as readonly string[]).includes(value);
}

export interface GpuDeviceRef {
  readonly domain: GpuDeviceDomain;
  readonly deviceId?: string;
}

export type GpuLayout =
  | { readonly kind: "contiguous" }
  | { readonly kind: "strided" }
  | { readonly kind: "channels-last" }
  | { readonly kind: "backend-specific"; readonly layoutId: string };

export type GpuMemorySpace = "global" | "shared" | "local" | "constant";

export type GpuMutability = "readonly" | "mutable";

export type GpuAliasing = "noalias" | "may-alias";

export interface GpuTensorType {
  readonly elementType: GpuScalarType;
  readonly rank: number;
  readonly shape: readonly GpuShapeExpr[];
  readonly strides?: readonly GpuShapeExpr[];
  readonly layout: GpuLayout;
  readonly device: GpuDeviceRef;
  readonly mutability: GpuMutability;
  readonly aliasing: GpuAliasing;
}
