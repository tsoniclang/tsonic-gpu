import type { GpuShapeExpr } from "./shape.js";

export type GpuStreamPolicy = "default" | "backend-managed";

export type GpuDevicePolicy = "single-device";

export interface GpuLaunchPlan {
  readonly grid: readonly GpuShapeExpr[];
  readonly block?: readonly GpuShapeExpr[];
  readonly metaParameters?: readonly string[];
  readonly streamPolicy: GpuStreamPolicy;
  readonly devicePolicy: GpuDevicePolicy;
}
