export type GpuEffect =
  | { readonly kind: "read"; readonly parameter: string }
  | { readonly kind: "write"; readonly parameter: string }
  | { readonly kind: "atomic"; readonly parameter: string }
  | { readonly kind: "barrier"; readonly scope: "block" };

export interface GpuAliasingConstraint {
  readonly kind: "no-overlap";
  readonly parameters: readonly string[];
}
