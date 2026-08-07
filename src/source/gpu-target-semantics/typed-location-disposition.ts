import {
  pointerOperationFactKey,
} from "@tsonic/tsts";
import type {
  ExtensionFactSubject,
  ReadonlySourceFactResolver,
} from "@tsonic/tsts";

export interface GpuUnsupportedTypedLocationOperation {
  readonly kind: "unsupported-typed-location";
  readonly operation: "address-of" | "allocate" | "load" | "store";
}

export function selectGpuTypedLocationDisposition(
  facts: ReadonlySourceFactResolver,
  subject: ExtensionFactSubject,
): GpuUnsupportedTypedLocationOperation | undefined {
  const sourceOperation = facts.getFact(subject, pointerOperationFactKey);
  return sourceOperation === undefined
    ? undefined
    : {
        kind: "unsupported-typed-location",
        operation: sourceOperation.operation,
      };
}
