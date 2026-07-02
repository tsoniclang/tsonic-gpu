import { test } from "node:test";
import assert from "node:assert/strict";
import { capabilityIds, compileGpu } from "./helpers/gpu-session.mjs";
import { ledgerLanes } from "./helpers/ledger-lanes.mjs";

// Every unsupported lane must fail closed: no artifacts, only GPU_-coded
// error diagnostics, and evidence naming the expected capability id.
for (const lane of ledgerLanes) {
  test(`ledger: ${lane.name} fails closed with ${lane.capability}`, () => {
    const { result } = compileGpu({ files: lane.files });
    assert.deepEqual(result.artifacts, [], `lane '${lane.name}' must not emit artifacts`);
    assert.ok(result.diagnostics.length > 0, `lane '${lane.name}' must produce diagnostics`);
    assert.ok(
      result.diagnostics.every((diagnostic) => diagnostic.code.startsWith("GPU_")),
      `lane '${lane.name}' diagnostics must be GPU-coded: ${result.diagnostics.map((diagnostic) => diagnostic.code).join(", ")}`,
    );
    assert.ok(
      result.diagnostics.every((diagnostic) => diagnostic.category === "error"),
      `lane '${lane.name}' diagnostics must be errors`,
    );
    assert.ok(
      capabilityIds(result.diagnostics).includes(lane.capability),
      `lane '${lane.name}' must diagnose ${lane.capability}; found: ${capabilityIds(result.diagnostics).join(", ")}`,
    );
  });
}

test("ledger lanes are unique per name", () => {
  const names = ledgerLanes.map((lane) => lane.name);
  assert.equal(new Set(names).size, names.length);
});
