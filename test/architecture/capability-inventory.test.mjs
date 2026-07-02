import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { ledgerLanes } from "../helpers/ledger-lanes.mjs";

// G7 inventory audit: every GPU capability id in product code must be
// classified here, and every legality-reject id must have a ledger lane.
// Adding a capability id without classifying it fails this test.

const capabilityCatalog = Object.freeze({
  // Dynamic id families completed per dtype/operator/device at runtime.
  "gpu.atomic": "capability-family",
  "gpu.device": "capability-family",
  "gpu.dtype": "capability-family",
  "gpu.layout": "capability-family",
  "gpu.layout.backend": "capability-family",
  "gpu.math": "capability-family",
  "gpu.op.binary": "capability-family",
  "gpu.op.unary": "capability-family",
  "gpu.reduce": "capability-family",
  "gpu.tensor.rank": "capability-family",
  "gpu.thread-index": "capability-family",
  // Static capability ids matched against backend capability sets.
  "gpu.barrier.block": "negotiation-capability",
  "gpu.control.if": "negotiation-capability",
  "gpu.control.loop": "negotiation-capability",
  "gpu.launch.meta": "negotiation-capability",
  "gpu.local.mutable": "negotiation-capability",
  "gpu.memory.load": "negotiation-capability",
  "gpu.memory.masked": "negotiation-capability",
  "gpu.memory.store": "negotiation-capability",
  "gpu.shape.symbolic": "negotiation-capability",
  // Legality-reject lanes; each must appear in the capability ledger.
  "gpu.device.host-call": "legality-reject",
  "gpu.device.mixed": "legality-reject",
  "gpu.kernel.assignment": "legality-reject",
  "gpu.kernel.binding": "legality-reject",
  "gpu.kernel.condition": "legality-reject",
  "gpu.kernel.declaration-form": "legality-reject",
  "gpu.kernel.expression": "legality-reject",
  "gpu.kernel.host-capture": "legality-reject",
  "gpu.kernel.index-arity": "legality-reject",
  "gpu.kernel.index-dtype": "legality-reject",
  "gpu.kernel.indexing": "legality-reject",
  "gpu.kernel.loop-form": "legality-reject",
  "gpu.kernel.mixed-dtype": "legality-reject",
  "gpu.kernel.mutable-local": "legality-reject",
  "gpu.kernel.no-output": "legality-reject",
  "gpu.kernel.operator": "legality-reject",
  "gpu.kernel.parameter-type": "legality-reject",
  "gpu.kernel.meta-parameter": "legality-reject",
  "gpu.kernel.reduce-operand": "legality-reject",
  "gpu.kernel.return-value": "legality-reject",
  "gpu.kernel.shape-dim": "legality-reject",
  "gpu.kernel.shape-symbols": "legality-reject",
  "gpu.kernel.statement": "legality-reject",
  "gpu.kernel.store-dtype": "legality-reject",
  "gpu.kernel.tensor-value": "legality-reject",
  "gpu.kernel.thread-index": "legality-reject",
  "gpu.kernel.unbounded-loop": "legality-reject",
  // Reject lanes that checked TypeScript cannot reach today; they guard
  // hand-authored input and future extraction growth.
  "gpu.kernel.launch": "defense-in-depth",
  "gpu.kernel.name": "defense-in-depth",
  "gpu.kernel.parameter-name": "defense-in-depth",
  // Structural GPU IR invariants validated before capability matching.
  "gpu.ir.aliasing": "ir-invariant",
  "gpu.ir.assign": "ir-invariant",
  "gpu.ir.effect": "ir-invariant",
  "gpu.ir.index-arity": "ir-invariant",
  "gpu.ir.kernel-name": "ir-invariant",
  "gpu.ir.launch.block": "ir-invariant",
  "gpu.ir.launch.grid": "ir-invariant",
  "gpu.ir.launch.meta": "ir-invariant",
  "gpu.ir.launch.symbol": "ir-invariant",
  "gpu.ir.load.dtype": "ir-invariant",
  "gpu.ir.module-name": "ir-invariant",
  "gpu.ir.parameter-name": "ir-invariant",
  "gpu.ir.scalar.dtype": "ir-invariant",
  "gpu.ir.store.mutability": "ir-invariant",
  "gpu.ir.symbol-collision": "ir-invariant",
  "gpu.ir.tensor-ref": "ir-invariant",
  "gpu.ir.tensor.device": "ir-invariant",
  "gpu.ir.tensor.dtype": "ir-invariant",
  "gpu.ir.tensor.mutability": "ir-invariant",
  "gpu.ir.tensor.rank": "ir-invariant",
  "gpu.ir.tensor.shape": "ir-invariant",
  "gpu.ir.tensor.strides": "ir-invariant",
  "gpu.ir.thread-index": "ir-invariant",
  "gpu.ir.value-def": "ir-invariant",
  "gpu.ir.value-ref": "ir-invariant",
  // Host boundary contract.
  "gpu.host.integration": "host-boundary",
  // Intrinsic spellings quoted in user-facing diagnostic messages.
  "gpu.dim": "intrinsic-spelling",
  "gpu.meta": "intrinsic-spelling",
  // Diagnostic evidence keys, not capabilities.
  "gpu.backend": "evidence-key",
  "gpu.host": "evidence-key",
  "gpu.hosts.registered": "evidence-key",
  "gpu.kernel": "evidence-key",
  "gpu.module": "evidence-key",
});

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const sourceRoot = join(repositoryRoot, "src");

function collectFiles(root, extension) {
  const results = [];
  const visit = (directory) => {
    for (const entry of readdirSync(directory)) {
      const fullPath = join(directory, entry);
      if (statSync(fullPath).isDirectory()) {
        visit(fullPath);
        continue;
      }
      if (fullPath.endsWith(extension)) {
        results.push(fullPath);
      }
    }
  };
  visit(root);
  return results;
}

function extractCapabilityIds() {
  const ids = new Set();
  const pattern = /(?<![.a-z-])gpu\.[a-z][a-z0-9.-]*/gu;
  for (const path of collectFiles(sourceRoot, ".ts")) {
    const text = readFileSync(path, "utf8");
    for (const match of text.matchAll(pattern)) {
      ids.add(match[0].replace(/\.+$/u, ""));
    }
  }
  return [...ids].sort();
}

test("every GPU capability id in product code is classified with zero drift", () => {
  const extracted = extractCapabilityIds();
  const catalogued = Object.keys(capabilityCatalog).sort();
  assert.deepEqual(extracted, catalogued);
});

test("every legality-reject capability has a ledger lane", () => {
  const laneCapabilities = new Set(ledgerLanes.map((lane) => lane.capability));
  const rejectIds = Object.entries(capabilityCatalog)
    .filter(([, classification]) => classification === "legality-reject")
    .map(([id]) => id);
  for (const id of rejectIds) {
    assert.ok(laneCapabilities.has(id), `legality-reject capability '${id}' has no ledger lane`);
  }
});

test("every ledger lane targets a catalogued legality-reject capability", () => {
  for (const lane of ledgerLanes) {
    assert.equal(
      capabilityCatalog[lane.capability],
      "legality-reject",
      `ledger lane '${lane.name}' targets '${lane.capability}', which is not a legality-reject capability`,
    );
  }
});
