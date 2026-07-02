import { test } from "node:test";
import assert from "node:assert/strict";
import { createGpuProviderPackage } from "../dist/index.js";
import { acmeTensorPackage } from "./helpers/gpu-session.mjs";

function definitionWithTensorRow(row, members, typeParameters = [{ name: "D0" }, { name: "D1" }]) {
  return {
    id: "acme-broken",
    displayName: "Broken",
    version: "1.0.0",
    modules: [
      {
        moduleSpecifier: "@acme/broken",
        providerModuleId: "acme.broken",
        exports: [{ id: "@acme/broken::T", name: "T", kind: "class", typeParameters, members: members ?? [] }],
      },
    ],
    tensorTypes: [{ exportId: "@acme/broken::T", elementType: "float32", rank: 1, device: "cuda", ...row }],
  };
}

test("a well-formed provider package definition validates", () => {
  const implementation = acmeTensorPackage();
  assert.equal(implementation.id, "acme-tensor");
});

test("duplicate module specifiers reject at creation", () => {
  const module = { moduleSpecifier: "@acme/dup", providerModuleId: "acme.dup", exports: [] };
  assert.throws(
    () => createGpuProviderPackage({ id: "p", displayName: "P", version: "1.0.0", modules: [module, module] }),
    /module '@acme\/dup' is declared more than once/u,
  );
});

test("duplicate export ids reject at creation", () => {
  const exportDeclaration = { id: "@acme/dup::T", name: "T", kind: "class" };
  assert.throws(
    () =>
      createGpuProviderPackage({
        id: "p",
        displayName: "P",
        version: "1.0.0",
        modules: [
          {
            moduleSpecifier: "@acme/dup",
            providerModuleId: "acme.dup",
            exports: [exportDeclaration, exportDeclaration],
          },
        ],
      }),
    /export '@acme\/dup::T' is declared more than once/u,
  );
});

test("tensor rows must reference a declared export", () => {
  const definition = definitionWithTensorRow({});
  definition.tensorTypes = [{ ...definition.tensorTypes[0], exportId: "@acme/broken::Missing" }];
  assert.throws(() => createGpuProviderPackage(definition), /references an export this package does not declare/u);
});

test("tensor rows validate dtype, device, and rank", () => {
  assert.throws(() => createGpuProviderPackage(definitionWithTensorRow({ elementType: "float128" })), /unknown element dtype/u);
  assert.throws(() => createGpuProviderPackage(definitionWithTensorRow({ device: "fpga" })), /unknown device domain/u);
  assert.throws(() => createGpuProviderPackage(definitionWithTensorRow({ rank: 0 })), /integer rank of at least 1/u);
  assert.throws(() => createGpuProviderPackage(definitionWithTensorRow({ rank: 1.5 })), /integer rank of at least 1/u);
});

test("shape symbol argument positions must match the rank and be unique", () => {
  assert.throws(
    () => createGpuProviderPackage(definitionWithTensorRow({ shapeSymbolArguments: [0, 1] })),
    /declares 2 shape symbol arguments for rank 1/u,
  );
  assert.throws(
    () => createGpuProviderPackage(definitionWithTensorRow({ rank: 2, shapeSymbolArguments: [0, 0] })),
    /repeats shape symbol argument position 0/u,
  );
  assert.throws(
    () => createGpuProviderPackage(definitionWithTensorRow({ shapeSymbolArguments: [-1] })),
    /invalid shape symbol argument position/u,
  );
});

test("shape symbol argument positions must reference the export's type parameters", () => {
  assert.throws(
    () => createGpuProviderPackage(definitionWithTensorRow({ shapeSymbolArguments: [1] }, undefined, [{ name: "T" }])),
    /position 1, but export '@acme\/broken::T' declares 1 type parameter/u,
  );
  assert.throws(
    () => createGpuProviderPackage(definitionWithTensorRow({ shapeSymbolArguments: [0] }, undefined, [])),
    /position 0, but export '@acme\/broken::T' declares 0 type parameter/u,
  );
});

test("load and store member ids must belong to the export", () => {
  assert.throws(
    () => createGpuProviderPackage(definitionWithTensorRow({ loadMember: "@acme/broken::T.at" })),
    /loadMember '@acme\/broken::T.at', which is not a member/u,
  );
  const withMember = definitionWithTensorRow(
    { loadMember: "@acme/broken::T.at" },
    [{ id: "@acme/broken::T.at", name: "at", kind: "method", signatures: [] }],
  );
  createGpuProviderPackage(withMember);
});

test("duplicate tensor rows for one export reject at creation", () => {
  const definition = definitionWithTensorRow({});
  definition.tensorTypes = [definition.tensorTypes[0], definition.tensorTypes[0]];
  assert.throws(() => createGpuProviderPackage(definition), /is declared more than once/u);
});
