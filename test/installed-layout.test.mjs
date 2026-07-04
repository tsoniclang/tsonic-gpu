import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { discoverInstalledTsonicPlugins } from "@tsonic/host";

// Installed-layout discovery through the same host path every target and
// capability uses: a node_modules-style project depending on the GPU
// packages, resolved by the host's own discovery. The full installed GPU
// proof stays skipped until the core host routes gpu-backend/gpu-host
// plugins into the GPU target plugin (docs/core-host-requests.md); the
// watchdog test below pins the pre-routing host behavior and fails loudly
// the moment core routing lands, so the skipped proof gets enabled.

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const tritonRoot = resolve(repositoryRoot, "../gpu-triton");
const projectRoot = join(repositoryRoot, ".temp", "installed-layout");

function createInstalledProject() {
  rmSync(projectRoot, { recursive: true, force: true });
  mkdirSync(join(projectRoot, "node_modules", "@tsonic"), { recursive: true });
  writeFileSync(
    join(projectRoot, "package.json"),
    `${JSON.stringify(
      {
        name: "installed-layout-fixture",
        private: true,
        dependencies: {
          "@tsonic/target-gpu": "*",
          "@tsonic/gpu-triton": "*",
        },
      },
      null,
      2,
    )}\n`,
  );
  writeFileSync(
    join(projectRoot, "tsonic.json"),
    `${JSON.stringify(
      {
        entryPoint: "src/index.ts",
        targets: [{ id: "gpu", options: { backendId: "triton", hostTargetId: "python" } }],
      },
      null,
      2,
    )}\n`,
  );
  symlinkSync(repositoryRoot, join(projectRoot, "node_modules", "@tsonic", "target-gpu"), "dir");
  symlinkSync(tritonRoot, join(projectRoot, "node_modules", "@tsonic", "gpu-triton"), "dir");
  return join(projectRoot, "tsonic.json");
}

test("host discovery finds the installed GPU target plugin", async () => {
  const projectFilePath = createInstalledProject();
  const registry = await discoverInstalledTsonicPlugins(projectFilePath);
  const gpuTarget = registry.targets.find((plugin) => plugin.id === "@tsonic/target-gpu");
  assert.notEqual(gpuTarget, undefined, "the GPU target plugin must be discovered from the installed layout");
  assert.equal(gpuTarget.targetId, "gpu");
  const pack = gpuTarget.createTargetPack();
  assert.equal(pack.id, "gpu");
  // Without core sub-plugin routing, the discovered pack has no backends:
  // any selection fails closed instead of composing silently.
  const target = { id: "gpu", options: { backendId: "triton", hostTargetId: "python" } };
  assert.throws(
    () => pack.createBackend({ project: { entryPoint: "src/index.ts", targets: [target] }, target }),
    /GPU backend 'triton' is not registered/u,
  );
});

test("watchdog: the host does not route gpu-backend plugins to the GPU target", async () => {
  const projectFilePath = createInstalledProject();
  const registry = await discoverInstalledTsonicPlugins(projectFilePath);
  const tritonRouted =
    registry.targets.some((plugin) => plugin.id === "@tsonic/gpu-triton") ||
    (registry.capabilities ?? []).some((plugin) => plugin.id === "@tsonic/gpu-triton");
  const tritonDiagnostic = (registry.diagnostics ?? []).some((diagnostic) =>
    JSON.stringify(diagnostic).includes("@tsonic/gpu-triton"),
  );
  assert.equal(tritonRouted, false, "core sub-plugin routing appears to have landed: enable the installed GPU proof below and retire this watchdog");
  assert.ok(tritonDiagnostic, "the host must surface the unrouted gpu-backend plugin as a diagnostic, not drop it silently");
});

test("installed GPU proof: discovery, extraction, Triton lowering, host merge", { skip: "enable when core sub-plugin routing lands (docs/core-host-requests.md)" }, () => {
  // Installed @tsonic/target-gpu + @tsonic/gpu-triton + @tsonic/target-python,
  // real host discovery, TS kernel extraction, Triton lowering, Python host
  // artifact merge, compileall/ast.parse; Triton runtime execution stays
  // dependency-gated.
});
