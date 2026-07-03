import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { TsonicTargetPlugin } from "@tsonic/target-api";
import type { GpuBackendPlugin } from "./backends/backend-contract.js";
import { createGpuTargetPack, type GpuTargetPackConfig } from "./descriptor/gpu-target-pack.js";
import { gpuTargetId } from "./descriptor/target-id.js";
import type { GpuHostIntegration } from "./hosts/host-contract.js";

export const gpuTargetPluginId = "@tsonic/target-gpu";

// The tsonic host plugin contract: discovery reads the package.json
// 'tsonic' manifest (kind 'plugin', contractVersion 1, entry resolved
// through package exports) and calls createTsonicPlugin() with no
// arguments. This manifest shape is owned by tsonic core; GPU packages
// follow it exactly and never invent parallel fields.
export interface TsonicGpuPluginManifest {
  readonly kind: "plugin";
  readonly contractVersion: 1;
  readonly entry: string;
}

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function readTsonicPluginManifest(): TsonicGpuPluginManifest {
  const packageJsonPath = resolve(packageRoot, "package.json");
  const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
    readonly tsonic?: unknown;
  };
  const manifest = packageJson.tsonic;
  if (manifest === undefined || typeof manifest !== "object" || manifest === null) {
    throw new Error(`${gpuTargetPluginId}: package.json is missing the 'tsonic' plugin manifest.`);
  }
  const { kind, contractVersion, entry } = manifest as {
    readonly kind?: unknown;
    readonly contractVersion?: unknown;
    readonly entry?: unknown;
  };
  if (kind !== "plugin") {
    throw new Error(`${gpuTargetPluginId}: tsonic manifest kind must be 'plugin', got '${String(kind)}'.`);
  }
  if (contractVersion !== 1) {
    throw new Error(`${gpuTargetPluginId}: tsonic manifest contractVersion must be 1, got '${String(contractVersion)}'.`);
  }
  if (typeof entry !== "string" || entry.length === 0) {
    throw new Error(`${gpuTargetPluginId}: tsonic manifest entry must be a non-empty string.`);
  }
  return { kind, contractVersion, entry };
}

// GPU sub-plugin entry contracts. Backend and host packages export a
// createTsonicPlugin() returning one of these shapes; the GPU target plugin
// composes them structurally. Selection stays data driven: the GPU core
// learns backend and host ids only from plugin entries and target options.

export interface GpuBackendTsonicPlugin {
  readonly kind: "gpu-backend";
  readonly id: string;
  readonly backendId: string;
  createBackend(): GpuBackendPlugin;
}

export interface GpuHostTsonicPlugin {
  readonly kind: "gpu-host";
  readonly id: string;
  readonly hostTargetId: string;
  createHostIntegration(): GpuHostIntegration;
}

export type GpuTsonicPlugin = GpuBackendTsonicPlugin | GpuHostTsonicPlugin;

export interface GpuPluginComposition {
  readonly plugins?: readonly unknown[];
}

// Structural, fail-closed composition: every entry must be a well-formed GPU
// sub-plugin whose created instance agrees with its declared id. Anything
// else throws before a target pack exists.
export function resolveGpuPluginComposition(composition: GpuPluginComposition = {}): GpuTargetPackConfig {
  const backends: GpuBackendPlugin[] = [];
  const hosts: GpuHostIntegration[] = [];
  const seenPluginIds = new Map<string, string>();
  const seenBackendIds = new Map<string, string>();
  const seenHostTargetIds = new Map<string, string>();
  for (const entry of composition.plugins ?? []) {
    if (entry === null || typeof entry !== "object") {
      throw new Error(`GPU plugin composition received a non-object plugin entry: ${String(entry)}.`);
    }
    const candidate = entry as {
      readonly kind?: unknown;
      readonly id?: unknown;
      readonly backendId?: unknown;
      readonly hostTargetId?: unknown;
      readonly createBackend?: unknown;
      readonly createHostIntegration?: unknown;
    };
    const pluginId = typeof candidate.id === "string" && candidate.id.length > 0 ? candidate.id : undefined;
    if (pluginId === undefined) {
      throw new Error("GPU plugin entries need a non-empty package id.");
    }
    if (seenPluginIds.has(pluginId)) {
      throw new Error(`GPU plugin '${pluginId}' is composed more than once.`);
    }
    seenPluginIds.set(pluginId, pluginId);
    if (candidate.kind === "gpu-backend") {
      if (typeof candidate.backendId !== "string" || candidate.backendId.length === 0 || typeof candidate.createBackend !== "function") {
        throw new Error(`GPU backend plugin '${pluginId}' needs a non-empty backendId and a createBackend() entry.`);
      }
      const instance = (candidate.createBackend as () => GpuBackendPlugin)();
      if (instance.id !== candidate.backendId) {
        throw new Error(
          `GPU backend plugin '${pluginId}' declares backendId '${candidate.backendId}' but created a backend with id '${instance.id}'.`,
        );
      }
      const previous = seenBackendIds.get(instance.id);
      if (previous !== undefined) {
        throw new Error(`GPU backend id '${instance.id}' is provided by both '${previous}' and '${pluginId}'.`);
      }
      seenBackendIds.set(instance.id, pluginId);
      backends.push(instance);
      continue;
    }
    if (candidate.kind === "gpu-host") {
      if (
        typeof candidate.hostTargetId !== "string" ||
        candidate.hostTargetId.length === 0 ||
        typeof candidate.createHostIntegration !== "function"
      ) {
        throw new Error(`GPU host plugin '${pluginId}' needs a non-empty hostTargetId and a createHostIntegration() entry.`);
      }
      const instance = (candidate.createHostIntegration as () => GpuHostIntegration)();
      if (instance.hostTargetId !== candidate.hostTargetId) {
        throw new Error(
          `GPU host plugin '${pluginId}' declares hostTargetId '${candidate.hostTargetId}' but created an integration for '${instance.hostTargetId}'.`,
        );
      }
      const previous = seenHostTargetIds.get(instance.hostTargetId);
      if (previous !== undefined) {
        throw new Error(`GPU host target '${instance.hostTargetId}' is provided by both '${previous}' and '${pluginId}'.`);
      }
      seenHostTargetIds.set(instance.hostTargetId, pluginId);
      hosts.push(instance);
      continue;
    }
    throw new Error(
      `GPU plugin '${pluginId}' has kind '${String(candidate.kind)}'; the GPU target composes 'gpu-backend' and 'gpu-host' plugins.`,
    );
  }
  return { backends, hosts };
}

// The host calls createTsonicPlugin() with no arguments; the zero-argument
// form yields a pack with no backends or hosts, which fails closed on any
// selection. The composition argument is the local wiring channel for GPU
// sub-plugins; routing discovered gpu-backend/gpu-host plugins into this
// factory is a tsonic core responsibility (docs/core-host-requests.md).
export function createTsonicPlugin(composition: GpuPluginComposition = {}): TsonicTargetPlugin {
  readTsonicPluginManifest();
  const config = resolveGpuPluginComposition(composition);
  return {
    kind: "target",
    id: gpuTargetPluginId,
    targetId: gpuTargetId,
    createTargetPack: () => createGpuTargetPack(config),
  };
}
