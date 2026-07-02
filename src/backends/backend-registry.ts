import type { GpuBackendPlugin } from "./backend-contract.js";

const backendIdPattern = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/u;

export interface GpuBackendRegistry {
  ids(): readonly string[];
  get(id: string): GpuBackendPlugin | undefined;
}

export function createGpuBackendRegistry(plugins: readonly GpuBackendPlugin[]): GpuBackendRegistry {
  const byId = new Map<string, GpuBackendPlugin>();
  for (const plugin of plugins) {
    if (!backendIdPattern.test(plugin.id)) {
      throw new Error(`GPU backend id '${plugin.id}' must match ${backendIdPattern.source}.`);
    }
    if (byId.has(plugin.id)) {
      throw new Error(`GPU backend id '${plugin.id}' is registered more than once.`);
    }
    byId.set(plugin.id, plugin);
  }
  return {
    ids: () => [...byId.keys()],
    get: (id) => byId.get(id),
  };
}
