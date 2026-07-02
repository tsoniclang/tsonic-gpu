import type { GpuHostIntegration } from "./host-contract.js";

const hostTargetIdPattern = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/u;

export interface GpuHostRegistry {
  ids(): readonly string[];
  get(hostTargetId: string): GpuHostIntegration | undefined;
}

export function createGpuHostRegistry(integrations: readonly GpuHostIntegration[]): GpuHostRegistry {
  const byId = new Map<string, GpuHostIntegration>();
  for (const integration of integrations) {
    if (!hostTargetIdPattern.test(integration.hostTargetId)) {
      throw new Error(`GPU host integration id '${integration.hostTargetId}' must match ${hostTargetIdPattern.source}.`);
    }
    if (byId.has(integration.hostTargetId)) {
      throw new Error(`GPU host integration '${integration.hostTargetId}' is registered more than once.`);
    }
    byId.set(integration.hostTargetId, integration);
  }
  return {
    ids: () => [...byId.keys()],
    get: (hostTargetId) => byId.get(hostTargetId),
  };
}
