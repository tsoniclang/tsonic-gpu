# Tsonic Host Requirements

Host-side support the GPU target family requires from tsonic core. Each entry
states the requirement and the local foundation this package provides in the
meantime. The core plugin manifest contract (`tsonic.kind = "plugin"`,
`contractVersion`, `entry`) is owned by tsonic core; GPU packages follow it
exactly and define no parallel manifest fields.

## Sub-plugin routing for target families

- Requirement: a core-supported channel that routes discovered plugins of a
  target family (here: `gpu-backend` and `gpu-host` entries returned by
  `createTsonicPlugin()` of backend/host packages) into the family's target
  plugin, so `@tsonic/target-gpu` receives them at `createTargetPack` time.
  Backend and host plugins are not `target-capability` plugins: capabilities
  extend source semantics, while backends and hosts implement the GPU
  compilation contract, and forcing them into the capability shape would be
  the wrong abstraction.
- Local foundation: `createTsonicPlugin(composition)` accepts explicitly
  constructed sub-plugin entries and composes them with fail-closed
  validation; the zero-argument host form yields a pack that fails closed on
  any backend or host selection.

## Installed-layout discovery proof

- Requirement: once the host contract for sub-plugin routing is stable, the
  GPU family needs node_modules-style discovery tests (project package.json
  dependencies, package exports resolution, `createTsonicPlugin()` entry) in
  addition to the local composition tests.
- Local foundation: composition, id-agreement, duplicate, and fail-closed
  tests run against explicitly constructed plugin objects.
