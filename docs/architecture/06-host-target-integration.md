# Host Target Integration

GPU kernels need a host target for packaging and launch. Python is the first host target, but the
GPU core must not become Python-specific.

## Integration Flow

```text
GPU target core
  produces backend-neutral kernel plans
       |
       v
GPU backend plugin
  produces backend artifacts and launch wrapper requests
       |
       v
host target
  places files, dependencies, imports, and project metadata
```

## Python Host Requirements

The Python host must be able to accept:

- Python module artifact,
- dependency declaration,
- import requirement,
- launch wrapper function,
- test/run command contribution.

Example artifact request:

```text
module: kernels/add_kernel.py
dependency: triton
dependency: torch
wrapper: add(a, b, out)
```

`tsonic-python` decides where `kernels/add_kernel.py` lives and how `pyproject.toml` records
dependencies.

## Data Movement

The GPU core does not decide host data allocation. Provider/library facts decide whether a value is
a device tensor, CPU tensor, or scalar. The GPU core verifies compatibility.

## No Automatic CPU Fallback

If the selected backend cannot run a kernel, compilation fails. CPU fallback is a separate backend
choice, not a silent recovery path.

## Cross-Target Reach

The same GPU IR should be usable by:

- Python + Triton,
- Python + CUDA extension,
- Rust host + CUDA/WGPU,
- C# host + CUDA/DirectML,
- Go host + another GPU backend.

This is why the GPU core cannot import or name Python libraries in product code.

