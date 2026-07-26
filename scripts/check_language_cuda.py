"""Fail fast unless this Linux host can run the pinned MLX-CUDA language workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from importlib.metadata import version
from typing import Any

MINIMUM_COMPUTE_CAPABILITY = (7, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def inspect_cuda_runtime() -> dict[str, Any]:
    """Return CUDA/MLX details or raise a concise compatibility error."""

    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("MLX-CUDA Step 11 requires Linux x86_64")

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU; enable a GPU runtime")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if capability < MINIMUM_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "GPU compute capability "
            f"{capability[0]}.{capability[1]} is unsupported; MLX-CUDA requires 7.5 or newer"
        )

    mx = importlib.import_module("mlx.core")
    mlx_lm = importlib.import_module("mlx_lm")
    mx.set_default_device(mx.gpu)
    if mx.default_device().type != mx.gpu:
        raise RuntimeError("MLX did not select its GPU device")
    left = mx.array([[1.0, 2.0]])
    right = mx.array([[3.0], [4.0]])
    product = left @ right
    mx.eval(product)
    if float(product.item()) != 11.0:
        raise RuntimeError("MLX GPU smoke calculation returned an unexpected result")

    properties = torch.cuda.get_device_properties(0)
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "gpu_memory_gib": round(float(properties.total_memory) / (1024**3), 2),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "mlx_device": str(mx.default_device()),
        "mlx_lm_import": str(getattr(mlx_lm, "__name__", "mlx_lm")),
        "smoke_result": float(product.item()),
    }


def main() -> None:
    args = parse_args()
    details = inspect_cuda_runtime()
    if args.json:
        print(json.dumps(details, sort_keys=True))
        return
    print("MLX-CUDA preflight passed")
    for key, value in details.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
