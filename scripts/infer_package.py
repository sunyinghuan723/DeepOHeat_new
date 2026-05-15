#!/usr/bin/env python3
"""Legacy 2D power-map DeepOHeat adapter for ChipletPart thermal instances.

This compatibility adapter maps ChipletPart's package JSON to the pretrained
DeepOHeat 2D power-map DeepONet input. The mapping is intentionally narrow:
chiplet/package metadata is preserved in the JSON, while the neural model
currently consumes only the top-surface power-density channel resampled to the
21x21 branch sensor used by the 2d_power_map checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _import_deepoheat():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    sys.path.insert(0, str(repo_root))
    try:
        import numpy as np
        import torch
        from src import dataio_deeponet, modules
    except ImportError as exc:
        raise SystemExit(
            "DeepOHeat adapter requires the DeepOHeat Python environment "
            "(numpy + torch). Try: source /opt/conda/miniforge3/etc/profile.d/conda.sh; "
            "conda run -p DeepOHeat/.conda/deepoheat-py38 python ..."
        ) from exc
    return np, torch, dataio_deeponet, modules


def _domain_definition():
    domain_0 = dict(
        domain_name=0,
        geometry=dict(
            starts=[0.0, 0.0, 0.0],
            ends=[1.0, 1.0, 0.5],
            num_intervals=[20, 20, 10],
            num_pde_points=2000,
            num_single_bc_points=200,
        ),
        conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
        power=dict(
            bc=True,
            num_power_points_per_volume=2,
            num_power_points_per_surface=500,
            num_power_points_per_cell=5,
            power_map=dict(
                power_0=dict(
                    type="surface_power",
                    surface="top",
                    location=dict(starts=(0, 0, 10), ends=(20, 20, 10)),
                    params=dict(dim=2, value=1, weight=1),
                )
            ),
        ),
        front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
        back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
        left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
        right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
        bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
        top=dict(bc=True),
        node=dict(root=True, leaf=True),
        parameterized=dict(variable=False),
    )
    global_params = {
        "loss_fun_type": "norm",
        "num_params_per_epoch": 50,
        "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
    }
    return [domain_0], global_params


def _resolve_device(torch, requested):
    device = requested or os.environ.get("DEEPOHEAT_DEVICE", "auto")
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested --device cuda, but CUDA is not available")
        return "cuda:0"
    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested --device {device}, but CUDA is not available")
        try:
            index = int(device.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"invalid CUDA device string: {device}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested --device {device}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        return device
    if device == "cpu":
        return "cpu"
    raise RuntimeError(f"unsupported --device value: {device}")


def _load_config(path):
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"thermal model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _grid_shape(instance):
    grid = instance.get("grid", {})
    grid_x = grid.get("x", instance.get("grid_x"))
    grid_y = grid.get("y", instance.get("grid_y"))
    if grid_x is None or grid_y is None:
        raise KeyError("thermal instance is missing grid x/y dimensions")
    return int(grid_y), int(grid_x)


def _resize_to_sensor(np, power, grid_y, grid_x, power_scale=1.0, normalize=False):
    source = np.asarray(power, dtype=np.float32).reshape(grid_y, grid_x)
    source = source * float(power_scale)
    if normalize and source.size and np.max(np.abs(source)) > 0:
        source = source / np.max(np.abs(source))

    y_old = np.linspace(0.0, 1.0, grid_y)
    x_old = np.linspace(0.0, 1.0, grid_x)
    y_new = np.linspace(0.0, 1.0, 21)
    x_new = np.linspace(0.0, 1.0, 21)

    tmp = np.vstack([np.interp(x_new, x_old, row) for row in source])
    resized = np.vstack([np.interp(y_new, y_old, tmp[:, col]) for col in range(tmp.shape[1])]).T
    return resized.reshape(-1, order="F"), source


class Legacy2DPowerMapPredictor:
    """Keep DeepOHeat imports, checkpoint, model, and eval mesh alive."""

    def __init__(self, args):
        self.np, self.torch, dataio_deeponet, modules = _import_deepoheat()
        self.config = _load_config(args.config)
        self.model_path = Path(args.model)
        if not self.model_path.exists():
            raise FileNotFoundError(f"thermal model checkpoint not found: {self.model_path}")
        self.power_scale_arg = args.power_scale
        self.normalize_power_arg = bool(args.normalize_power)
        self.device = _resolve_device(self.torch, args.device)
        self.model = modules.DeepONet(
            trunk_in_features=3,
            trunk_hidden_features=128,
            branch_in_features=441,
            branch_hidden_features=256,
            inner_prod_features=128,
            num_trunk_hidden_layers=3,
            num_branch_hidden_layers=7,
            nonlinearity="silu",
            freq=2 * self.torch.pi,
            std=1,
            freq_trainable=True,
            device=self.device,
        )
        checkpoint = self.torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        domains_list, global_params = _domain_definition()
        dataset = dataio_deeponet.DeepONetMeshDataIO(
            domains_list, global_params, dim=2, var=1, len_scale=0.3
        )
        eval_data = dataset.eval()
        self.coords = eval_data["coords"].float().to(self.device)
        self.coords_np = self.coords.detach().cpu().numpy().astype(self.np.float32)

    def predict(self, instance_path, output_path):
        with open(instance_path, "r", encoding="utf-8") as f:
            instance = json.load(f)

        channels = instance["channels"]
        power = channels["power_density_w_per_mm2"]
        grid_y, grid_x = _grid_shape(instance)
        power_scale = (
            self.power_scale_arg
            if self.power_scale_arg is not None
            else float(self.config.get("power_scale", 1.0))
        )
        normalize_power = bool(
            self.normalize_power_arg or self.config.get("normalize_power", False)
        )
        sensor, scaled_power_grid = _resize_to_sensor(
            self.np,
            power,
            grid_y,
            grid_x,
            power_scale=power_scale,
            normalize=normalize_power,
        )

        beta = self.torch.tensor(
            sensor.reshape(1, -1).repeat(self.coords.shape[0], 0),
            device=self.device,
        ).float()
        eval_data = {"coords": self.coords, "beta": beta}

        start = time.perf_counter()
        with self.torch.no_grad():
            u = self.model(eval_data)["model_out"].detach().cpu().numpy().reshape(-1)
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.torch.device(self.device))
        runtime_sec = time.perf_counter() - start

        ambient = float(instance.get("boundary_conditions", {}).get("ambient_temperature", 293.15))
        temperature_scale = float(self.config.get("temperature_scale", 25.0))
        temperatures = ambient + temperature_scale * u
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        field_path = str(output_path.with_suffix(".field.npz"))
        self.np.savez_compressed(
            field_path,
            coords=self.coords_np,
            temperature=temperatures.astype(self.np.float32),
            sensor=sensor.reshape(21, 21, order="F").astype(self.np.float32),
            scaled_power_grid=scaled_power_grid.astype(self.np.float32),
        )
        gpu_name = ""
        if self.torch.cuda.is_available() and str(self.device).startswith("cuda"):
            gpu_name = self.torch.cuda.get_device_name(self.torch.device(self.device))
        result = {
            "t_max": float(self.np.max(temperatures)),
            "t_avg": float(self.np.mean(temperatures)),
            "field_path": field_path,
            "adapter": "deeponet_2d_power_map_resampled",
            "runtime_sec": float(runtime_sec),
            "device": str(self.device),
            "cuda_available": bool(self.torch.cuda.is_available()),
            "cuda_device_count": int(self.torch.cuda.device_count()),
            "gpu_name": gpu_name,
            "power_scale": float(power_scale),
            "normalize_power": normalize_power,
            "temperature_scale": temperature_scale,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result


def infer(args):
    predictor = Legacy2DPowerMapPredictor(args)
    predictor.predict(args.instance, args.output)


def serve(args):
    predictor = Legacy2DPowerMapPredictor(args)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request.get("shutdown"):
                break
            result = predictor.predict(request["instance"], request["output"])
            print(json.dumps({"ok": True, **result}), flush=True)
        except Exception as exc:  # Keep the C++ parent alive long enough to see the error.
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--instance")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config")
    parser.add_argument("--power_scale", type=float)
    parser.add_argument("--normalize_power", action="store_true")
    args = parser.parse_args()
    if args.server:
        serve(args)
    else:
        if not args.instance or not args.output:
            parser.error("--instance and --output are required unless --server is set")
        infer(args)


if __name__ == "__main__":
    main()
