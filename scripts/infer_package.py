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


def infer(args):
    np, torch, dataio_deeponet, modules = _import_deepoheat()
    config = _load_config(args.config)

    with open(args.instance, "r", encoding="utf-8") as f:
        instance = json.load(f)

    channels = instance["channels"]
    power = channels["power_density_w_per_mm2"]
    grid_y, grid_x = _grid_shape(instance)
    power_scale = (
        args.power_scale
        if args.power_scale is not None
        else float(config.get("power_scale", 1.0))
    )
    normalize_power = bool(args.normalize_power or config.get("normalize_power", False))
    sensor, scaled_power_grid = _resize_to_sensor(
        np,
        power,
        grid_y,
        grid_x,
        power_scale=power_scale,
        normalize=normalize_power,
    )

    device = _resolve_device(torch, args.device)
    model = modules.DeepONet(
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=441,
        branch_hidden_features=256,
        inner_prod_features=128,
        num_trunk_hidden_layers=3,
        num_branch_hidden_layers=7,
        nonlinearity="silu",
        freq=2 * torch.pi,
        std=1,
        freq_trainable=True,
        device=device,
    )

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    domains_list, global_params = _domain_definition()
    dataset = dataio_deeponet.DeepONetMeshDataIO(
        domains_list, global_params, dim=2, var=1, len_scale=0.3
    )
    eval_data = dataset.eval()
    coords = eval_data["coords"].float().to(device)
    beta = torch.tensor(sensor.reshape(1, -1).repeat(coords.shape[0], 0), device=device).float()
    eval_data = {"coords": coords, "beta": beta}

    start = time.perf_counter()
    with torch.no_grad():
        u = model(eval_data)["model_out"].detach().cpu().numpy().reshape(-1)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    runtime_sec = time.perf_counter() - start

    ambient = float(instance.get("boundary_conditions", {}).get("ambient_temperature", 293.15))
    temperature_scale = float(config.get("temperature_scale", 25.0))
    temperatures = ambient + temperature_scale * u
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_path = str(output_path.with_suffix(".field.npz"))
    np.savez_compressed(
        field_path,
        coords=coords.detach().cpu().numpy().astype(np.float32),
        temperature=temperatures.astype(np.float32),
        sensor=sensor.reshape(21, 21, order="F").astype(np.float32),
        scaled_power_grid=scaled_power_grid.astype(np.float32),
    )
    gpu_name = ""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(torch.device(device))
    result = {
        "t_max": float(np.max(temperatures)),
        "t_avg": float(np.mean(temperatures)),
        "field_path": field_path,
        "adapter": "deeponet_2d_power_map_resampled",
        "runtime_sec": float(runtime_sec),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": gpu_name,
        "power_scale": float(power_scale),
        "normalize_power": normalize_power,
        "temperature_scale": temperature_scale,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config")
    parser.add_argument("--power_scale", type=float)
    parser.add_argument("--normalize_power", action="store_true")
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
