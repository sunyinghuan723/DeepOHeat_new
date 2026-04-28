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


def _resize_to_sensor(np, power, grid_y, grid_x):
    source = np.asarray(power, dtype=np.float32).reshape(grid_y, grid_x)
    if np.max(np.abs(source)) > 0:
        source = source / np.max(np.abs(source))

    y_old = np.linspace(0.0, 1.0, grid_y)
    x_old = np.linspace(0.0, 1.0, grid_x)
    y_new = np.linspace(0.0, 1.0, 21)
    x_new = np.linspace(0.0, 1.0, 21)

    tmp = np.vstack([np.interp(x_new, x_old, row) for row in source])
    resized = np.vstack([np.interp(y_new, y_old, tmp[:, col]) for col in range(tmp.shape[1])]).T
    return resized.reshape(-1, order="F")


def infer(args):
    np, torch, dataio_deeponet, modules = _import_deepoheat()

    with open(args.instance, "r", encoding="utf-8") as f:
        instance = json.load(f)

    grid = instance["grid"]
    channels = instance["channels"]
    power = channels["power_density_w_per_mm2"]
    sensor = _resize_to_sensor(np, power, grid["y"], grid["x"])

    device = args.device or os.environ.get(
        "DEEPOHEAT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
    )
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

    with torch.no_grad():
        u = model(eval_data)["model_out"].detach().cpu().numpy().reshape(-1)

    ambient = float(instance.get("boundary_conditions", {}).get("ambient_temperature", 293.15))
    temperatures = ambient + 25.0 * u
    result = {
        "t_max": float(np.max(temperatures)),
        "t_avg": float(np.mean(temperatures)),
        "field_path": "",
        "adapter": "deeponet_2d_power_map_resampled",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
