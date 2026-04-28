#!/usr/bin/env python3
"""Evaluate a package-level surrogate checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_CHANNELS, PackageThermalDataset, normalized_coords
from model import PackageThermalDeepONet


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[PackageThermalDeepONet, dict]:
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    channels = config.get("channel_names", DEFAULT_CHANNELS)
    model = PackageThermalDeepONet(
        len(channels),
        feature_dim=int(config.get("feature_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    channels = config.get("channel_names", DEFAULT_CHANNELS)
    dataset = PackageThermalDataset(args.manifest, split=args.split, channel_names=channels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    grid_y = int(config.get("grid_y", dataset[0]["temperature"].shape[0]))
    grid_x = int(config.get("grid_x", dataset[0]["temperature"].shape[1]))
    coords = normalized_coords(grid_y, grid_x).to(device)

    maes = []
    rmses = []
    max_errors = []
    avg_errors = []
    mapes = []
    runtimes = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].float().to(device)
            target = batch["temperature"].float().to(device)
            start = time.perf_counter()
            pred = model(x, coords).reshape_as(target)
            runtimes.append(time.perf_counter() - start)
            diff = pred - target
            maes.append(float(torch.mean(torch.abs(diff))))
            rmses.append(float(torch.sqrt(torch.mean(diff ** 2))))
            max_errors.append(float(torch.abs(pred.amax() - target.amax())))
            avg_errors.append(float(torch.abs(pred.mean() - target.mean())))
            denom = torch.clamp(torch.abs(target), min=1.0e-6)
            mapes.append(float(torch.mean(torch.abs(diff) / denom) * 100.0))

    metrics = {
        "num_instances": len(dataset),
        "field_mae": float(np.mean(maes)),
        "field_rmse": float(np.mean(rmses)),
        "t_max_abs_error": float(np.mean(max_errors)),
        "t_avg_abs_error": float(np.mean(avg_errors)),
        "mape_percent": float(np.mean(mapes)),
        "runtime_per_instance_sec": float(np.mean(runtimes)),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
