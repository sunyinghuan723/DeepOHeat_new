#!/usr/bin/env python3
"""Inference adapter for package-level ChipletPart thermal surrogate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dataset import DEFAULT_CHANNELS, load_instance_tensor, normalized_coords
from model import PackageThermalDeepONet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dump_field", action="store_true")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"package_thermal checkpoint not found: {args.model}")
    if not args.instance.exists():
        raise SystemExit(f"thermal instance not found: {args.instance}")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint.get("config", {})
    if args.config and args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            config.update(json.load(f))
    channels = config.get("channel_names", DEFAULT_CHANNELS)
    tensor = load_instance_tensor(args.instance, channels)
    grid_y, grid_x = tensor.shape[1], tensor.shape[2]

    model = PackageThermalDeepONet(
        len(channels),
        feature_dim=int(config.get("feature_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.from_numpy(tensor).unsqueeze(0).float().to(device)
    coords = normalized_coords(grid_y, grid_x).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        field = model(x, coords).reshape(grid_y, grid_x).detach().cpu().numpy()
    runtime = time.perf_counter() - start

    field_path = ""
    if args.dump_field:
        field_path = str(args.output.with_suffix(".field.npz"))
        np.savez_compressed(field_path, temperature_map=field.astype(np.float32))

    result = {
        "t_max": float(np.max(field)),
        "t_avg": float(np.mean(field)),
        "field_path": field_path,
        "runtime_sec": float(runtime),
        "model_type": "package_thermal",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[PACKAGE-THERMAL] inference failed: {exc}", file=sys.stderr)
        raise
