#!/usr/bin/env python3
"""Evaluate a package-level surrogate checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_CHANNELS, PackageThermalDataset, normalized_coords
from device import device_metadata, log_device, resolve_device
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
        branch_dim=int(config.get("branch_dim", config.get("feature_dim", 64))),
        trunk_dim=int(config.get("trunk_dim", config.get("feature_dim", 64))),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.0)),
        use_batchnorm=bool(config.get("use_batchnorm", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default=None)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--metrics_json", type=Path)
    parser.add_argument("--errors_csv", type=Path)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--worst_k", type=int, default=5)
    args = parser.parse_args()

    device = resolve_device(args.device)
    log_device("[PACKAGE-THERMAL]", device)
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
    error_rows = []
    prediction_dir = None
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir = args.out_dir / "predictions"
        if args.save_predictions:
            prediction_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].float().to(device)
            target = batch["temperature"].float().to(device)
            start = time.perf_counter()
            pred = model(x, coords).reshape_as(target)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtimes.append(time.perf_counter() - start)
            diff = pred - target
            mae = float(torch.mean(torch.abs(diff)))
            rmse = float(torch.sqrt(torch.mean(diff ** 2)))
            max_error = float(torch.abs(pred.amax() - target.amax()))
            avg_error = float(torch.abs(pred.mean() - target.mean()))
            maes.append(mae)
            rmses.append(rmse)
            max_errors.append(max_error)
            avg_errors.append(avg_error)
            denom = torch.clamp(torch.abs(target), min=1.0e-6)
            mape = float(torch.mean(torch.abs(diff) / denom) * 100.0)
            mapes.append(mape)
            instance_id = batch["instance_id"][0]
            pred_tmax = float(pred.amax())
            target_tmax = float(target.amax())
            pred_tavg = float(pred.mean())
            target_tavg = float(target.mean())
            prediction_path = ""
            if args.save_predictions and prediction_dir is not None:
                prediction_path = str(prediction_dir / f"{instance_id}_pred.npz")
                np.savez_compressed(
                    prediction_path,
                    temperature_map=pred.detach().cpu().numpy().reshape(grid_y, grid_x).astype(np.float32),
                    t_max=np.asarray(pred_tmax, dtype=np.float32),
                    t_avg=np.asarray(pred_tavg, dtype=np.float32),
                )
            error_rows.append(
                {
                    "instance_id": instance_id,
                    "field_mae": mae,
                    "field_rmse": rmse,
                    "t_max_abs_error": max_error,
                    "t_avg_abs_error": avg_error,
                    "mape_percent": mape,
                    "pred_t_max": pred_tmax,
                    "label_t_max": target_tmax,
                    "pred_t_avg": pred_tavg,
                    "label_t_avg": target_tavg,
                    "runtime_sec": runtimes[-1],
                    "prediction_path": prediction_path,
                }
            )

    worst = sorted(error_rows, key=lambda row: row["t_max_abs_error"], reverse=True)[: args.worst_k]
    metrics = {
        "num_instances": len(dataset),
        "field_mae": float(np.mean(maes)),
        "field_rmse": float(np.mean(rmses)),
        "t_max_abs_error": float(np.mean(max_errors)),
        "t_avg_abs_error": float(np.mean(avg_errors)),
        "mape_percent": float(np.mean(mapes)),
        "runtime_per_instance_sec": float(np.mean(runtimes)),
        "runtime_p50_sec": float(np.percentile(runtimes, 50)),
        "runtime_p95_sec": float(np.percentile(runtimes, 95)),
        "checkpoint": str(args.checkpoint),
        "worst_tmax_error_instance_ids": [row["instance_id"] for row in worst],
        **device_metadata(device),
    }
    metrics_json = args.metrics_json or (args.out_dir / "metrics.json" if args.out_dir else None)
    errors_csv = args.errors_csv or (args.out_dir / "per_instance_errors.csv" if args.out_dir else None)
    if metrics_json:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with metrics_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    if errors_csv:
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        with errors_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "instance_id",
                "field_mae",
                "field_rmse",
                "t_max_abs_error",
                "t_avg_abs_error",
                "mape_percent",
                "pred_t_max",
                "label_t_max",
                "pred_t_avg",
                "label_t_avg",
                "runtime_sec",
                "prediction_path",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
