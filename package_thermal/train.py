#!/usr/bin/env python3
"""Train a small package-level DeepOHeat-style surrogate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_CHANNELS, PackageThermalDataset, normalized_coords
from model import PackageThermalDeepONet


def _make_dataset(manifest: Path, split: str) -> PackageThermalDataset:
    try:
        return PackageThermalDataset(manifest, split=split)
    except ValueError:
        return PackageThermalDataset(manifest, split=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grid_x", type=int, default=32)
    parser.add_argument("--grid_y", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lambda_peak", type=float, default=0.05)
    parser.add_argument("--lambda_mean", type=float, default=0.01)
    parser.add_argument("--beta_phys", type=float, default=0.0)
    args = parser.parse_args()

    if args.beta_phys != 0.0:
        print("[PACKAGE-THERMAL] beta_phys scaffold is present; residual loss is TODO")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "package_thermal",
        "channel_names": DEFAULT_CHANNELS,
        "grid_x": args.grid_x,
        "grid_y": args.grid_y,
        "feature_dim": 64,
        "hidden_dim": 128,
        "lambda_peak": args.lambda_peak,
        "lambda_mean": args.lambda_mean,
        "beta_phys": args.beta_phys,
    }
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_dataset = _make_dataset(args.manifest, "train")
    val_dataset = _make_dataset(args.manifest, "val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = PackageThermalDeepONet(len(DEFAULT_CHANNELS)).to(device)
    coords = normalized_coords(args.grid_y, args.grid_x).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].float().to(device)
            target = batch["temperature"].float().to(device)
            pred = model(x, coords).reshape_as(target)
            field_loss = torch.mean((pred - target) ** 2)
            peak_loss = torch.mean((pred.amax(dim=(1, 2)) - target.amax(dim=(1, 2))) ** 2)
            mean_loss = torch.mean((pred.mean(dim=(1, 2)) - target.mean(dim=(1, 2))) ** 2)
            loss = field_loss + args.lambda_peak * peak_loss + args.lambda_mean * mean_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * x.shape[0]
        train_loss /= len(train_dataset)

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].float().to(device)
                target = batch["temperature"].float().to(device)
                pred = model(x, coords).reshape_as(target)
                val_mse += float(torch.mean((pred - target) ** 2)) * x.shape[0]
        val_mse /= len(val_dataset)
        print(f"[PACKAGE-THERMAL] epoch={epoch} train_loss={train_loss:.6f} val_mse={val_mse:.6f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_mse": val_mse,
        }
        torch.save(checkpoint, args.out_dir / "checkpoint_last.pt")
        if val_mse <= best_val:
            best_val = val_mse
            torch.save(checkpoint, args.out_dir / "checkpoint_best.pt")

    runtime = time.perf_counter() - start
    with (args.out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_mse": best_val, "runtime_sec": runtime}, f, indent=2)
    print(f"[PACKAGE-THERMAL] training complete runtime_sec={runtime:.3f}")


if __name__ == "__main__":
    main()
