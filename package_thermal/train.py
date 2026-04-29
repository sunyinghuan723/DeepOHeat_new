#!/usr/bin/env python3
"""Train a small package-level DeepOHeat-style surrogate."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_CHANNELS, PackageThermalDataset, normalized_coords
from device import device_metadata, log_device, resolve_device
from model import PackageThermalDeepONet


def _make_dataset(manifest: Path, split: str) -> PackageThermalDataset:
    try:
        return PackageThermalDataset(manifest, split=split)
    except ValueError:
        return PackageThermalDataset(manifest, split=None)


def _split_count(manifest: Path, split: str) -> int:
    try:
        return len(PackageThermalDataset(manifest, split=split))
    except ValueError:
        return 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _dataset_from_args(args: argparse.Namespace, split: str) -> PackageThermalDataset:
    explicit = getattr(args, f"{split}_manifest")
    if explicit:
        return PackageThermalDataset(explicit, split=None)
    if split == "test":
        try:
            return PackageThermalDataset(args.manifest, split="test")
        except ValueError:
            return PackageThermalDataset(args.manifest, split="val")
    return _make_dataset(args.manifest, split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--train_manifest", type=Path)
    parser.add_argument("--val_manifest", type=Path)
    parser.add_argument("--test_manifest", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grid_x", type=int, default=32)
    parser.add_argument("--grid_y", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lambda_peak", "--beta_peak", dest="lambda_peak", type=float, default=0.05)
    parser.add_argument("--lambda_mean", "--beta_avg", dest="lambda_mean", type=float, default=0.01)
    parser.add_argument("--beta_phys", type=float, default=0.0)
    parser.add_argument("--branch_dim", type=int, default=64)
    parser.add_argument("--trunk_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_batchnorm", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    args = parser.parse_args()

    if not args.manifest and not (args.train_manifest and args.val_manifest):
        raise SystemExit("provide --manifest or both --train_manifest and --val_manifest")
    if args.beta_phys != 0.0:
        print("[PACKAGE-THERMAL] beta_phys scaffold is present; residual loss is TODO")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(args.out_dir / "training_log.txt")
    _set_seed(args.seed)
    device = resolve_device(args.device)
    log_device("[PACKAGE-THERMAL]", device)
    logger.log("[PACKAGE-THERMAL] device_metadata=" + json.dumps(device_metadata(device), sort_keys=True))
    config = {
        "model_type": "package_thermal",
        "channel_names": DEFAULT_CHANNELS,
        "grid_x": args.grid_x,
        "grid_y": args.grid_y,
        "feature_dim": args.branch_dim,
        "branch_dim": args.branch_dim,
        "trunk_dim": args.trunk_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_batchnorm": args.use_batchnorm,
        "lambda_peak": args.lambda_peak,
        "lambda_mean": args.lambda_mean,
        "beta_phys": args.beta_phys,
        "seed": args.seed,
        "device_requested": args.device,
        **device_metadata(device),
    }
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_dataset = _dataset_from_args(args, "train")
    val_dataset = _dataset_from_args(args, "val")
    test_dataset = _dataset_from_args(args, "test") if (args.test_manifest or args.manifest) else None
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = PackageThermalDeepONet(
        len(DEFAULT_CHANNELS),
        branch_dim=args.branch_dim,
        trunk_dim=args.trunk_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batchnorm=args.use_batchnorm,
    ).to(device)
    coords = normalized_coords(args.grid_y, args.grid_x).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    logger.log(
        "[PACKAGE-THERMAL] "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} "
        f"train_size={len(train_dataset)} val_size={len(val_dataset)} "
        f"test_size={len(test_dataset) if test_dataset is not None else 0} "
        f"seed={args.seed}"
    )
    logger.log(
        "[PACKAGE-THERMAL] split_counts="
        + json.dumps(
            {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset) if test_dataset is not None else 0,
            },
            sort_keys=True,
        )
    )

    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_mse", "runtime_sec"])
        writer.writeheader()

    best_val = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
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
        epoch_runtime = time.perf_counter() - epoch_start
        logger.log(
            f"[PACKAGE-THERMAL] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_mse={val_mse:.6f} runtime_sec={epoch_runtime:.3f}"
        )
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "train_loss", "val_mse", "runtime_sec"]
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_mse": val_mse,
                    "runtime_sec": epoch_runtime,
                }
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_mse": val_mse,
        }
        torch.save(checkpoint, args.out_dir / "checkpoint_last.pt")
        if val_mse <= best_val:
            best_val = val_mse
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(checkpoint, args.out_dir / "checkpoint_best.pt")
        else:
            epochs_since_improve += 1
        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            logger.log(
                f"[PACKAGE-THERMAL] early_stop epoch={epoch} "
                f"best_epoch={best_epoch} best_val_mse={best_val:.6f}"
            )
            break

    runtime = time.perf_counter() - start
    with (args.out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_mse": best_val,
                "best_epoch": best_epoch,
                "runtime_sec": runtime,
                "checkpoint_best": str(args.out_dir / "checkpoint_best.pt"),
                "checkpoint_last": str(args.out_dir / "checkpoint_last.pt"),
                "dataset_size": {
                    "train": len(train_dataset),
                    "val": len(val_dataset),
                    "test": len(test_dataset) if test_dataset is not None else 0,
                },
                **device_metadata(device),
            },
            f,
            indent=2,
        )
    logger.log(
        f"[PACKAGE-THERMAL] training complete runtime_sec={runtime:.3f} "
        f"best_epoch={best_epoch} checkpoint_best={args.out_dir / 'checkpoint_best.pt'}"
    )
    logger.close()


if __name__ == "__main__":
    main()
