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
from device import device_metadata, resolve_device
from model import PackageThermalDeepONet


class PackageThermalPredictor:
    """Persistent package-level DeepONet inference state."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not args.model.exists():
            raise FileNotFoundError(f"package_thermal checkpoint not found: {args.model}")
        self.device = resolve_device(args.device)
        checkpoint = torch.load(args.model, map_location=self.device)
        self.config = checkpoint.get("config", {})
        if args.config and args.config.exists():
            with args.config.open("r", encoding="utf-8") as f:
                self.config.update(json.load(f))
        self.channels = self.config.get("channel_names", DEFAULT_CHANNELS)
        self.model = PackageThermalDeepONet(
            len(self.channels),
            feature_dim=int(self.config.get("feature_dim", 64)),
            branch_dim=int(self.config.get("branch_dim", self.config.get("feature_dim", 64))),
            trunk_dim=int(self.config.get("trunk_dim", self.config.get("feature_dim", 64))),
            hidden_dim=int(self.config.get("hidden_dim", 128)),
            num_layers=int(self.config.get("num_layers", 2)),
            dropout=float(self.config.get("dropout", 0.0)),
            use_batchnorm=bool(self.config.get("use_batchnorm", False)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.coords_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _coords(self, grid_y: int, grid_x: int) -> torch.Tensor:
        key = (grid_y, grid_x)
        coords = self.coords_cache.get(key)
        if coords is None:
            coords = normalized_coords(grid_y, grid_x).to(self.device)
            self.coords_cache[key] = coords
        return coords

    def predict(self, instance: Path, output: Path, *, dump_field: bool = False) -> dict:
        if not instance.exists():
            raise FileNotFoundError(f"thermal instance not found: {instance}")
        tensor = load_instance_tensor(instance, self.channels)
        grid_y, grid_x = tensor.shape[1], tensor.shape[2]

        x = torch.from_numpy(tensor).unsqueeze(0).float().to(self.device)
        coords = self._coords(grid_y, grid_x)
        start = time.perf_counter()
        with torch.no_grad():
            field = self.model(x, coords).reshape(grid_y, grid_x).detach().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - start

        field_path = ""
        if dump_field:
            field_path = str(output.with_suffix(".field.npz"))
            np.savez_compressed(field_path, temperature_map=field.astype(np.float32))

        result = {
            "t_max": float(np.max(field)),
            "t_avg": float(np.mean(field)),
            "field_path": field_path,
            "runtime_sec": float(runtime),
            "model_type": "package_thermal",
            "device": str(self.device),
            **{k: v for k, v in device_metadata(self.device).items() if k != "device"},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result


def infer(args: argparse.Namespace) -> None:
    predictor = PackageThermalPredictor(args)
    predictor.predict(args.instance, args.output, dump_field=args.dump_field)


def serve(args: argparse.Namespace) -> None:
    predictor = PackageThermalPredictor(args)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request.get("shutdown"):
                break
            result = predictor.predict(
                Path(request["instance"]),
                Path(request["output"]),
                dump_field=bool(request.get("dump_field", False)),
            )
            print(json.dumps({"ok": True, **result}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dump_field", action="store_true")
    args = parser.parse_args()

    if args.server:
        serve(args)
    else:
        if args.instance is None or args.output is None:
            parser.error("--instance and --output are required unless --server is set")
        infer(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[PACKAGE-THERMAL] inference failed: {exc}", file=sys.stderr)
        raise
