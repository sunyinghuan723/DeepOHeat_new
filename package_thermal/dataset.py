"""Dataset utilities for package-level ChipletPart thermal instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_CHANNELS = [
    "package_domain",
    "chiplet_footprint",
    "chiplet_boundary",
    "power_density_w_per_mm2",
    "silicon_material",
    "interposer_material",
    "tim_material",
    "package_material",
    "ambient_temperature",
    "heat_transfer_coefficient",
]


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_instance_tensor(path: str | Path, channel_names: Iterable[str] | None = None) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        instance = json.load(f)
    grid_y = int(instance.get("grid_y", instance.get("grid", {}).get("y")))
    grid_x = int(instance.get("grid_x", instance.get("grid", {}).get("x")))
    channels = instance["channels"]
    names = list(channel_names or instance.get("channel_names") or DEFAULT_CHANNELS)
    arrays = []
    for name in names:
        if name not in channels:
            raise KeyError(f"{path} missing channel {name}")
        arrays.append(np.asarray(channels[name], dtype=np.float32).reshape(grid_y, grid_x))
    tensor = np.stack(arrays, axis=0)
    return tensor


def load_label(path: str | Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["temperature_map"], dtype=np.float32)


def normalized_coords(grid_y: int, grid_x: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, grid_y)
    x = torch.linspace(0.0, 1.0, grid_x)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


class PackageThermalDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str | None = None,
        channel_names: Iterable[str] | None = None,
    ) -> None:
        self.records = read_manifest(manifest)
        if split is not None:
            filtered = [record for record in self.records if record.get("split") == split]
            if filtered:
                self.records = filtered
        if not self.records:
            raise ValueError(f"no records found in {manifest} for split={split}")
        self.channel_names = list(channel_names or DEFAULT_CHANNELS)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if not record.get("label_path"):
            raise ValueError(f"record {record.get('instance_id')} has no label_path")
        tensor = load_instance_tensor(record["json_path"], self.channel_names)
        label = load_label(record["label_path"])
        return {
            "x": torch.from_numpy(tensor),
            "temperature": torch.from_numpy(label),
            "instance_id": record.get("instance_id", str(index)),
        }
