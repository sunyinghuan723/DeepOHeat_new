# Package Thermal Surrogate

This directory contains a lightweight package-level DeepOHeat-style surrogate
for ChipletPart thermal instances. It is separate from the original
`2d_power_map` demo.

The model follows an operator-learning shape:

```text
G_theta(X, y) -> T(y)
```

`X` is the multi-channel package tensor dumped by ChipletPart and `y` is a
normalized grid coordinate. A CNN branch encodes `X`, a trunk MLP encodes `y`,
and their product predicts the temperature field.

Device selection supports `cpu`, `cuda`, `cuda:0`, `cuda:1`, and `auto`.
`auto` uses `cuda:0` when PyTorch can initialize CUDA and otherwise falls back
to CPU with a warning. Explicit CUDA requests fail if the device is unavailable.

Quick training:

```bash
python ../../DeepOHeat/package_thermal/train.py \
  --manifest /tmp/chipletpart_thermal_dataset/manifest_labeled.jsonl \
  --out_dir /tmp/deepoheat_package_run \
  --epochs 5 --batch_size 2 \
  --grid_x 32 --grid_y 32 \
  --device auto
```

Evaluation:

```bash
python ../../DeepOHeat/package_thermal/evaluate.py \
  --manifest /tmp/chipletpart_thermal_dataset/manifest_labeled.jsonl \
  --checkpoint /tmp/deepoheat_package_run/checkpoint_best.pt \
  --device auto
```

Inference adapter:

```bash
python ../../DeepOHeat/package_thermal/infer_package.py \
  --instance /tmp/chipletpart_thermal_dataset/raw/seed_1/example.json \
  --model /tmp/deepoheat_package_run/checkpoint_best.pt \
  --output /tmp/package_result.json \
  --device auto
```

The labels used in the mini pipeline come from ChipletPart's simplified
reference solver. They are suitable for developing the surrogate flow, not for
thermal signoff.

Medium pilot training used:

```bash
python package_thermal/train.py \
  --train_manifest /tmp/chipletpart_thermal_dataset_v2/manifest_train.jsonl \
  --val_manifest /tmp/chipletpart_thermal_dataset_v2/manifest_val.jsonl \
  --test_manifest /tmp/chipletpart_thermal_dataset_v2/manifest_test.jsonl \
  --out_dir /tmp/deepoheat_package_run_v2 \
  --epochs 50 --batch_size 16 \
  --grid_x 32 --grid_y 32 \
  --device auto \
  --branch_dim 96 --trunk_dim 96 --hidden_dim 160 --num_layers 3 \
  --dropout 0.05 --seed 2026
```

On this server CUDA was not available to PyTorch because the NVIDIA driver was
not reachable, so `auto` used CPU. The run still completed in about 21 seconds
for 230 pilot instances. Test metrics were field MAE `8.78 K`, field RMSE
`9.74 K`, T_max absolute error `11.23 K`, and T_avg absolute error `7.86 K`.
