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

Quick training:

```bash
python ../../DeepOHeat/package_thermal/train.py \
  --manifest /tmp/chipletpart_thermal_dataset/manifest_labeled.jsonl \
  --out_dir /tmp/deepoheat_package_run \
  --epochs 5 --batch_size 2 \
  --grid_x 32 --grid_y 32 \
  --device cpu
```

Evaluation:

```bash
python ../../DeepOHeat/package_thermal/evaluate.py \
  --manifest /tmp/chipletpart_thermal_dataset/manifest_labeled.jsonl \
  --checkpoint /tmp/deepoheat_package_run/checkpoint_best.pt \
  --device cpu
```

Inference adapter:

```bash
python ../../DeepOHeat/package_thermal/infer_package.py \
  --instance /tmp/chipletpart_thermal_dataset/raw/seed_1/example.json \
  --model /tmp/deepoheat_package_run/checkpoint_best.pt \
  --output /tmp/package_result.json \
  --device cpu
```

The labels used in the mini pipeline come from ChipletPart's simplified
reference solver. They are suitable for developing the surrogate flow, not for
thermal signoff.
