
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset_v1"
    / "manifest.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_lr001_30epochs"
    / "lr_0.001"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradient_learning_diagnostics"
)

MAX_STUDIES = 8

PLANE_NAMES = ["sagittal", "coronal", "axial"]

DEVICE = torch.device("cpu")


# ------------------------------------------------------------
# Imports from the project
# ------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def tensor_stats(tensor: torch.Tensor) -> dict:
    """Return basic numerical statistics for a tensor."""

    values = tensor.detach().float()

    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "finite": bool(torch.isfinite(values).all().item()),
    }


def gradient_norm(parameter: torch.nn.Parameter) -> float:
    """Return the L2 norm of a parameter's gradient."""

    if parameter.grad is None:
        return 0.0

    return float(parameter.grad.detach().norm().item())


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to CSV without requiring pandas."""

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------

def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Load a model checkpoint without changing the checkpoint file."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Supports a checkpoint containing only the state dict.
            state_dict = checkpoint
    else:
        raise ValueError(
            "Unexpected checkpoint format. "
            "Inspect the checkpoint structure before proceeding."
        )

    model.load_state_dict(state_dict, strict=True)


# ------------------------------------------------------------
# Main diagnostic
# ------------------------------------------------------------

def main() -> None:

    # Never overwrite previous diagnostic results.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output folder already exists:\n{OUTPUT_DIR}\n\n"
            "Rename the existing folder or choose a new OUTPUT_DIR "
            "before running again."
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True)

    print("=" * 72)
    print("GRADIENT AND LEARNING DIAGNOSTIC")
    print("=" * 72)
    print(f"Project:    {PROJECT_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Manifest:   {MANIFEST_PATH}")
    print(f"Device:     {DEVICE}")
    print()

    # --------------------------------------------------------
    # 1. Load model and checkpoint
    # --------------------------------------------------------

    model = Knee3DCNN().to(DEVICE)
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()

    print("Checkpoint loaded successfully.")
    print()

    # --------------------------------------------------------
    # 2. Load the training dataset
    # --------------------------------------------------------

    dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )

    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    count = min(MAX_STUDIES, len(dataset))

    print(f"Training studies available: {len(dataset)}")
    print(f"Studies examined:            {count}")
    print()

    # --------------------------------------------------------
    # 3. Capture convolutional and linear activations
    # --------------------------------------------------------

    activation_records = []
    hook_handles = []

    def make_hook(module_name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                stats = tensor_stats(output)
                stats["module"] = module_name
                activation_records.append(stats)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            hook_handles.append(
                module.register_forward_hook(make_hook(name))
            )

    # --------------------------------------------------------
    # 4. Measure input and prediction variation
    # --------------------------------------------------------

    input_records = []
    prediction_records = []
    target_records = []

    try:
        with torch.no_grad():
            for index in range(count):
                study_uid, images, labels = dataset[index]

                images = torch.as_tensor(
                    images, dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)

                labels = torch.as_tensor(
                    labels, dtype=torch.float32
                ).reshape(-1)

                if images.ndim != 5 or images.shape[1] != 3:
                    raise ValueError(
                        "Expected image tensor shape "
                        "(1, 3, D, H, W); received "
                        f"{tuple(images.shape)}"
                    )

                # Each branch receives one volume:
                # (batch, depth, height, width).
                volumes = [
                    images[:, plane_index]
                    for plane_index in range(3)
                ]

                row = {"study_instance_uid": study_uid,"study_index": index}

                for plane_index, plane_name in enumerate(PLANE_NAMES):
                    stats = tensor_stats(volumes[plane_index])
                    row[f"{plane_name}_mean"] = stats["mean"]
                    row[f"{plane_name}_std"] = stats["std"]
                    row[f"{plane_name}_min"] = stats["min"]
                    row[f"{plane_name}_max"] = stats["max"]
                    row[f"{plane_name}_finite"] = stats["finite"]

                input_records.append(row)

                logits = model(*volumes)
                probabilities = torch.sigmoid(logits).reshape(-1)

                pred_row = {"study_index": index}

                for target_index, probability in enumerate(
                    probabilities.tolist()
                ):
                    pred_row[f"logit_{target_index}"] = float(
                        logits.reshape(-1)[target_index].item()
                    )
                    pred_row[f"probability_{target_index}"] = float(
                        probability
                    )
                    pred_row[f"label_{target_index}"] = float(
                        labels[target_index].item()
                    )

                prediction_records.append(pred_row)

                target_records.append(labels.numpy().astype(float))

    finally:
        for handle in hook_handles:
            handle.remove()

    write_csv(OUTPUT_DIR / "input_statistics.csv", input_records)
    write_csv(OUTPUT_DIR / "predictions.csv", prediction_records)

    # Aggregate variation across studies.
    prediction_summary = []

    for target_index in range(12):
        logits = np.array(
            [
                row[f"logit_{target_index}"]
                for row in prediction_records
            ],
            dtype=np.float64,
        )

        probabilities = np.array(
            [
                row[f"probability_{target_index}"]
                for row in prediction_records
            ],
            dtype=np.float64,
        )

        labels = np.array(
            [row[target_index] for row in target_records],
            dtype=np.float64,
        )

        prediction_summary.append({
            "target_index": target_index,
            "label_mean": float(labels.mean()),
            "label_std": float(labels.std()),
            "logit_mean": float(logits.mean()),
            "logit_std": float(logits.std()),
            "logit_min": float(logits.min()),
            "logit_max": float(logits.max()),
            "probability_mean": float(probabilities.mean()),
            "probability_std": float(probabilities.std()),
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
        })

    write_csv(
        OUTPUT_DIR / "prediction_variation.csv",
        prediction_summary,
    )

    # Summarize the activations collected during inference.
    activation_summary = {}

    for record in activation_records:
        name = record["module"]
        activation_summary.setdefault(name, []).append(record)

    activation_rows = []

    for name, records in activation_summary.items():
        activation_rows.append({
            "module": name,
            "forward_calls": len(records),
            "mean_activation_mean": float(
                np.mean([r["mean"] for r in records])
            ),
            "mean_activation_std": float(
                np.mean([r["std"] for r in records])
            ),
            "mean_activation_min": float(
                np.mean([r["min"] for r in records])
            ),
            "mean_activation_max": float(
                np.mean([r["max"] for r in records])
            ),
            "all_finite": all(r["finite"] for r in records),
        })

    write_csv(
        OUTPUT_DIR / "activation_statistics.csv",
        activation_rows,
    )

    # --------------------------------------------------------
    # 5. Compute gradients without updating parameters
    # --------------------------------------------------------

    # Re-register hooks to capture activations during one
    # gradient-enabled forward pass.
    gradient_activation_records = []
    gradient_hooks = []

    def make_gradient_hook(module_name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                gradient_activation_records.append({
                    "module": module_name,
                    **tensor_stats(output),
                })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            gradient_hooks.append(
                module.register_forward_hook(
                    make_gradient_hook(name)
                )
            )

    model.zero_grad(set_to_none=True)
    model.train()

    try:
        images, labels = dataset[0]

        images = torch.as_tensor(
            images, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        labels = torch.as_tensor(
            labels, dtype=torch.float32
        ).reshape(1, -1).to(DEVICE)

        volumes = [
            images[:, plane_index]
            for plane_index in range(3)
        ]

        logits = model(*volumes)

        if logits.shape != labels.shape:
            raise ValueError(
                f"Logits shape {tuple(logits.shape)} does not match "
                f"labels shape {tuple(labels.shape)}."
            )

        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(logits, labels)

        loss.backward()

        gradient_rows = []

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            gradient_rows.append({
                "parameter": name,
                "shape": str(tuple(parameter.shape)),
                "gradient_norm": gradient_norm(parameter),
                "gradient_is_none": parameter.grad is None,
                "gradient_finite": (
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all().item())
                ),
            })

        write_csv(
            OUTPUT_DIR / "gradient_statistics.csv",
            gradient_rows,
        )

        write_csv(
            OUTPUT_DIR / "gradient_forward_activations.csv",
            gradient_activation_records,
        )

        gradient_summary = {
            "study_index": 0,
            "loss": float(loss.item()),
            "logit_mean": float(logits.detach().mean().item()),
            "logit_std": float(
                logits.detach().std(unbiased=False).item()
            ),
            "probability_mean": float(
                torch.sigmoid(logits.detach()).mean().item()
            ),
            "parameters_with_gradients": sum(
                p.grad is not None for p in model.parameters()
            ),
            "parameters_with_zero_gradient": sum(
                p.grad is not None
                and torch.count_nonzero(p.grad).item() == 0
                for p in model.parameters()
            ),
            "parameters_with_nonfinite_gradients": sum(
                p.grad is not None
                and not torch.isfinite(p.grad).all().item()
                for p in model.parameters()
            ),
        }

        with (OUTPUT_DIR / "gradient_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(gradient_summary, file, indent=2)

    finally:
        for handle in gradient_hooks:
            handle.remove()

        # No optimizer step is performed. Restore evaluation mode.
        model.zero_grad(set_to_none=True)
        model.eval()

    # --------------------------------------------------------
    # 6. Save run metadata
    # --------------------------------------------------------

    metadata = {
        "checkpoint": str(CHECKPOINT_PATH),
        "manifest": str(MANIFEST_PATH),
        "device": str(DEVICE),
        "training_studies_examined": count,
        "gradient_study_index": 0,
        "optimizer_step_performed": False,
        "checkpoint_modified": False,
        "target_order": [
            "acl",
            "mcl",
            "medial_meniscus",
            "lateral_meniscus",
            "medial_oa",
            "lateral_oa",
            "pf_oa",
            "effusion",
            "synovitis",
            "bakers",
            "contusion",
            "fracture",
        ],
    }

    with (OUTPUT_DIR / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    # --------------------------------------------------------
    # 7. Print a concise report
    # --------------------------------------------------------

    print("=" * 72)
    print("INPUT VARIATION")
    print("=" * 72)

    for plane_name in PLANE_NAMES:
        std_values = [
            row[f"{plane_name}_std"] for row in input_records
        ]

        print(
            f"{plane_name:10s} "
            f"average within-volume intensity std = "
            f"{np.mean(std_values):.6f}"
        )

    print()
    print("=" * 72)
    print("PREDICTION VARIATION ACROSS STUDIES")
    print("=" * 72)

    for row in prediction_summary:
        print(
            f"Target {row['target_index']:2d}: "
            f"logit std={row['logit_std']:.6f}, "
            f"probability std={row['probability_std']:.6f}, "
            f"probability range="
            f"[{row['probability_min']:.4f}, "
            f"{row['probability_max']:.4f}]"
        )

    print()
    print("=" * 72)
    print("GRADIENT SUMMARY")
    print("=" * 72)
    print(f"Loss: {gradient_summary['loss']:.6f}")
    print(
        "Parameters with gradients: "
        f"{gradient_summary['parameters_with_gradients']}"
    )
    print(
        "Parameters with zero gradients: "
        f"{gradient_summary['parameters_with_zero_gradient']}"
    )
    print(
        "Parameters with non-finite gradients: "
        f"{gradient_summary['parameters_with_nonfinite_gradients']}"
    )

    print()
    print(f"Results saved to:\n{OUTPUT_DIR}")
    print("No optimizer step was performed.")
    print("The original checkpoint was not modified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
