import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "checkpoints" / "training_results.json"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"
REPORT_PATH = OUTPUT_DIR / "baseline_evaluation.txt"

def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Training results not found: {RESULTS_PATH}")

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "RSNA KNEE AI — BASELINE EVALUATION",
        "=" * 65,
        f"Run timestamp (UTC): {results.get('run_timestamp_utc', 'Unknown')}",
        f"Device: {results.get('device', 'Unknown')}",
        f"Epochs configured: {results.get('epochs', 'Unknown')}",
        f"Batch size: {results.get('batch_size', 'Unknown')}",
        f"Learning rate: {results.get('learning_rate', 'Unknown')}",
        f"Random seed: {results.get('random_seed', 'Unknown')}",
        "",
        "DATASET SUMMARY",
        "-" * 65,
        f"Fully labeled studies: {results.get('total_fully_labeled_studies', 'Unknown')}",
        f"Training studies: {results.get('training_study_count', 'Unknown')}",
        f"Validation studies: {results.get('validation_study_count', 'Unknown')}",
        f"Validation fraction: {results.get('validation_fraction', 'Unknown')}",
        f"Target spacing: {results.get('target_spacing', 'Unknown')}",
        f"Target shape: {results.get('target_shape', 'Unknown')}",
        "",
        "TRAINING AND VALIDATION LOSS",
        "-" * 65
    ]

    training_losses = results.get("training_losses", [])
    for epoch, loss in enumerate(training_losses, start=1):
        lines.append(f"Epoch {epoch} training loss: {loss:.6f}")

    validation_loss = results.get("validation_loss")
    if validation_loss is not None:
        lines.append(f"Validation loss: {validation_loss:.6f}")

    lines.extend([
        "",
        "PER-LABEL METRICS",
        "-" * 65
    ])

    metrics = results.get("per_label_metrics", {})
    if isinstance(metrics, dict):
        for label, values in metrics.items():
            lines.append(f"\n{label}")
            if isinstance(values, dict):
                for metric, value in values.items():
                    lines.append(f"  {metric}: {value}")
            else:
                lines.append(f"  {values}")
    else:
        lines.append("Unexpected per_label_metrics format.")

    lines.extend([
        "",
        "CHECKPOINT",
        "-" * 65,
        f"Checkpoint path: {results.get('checkpoint_path', 'Unknown')}",
        "",
        "INTERPRETATION",
        "-" * 65,
        "This is an exploratory baseline, not a clinically validated model.",
        "The validation set is small, so per-label metrics are unstable.",
        "Zero true positives can indicate that the model is predicting",
        "all-negative results for a label at the current threshold.",
        "Review class balance and prediction probabilities before",
        "changing thresholds or drawing performance conclusions.",
        "",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("BASELINE EVALUATION COMPLETED")
    print(f"Report: {REPORT_PATH}")
    print(f"Report exists: {REPORT_PATH.exists()}")

if __name__ == "__main__":
    main()

