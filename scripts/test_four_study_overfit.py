
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from ish_knee.dataset.knee_mri_dataset import KneeMRIDataset
from ish_knee.model.knee_3d_cnn import Knee3DCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "processed_dataset_v1" / "manifest.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_phase3d"

RESULTS_PATH = OUTPUT_DIR / "four_study_overfit_results.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "four_study_overfit_test.pt"

SEED = 42
STUDY_COUNT = 4
STEPS = 200
LEARNING_RATE = 1e-3
REPORT_EVERY = 10


def main() -> None:
    if RESULTS_PATH.exists() or CHECKPOINT_PATH.exists():
        raise FileExistsError(
            "An output file already exists:\n"
            f"{RESULTS_PATH}\n{CHECKPOINT_PATH}\n"
            "Rename the existing file or change the output paths. "
            "Nothing has been overwritten."
        )

    torch.manual_seed(SEED)

    dataset = KneeMRIDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )

    if len(dataset) < STUDY_COUNT:
        raise ValueError("Not enough training studies.")

    # Use a fixed subset so every optimization step sees the same data.
    samples = [dataset[i] for i in range(STUDY_COUNT)]

    images = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])

    print("=" * 70)
    print("FOUR-STUDY OVERFITTING TEST")
    print("=" * 70)
    print("Selected dataset indices:", list(range(STUDY_COUNT)))
    print("Images shape:", tuple(images.shape))
    print("Labels shape:", tuple(labels.shape))
    print("Learning rate:", LEARNING_RATE)
    print("Optimization steps:", STEPS)

    model = Knee3DCNN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    def evaluate(step: int) -> float:
        model.eval()
        with torch.no_grad():
            logits = model(
                images[:, 0],
                images[:, 1],
                images[:, 2],
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        value = float(loss.item())
        history.append({"step": step, "bce_loss": value})
        return value

    initial_loss = evaluate(0)
    print(f"\nInitial BCE: {initial_loss:.6f}")

    model.train()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            images[:, 0],
            images[:, 1],
            images[:, 2],
        )

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        if step % REPORT_EVERY == 0 or step == 1:
            current_loss = evaluate(step)
            print(
                f"Step {step:3d}/{STEPS} | "
                f"BCE: {current_loss:.6f}"
            )
            model.train()

    final_loss = evaluate(STEPS + 1)

    # Save only to new, test-specific paths.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": SEED,
            "study_count": STUDY_COUNT,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "initial_bce": initial_loss,
            "final_bce": final_loss,
        },
        CHECKPOINT_PATH,
    )

    pd.DataFrame(history).to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Initial BCE: {initial_loss:.6f}")
    print(f"Final BCE:   {final_loss:.6f}")
    print(f"Loss reduction: {initial_loss - final_loss:.6f}")
    print(f"Results:   {RESULTS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if final_loss < initial_loss * 0.5:
        print(
            "\nInterpretation: The model substantially reduced loss "
            "on the four fixed studies. This supports the ability to "
            "fit a small subset, though it does not establish "
            "generalization."
        )
    else:
        print(
            "\nInterpretation: The model did not reduce loss by 50%. "
            "We should inspect optimization, architecture, and "
            "gradient behavior before drawing a conclusion."
        )


if __name__ == "__main__":
    main()
