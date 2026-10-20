from pathlib import Path

from ish_knee.model.training_config import TrainingConfig

def main() -> None:

    print("=" * 70)
    print("TRAINING CONFIG TEST")
    print("=" * 70)

    config = TrainingConfig()

    assert config.batch_size == 1
    assert config.learning_rate == 0.001
    assert config.epochs == 10
    assert config.num_workers == 0
    assert config.checkpoint_dir == Path("checkpoints")

    print("default configuration: ", config)

    try:
        TrainingConfig(batch_size=0)
        raise AssertionError("Invalid batch size was accepted.")
    except ValueError:
        print("Invalid batch size correctly rejected")

    try:
        TrainingConfig(learning_rate=0)
        raise AssertionError("Invalid learning rate was accepted.")
    except ValueError:
        print("Invalid learning rate correctly rejected")

    print()
    print("TRAINING CONFIG TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()


