from ish_knee.model.study_split import StudySplit

def main() -> None:

    print("=" * 70)
    print("STUDT SPLIT TEST")
    print("=" * 70)

    studies = [f"STUDY-{index:03d}" for index in range(58)]

    training, validation = StudySplit.split(
        studies,
        validation_fraction = 0.20,
        seed = 42
    )

    # ---------------------------------------------------------------
    # Check the expected split sizes
    # ---------------------------------------------------------------

    assert len(training) == 46
    assert len(validation) == 12

    # ---------------------------------------------------------------
    # Check that every study is accounted for exactly once
    # ---------------------------------------------------------------

    assert len(training) + len(validation) == len(studies)
    assert set(training).isdisjoint(set(validation))
    assert set(training) | set(validation) == set(studies)

    # ---------------------------------------------------------------
    # Check that the same seed produces the same split
    # ---------------------------------------------------------------

    training_again, validation_again = StudySplit.split( 
        studies,
        validation_fraction = 0.20,
        seed = 42
        )

    assert training == training_again
    assert validation == validation_again

    print("Total studies:", len(studies))
    print("Training studies:", len(training))
    print("Validation studies:", len(validation))
    print("No overlap between groups: True")
    print("Reproducible split: True")
    print()
    print("STUDY SPLIT TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    main()