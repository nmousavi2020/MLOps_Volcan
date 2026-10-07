from src.config import (
    N_SPLITS,
    EPS,
    EXTREME_THRESHOLD,
    QUANTILES,
    RANDOM_STATE,
    TRAIN_FILE,
    TEST_FILE,
)


def test_config():
    assert N_SPLITS >= 2
    assert EPS > 0
    assert EXTREME_THRESHOLD > 0
    assert len(QUANTILES) > 0
    assert RANDOM_STATE >= 0
    assert TRAIN_FILE == "global.csv"
    assert TEST_FILE == "gris_features.csv"