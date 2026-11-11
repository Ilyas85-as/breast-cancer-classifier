from classifier import build_models, compare_models, load_data


def test_data_shape():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}


def test_all_models_beat_baseline():
    X, y = load_data()
    baseline = y.value_counts(normalize=True).max()  # always guess the majority class
    results = compare_models(X, y, cv=3)
    assert len(results) == len(build_models())
    assert (results["mean_accuracy"] > baseline).all()
