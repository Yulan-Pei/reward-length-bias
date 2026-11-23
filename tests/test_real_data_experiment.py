import numpy as np

from real_data_experiment import (
    accuracy,
    balanced_indices,
    metrics,
    response_only,
    train_pairwise,
    vectorize,
)


def test_response_only_returns_last_assistant_answer():
    transcript = "Human: first\n\nAssistant: old\n\nHuman: next\n\nAssistant: final answer"
    assert response_only(transcript) == "final answer"


def test_vectorize_is_normalized_and_ignores_unknown_tokens():
    vocabulary = {"helpful": 0, "answer": 1}
    idf = np.array([2.0, 1.0], dtype=np.float32)
    vector = vectorize("helpful helpful answer unknown", vocabulary, idf)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert vector[0] > vector[1] > 0
    assert np.all(vectorize("unknown", vocabulary, idf) == 0)


def test_balanced_indices_equalizes_length_directions_and_keeps_ties():
    raw_length = np.array([5, 4, 3, -1, -2, 0], dtype=np.float32)
    selected = balanced_indices(raw_length, seed=7)
    selected_lengths = raw_length[selected]
    assert (selected_lengths > 0).sum() == (selected_lengths < 0).sum() == 2
    assert (selected_lengths == 0).sum() == 1


def test_accuracy_and_slice_metrics_on_known_margins():
    margins = np.array([2.0, -1.0, 0.0, 3.0])
    lengths = np.array([4.0, 5.0, -2.0, -3.0])
    assert accuracy(margins) == 0.625
    result = metrics(margins, lengths)
    assert result["n"] == 4
    assert result["chosen_long_n"] == 2
    assert result["rejected_long_n"] == 2


def test_pairwise_training_learns_a_separable_direction():
    features = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]], dtype=np.float32)
    weights, losses = train_pairwise(features, seed=3, epochs=10, learning_rate=0.1)
    assert losses[-1] < losses[0]
    assert np.all(features @ weights > 0)
