import numpy as np
import torch

from nectar_listwise_experiment import centered_correlation, group_metrics, preference_loss


def test_two_candidate_pl_equals_bradley_terry():
    scores = torch.tensor([[2.0, -1.0], [0.2, 0.1]])
    assert torch.allclose(preference_loss(scores, "pairwise_bt"), preference_loss(scores, "top1_pl"))
    assert torch.allclose(preference_loss(scores, "pairwise_bt"), preference_loss(scores, "full_pl"))


def test_all_losses_prefer_correct_ordering():
    correct = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]])
    reversed_scores = torch.flip(correct, dims=[1])
    for kind in ("pairwise_bt", "top1_pl", "full_pl"):
        assert preference_loss(correct, kind) < preference_loss(reversed_scores, kind)


def test_centered_correlation_ignores_group_offsets():
    lengths = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    scores = lengths + torch.tensor([[100.0], [-50.0]])
    assert torch.isclose(centered_correlation(scores, lengths), torch.tensor(1.0))


def test_perfect_ranking_metrics_are_one():
    scores = np.tile(np.arange(7, 0, -1, dtype=np.float32), (3, 1))
    lengths = np.tile(np.array([3, 7, 2, 6, 1, 5, 4], dtype=np.float32), (3, 1))
    metrics, _ = group_metrics(scores, lengths)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["pairwise_accuracy"] == 1.0
    assert np.isclose(metrics["ndcg_at_7"], 1.0)
