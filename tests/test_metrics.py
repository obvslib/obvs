from __future__ import annotations

import pytest
import torch

from obvs.metrics import PrecisionAtKMetric, SurprisalMetric


def test_precision_at_k_metric_update_correct_prediction():
    # Create fake logits
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    # The true token index for each example in the batch
    true_token_index = torch.tensor([1, 0])

    metric = PrecisionAtKMetric(topk=1, batch_size=2)
    metric.update(logits, true_token_index)
    assert metric.correct.int().tolist() == [1, 1]
    assert metric.total == 2


def test_precision_at_k_metric_compute_correct_prediction():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    true_token_index = torch.tensor([1, 0])

    metric = PrecisionAtKMetric(topk=1, batch_size=2)
    metric.update(logits, true_token_index)
    assert metric.compute() == 1


def test_precision_at_k_metric_no_correct_prediction():
    logits = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
    true_token_index = torch.tensor([1, 1])

    metric = PrecisionAtKMetric(topk=1, batch_size=2)
    metric.update(logits, true_token_index)
    assert metric.correct.int().tolist() == [0, 0]
    assert metric.total == 2
    assert metric.compute() == 0.0


def test_precision_at_k_metric_topk_2():
    logits = torch.tensor([[0.1, 0.9, 0.05], [0.8, 0.2, 0.05]])
    true_token_index = torch.tensor([0, 1])

    metric = PrecisionAtKMetric(topk=2, batch_size=2)
    metric.update(logits, true_token_index)
    assert metric.correct.float().tolist() == [1, 1]
    assert metric.total == 2
    assert metric.compute() == 1.0


def test_surprisal_metric_update():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    true_token_index = torch.tensor([0, 1])
    # To calculate expected surprisal, we need to calculate the negative
    # log of the probability of the true token
    expected = -torch.log(torch.softmax(logits, dim=-1)[torch.arange(2), true_token_index])

    metric = SurprisalMetric(batch_size=2)
    metric.update(logits, true_token_index)
    assert metric.surprisal.float().tolist() == expected.tolist()
    assert metric.total == 2


def test_surprisal_metric_compute():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    true_token_index = torch.tensor([0, 1])
    # To calculate expected surprisal, we need to calculate the negative
    # log of the probability of the true token
    expected = -torch.log(torch.softmax(logits, dim=-1)[torch.arange(2), true_token_index])

    metric = SurprisalMetric(batch_size=2)
    metric.update(logits, true_token_index)
    assert metric.compute() == expected.mean()


def test_precision_at_k_metric_single_true_token():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    true_token_index = 0

    metric = PrecisionAtKMetric(topk=1, batch_size=2)
    metric.update(logits, true_token_index)
    assert metric.correct.float().tolist() == [0, 1]
    assert metric.total == 2
    assert metric.compute() == 0.5


def test_surprisal_metric_single_true_token():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    true_token_index = 1
    expected = -torch.log(torch.softmax(logits, dim=-1)[torch.arange(2), true_token_index])

    metric = SurprisalMetric(batch_size=2)
    metric.update(logits, true_token_index)
    assert metric.surprisal.float().tolist() == pytest.approx(expected.tolist())
    assert metric.total == 2
    assert metric.compute() == expected.mean()
