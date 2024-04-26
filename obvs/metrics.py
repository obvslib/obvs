from __future__ import annotations

import torch
from torchmetrics import Metric


class PrecisionAtKMetric(Metric):
    """
    Compute Precision@k metric for a batch of estimated probabilities vs true token indices.
    The update method takes in the top-k predicted token indices and the true token indices for each example in the batch.
    The compute method returns the Precision@k metric result for the accumulated batches, where a correct prediction
    is considered if the true token is present anywhere in the top-k predictions.
    """

    def __init__(self, topk=10, dist_sync_on_step=False, batch_size=None) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.topk = topk
        self.add_state("correct", default=torch.zeros(batch_size), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    # pylint: disable=arguments-differ
    def update(self, logits, true_token_index) -> None:
        batch_size = logits.shape[0]
        self.correct[:batch_size] += self.batch(logits, true_token_index, self.topk)
        self.total += batch_size

    @staticmethod
    def batch(logits, true_token_index, topk) -> torch.Tensor:
        if not torch.is_tensor(true_token_index) and len(logits.shape) == 2:
            true_token_index = torch.tensor(
                [true_token_index],
                dtype=torch.long,
                device=logits.device,
            ).repeat(logits.size(0))
        elif (
            torch.is_tensor(true_token_index)
            and true_token_index.dim()  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
            == 0
            and len(logits.shape) == 2
        ):
            true_token_index = true_token_index.repeat(logits.size(0))
        elif len(logits.shape) == 1:
            true_token_index = torch.tensor(
                [true_token_index],
                dtype=torch.long,
                device=logits.device,
            )

        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)

        true_token_index = true_token_index.unsqueeze(1)

        topk_pred_indices = torch.topk(logits, k=topk, dim=1).indices

        return (topk_pred_indices == true_token_index).any(dim=1)

    def compute(self) -> torch.Tensor:
        return self.correct.float().sum() / self.total.float()


class SurprisalMetric(Metric):
    """
    Compute Surprisal metric for a batch of estimated probabilities vs true token indices.
    The update method takes in the estimated probabilities and the true token indices for each example in the batch.
    The compute method returns the average Surprisal metric result for the accumulated batches.
    """

    def __init__(self, dist_sync_on_step=False, batch_size=None) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "surprisal",
            default=torch.zeros(batch_size, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    # pylint: disable=arguments-differ
    def update(self, logits, true_token_index) -> None:
        batch_size = logits.shape[0]
        self.surprisal[:batch_size] += self.batch(logits, true_token_index)
        self.total += batch_size

    @staticmethod
    def batch(logits, true_token_index) -> torch.Tensor:
        if not torch.is_tensor(true_token_index) and len(logits.shape) == 2:
            true_token_index = torch.tensor(
                [true_token_index],
                dtype=torch.long,
                device=logits.device,
            ).repeat(logits.size(0))
        elif (
            torch.is_tensor(true_token_index)
            and true_token_index.dim()  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
            == 0
            and len(logits.shape) == 2
        ):
            true_token_index = true_token_index.repeat(logits.size(0))
        elif len(logits.shape) == 1:
            true_token_index = torch.tensor(
                [true_token_index],
                dtype=torch.long,
                device=logits.device,
            )

        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = torch.clamp(probabilities, min=1e-12)
        batch_indices = torch.arange(probabilities.size(0), device=probabilities.device)
        true_probs = probabilities[batch_indices, true_token_index]
        return -torch.log(true_probs)

    def compute(self) -> torch.Tensor:
        return self.surprisal.float().sum() / self.total.float()
