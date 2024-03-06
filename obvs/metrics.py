from torchmetrics import Metric
import torch


class PrecisionAtKMetric(Metric):
    """
    Compute Precision@k metric for a batch of estimated probabilities vs true token indices.
    The update method takes in the top-k predicted token indices and the true token indices for each example in the batch.
    The compute method returns the Precision@k metric result for the accumulated batches, where a correct prediction
    is considered if the true token is present anywhere in the top-k predictions.
    """

    def __init__(self, topk=10, dist_sync_on_step=False, batch_size=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.topk = topk
        self.add_state("correct", default=torch.zeros(batch_size), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits, true_token_index):
        batch_size = logits.shape[0]
        if not torch.is_tensor(true_token_index):
            true_token_index = torch.tensor(
                [true_token_index], dtype=torch.long, device=logits.device
            ).repeat(logits.size(0))
        elif true_token_index.dim() == 0:
            true_token_index = true_token_index.repeat(logits.size(0))

        true_token_index = true_token_index.unsqueeze(1)

        topk_pred_indices = torch.topk(logits, k=self.topk, dim=1).indices

        matches = (topk_pred_indices == true_token_index).any(dim=1)
        self.correct[:batch_size] += matches
        self.total += logits.shape[0]

    def compute(self):
        return self.correct.float().sum() / self.total.float()


class SurprisalMetric(Metric):
    """
    Compute Surprisal metric for a batch of estimated probabilities vs true token indices.
    The update method takes in the estimated probabilities and the true token indices for each example in the batch.
    The compute method returns the average Surprisal metric result for the accumulated batches.
    """

    def __init__(self, dist_sync_on_step=False, batch_size=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("surprisal", default=torch.zeros(batch_size, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits, true_token_index):
        batch_size = logits.shape[0]
        if not torch.is_tensor(true_token_index):
            true_token_index = torch.tensor(
                [true_token_index], dtype=torch.long, device=logits.device
            ).repeat(logits.size(0))
        elif true_token_index.dim() == 0:
            true_token_index = true_token_index.repeat(logits.size(0))

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = torch.clamp(probabilities, min=1e-12)
        batch_indices = torch.arange(probabilities.size(0), device=probabilities.device)
        true_probs = probabilities[batch_indices, true_token_index]
        surprisal = -torch.log(true_probs)
        self.surprisal[:batch_size] += surprisal
        self.total += surprisal.shape[0]

    def compute(self):
        return self.surprisal.float().sum() / self.total.float()
