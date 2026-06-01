import torch

from torch import Tensor

from torch.nn import Module, Parameter


class AdaptiveStageWeighting(Module):
    """
    Adaptively weighting the loss of each stage of the embeddings.
    """

    def __init__(self, num_losses: int, min_weight: float):
        super().__init__()

        assert num_losses > 0, "Number of losses must be positive"

        self.log_sigmas = Parameter(torch.zeros(num_losses))

        self.num_losses = num_losses
        self.min_weight = min_weight

    @property
    def loss_weights(self) -> Tensor:
        """
        Get current loss weights based on learned uncertainties.

        Returns:
            Tensor of loss weights for each task.
        """

        weights = torch.exp(-2.0 * self.log_sigmas)

        weights = weights.clamp(min=self.min_weight)

        return weights

    def forward(self, losses: Tensor) -> Tensor:
        """
        Compute task uncertainty-weighted combined loss.

        Args:
            losses: Tensor of individual loss values for each task.

        Returns:
            Combined task uncertainty-weighted loss.
        """

        assert (
            losses.size(0) == self.num_losses
        ), "Number of losses must match number of tasks."

        weighted_losses = 0.5 * self.loss_weights * losses

        regularized_losses = weighted_losses + self.log_sigmas

        combined_loss = regularized_losses.sum()

        return combined_loss
