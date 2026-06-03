import torch

from torch import Tensor

from torch.nn import Module, Buffer


class DecomposedNormalizedMSE(Module):
    """
    MSE loss decomposed into independent direction and magnitude components and normalized.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()

        assert epsilon > 0, "Epsilon must be a positive value."

        self.epsilon = epsilon

    def forward(
        self, y_student: Tensor, y_teacher: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        student_norm = y_student.norm(dim=-1, keepdim=True)
        teacher_norm = y_teacher.norm(dim=-1, keepdim=True)

        student_norm = student_norm.clamp(min=self.epsilon)
        teacher_norm = teacher_norm.clamp(min=self.epsilon)

        y_student_normalized = y_student / student_norm
        y_teacher_normalized = y_teacher / teacher_norm

        direction_loss = y_student_normalized - y_teacher_normalized
        magnitude_loss = (student_norm - teacher_norm) / teacher_norm

        direction_loss = direction_loss.pow(2)
        magnitude_loss = magnitude_loss.pow(2)

        mask = mask.unsqueeze(-1)

        direction_loss = direction_loss.masked_select(mask).mean()
        magnitude_loss = magnitude_loss.masked_select(mask).mean()

        return direction_loss, magnitude_loss


class WeightedMultistageLoss(Module):
    """A multistage loss weighting where each stage contributes based on a static scalar."""

    def __init__(self, weights: list[float]):
        super().__init__()

        num_losses = len(weights)

        assert num_losses > 0, "Number of losses must be positive."

        self.weights = Buffer(torch.tensor(weights, dtype=torch.float32))

        self.num_losses = num_losses

    def forward(self, losses: Tensor) -> Tensor:
        assert (
            losses.size(0) == self.num_losses
        ), "Number of losses must match number of tasks."

        weighted_losses = self.weights * losses

        combined_loss = weighted_losses.sum()

        return combined_loss
