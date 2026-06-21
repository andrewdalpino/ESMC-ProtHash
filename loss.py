import torch

from torch import Tensor

from torch.nn import Module, Buffer


class DecomposedTokenRepresentationLoss(Module):
    """
    MSE loss decomposed into independent direction and magnitude components and normalized.
    """

    def __init__(self, norm_epsilon: float):
        super().__init__()

        assert norm_epsilon > 0, "Epsilon must be a positive value."

        self.norm_epsilon = norm_epsilon

    def forward(
        self, y_student: Tensor, y_teacher: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        assert (
            y_student.size() == y_teacher.size()
        ), f"y_student and y_teacher must have the same dimensionality. Got {y_student.size()} and {y_teacher.size()}."

        assert mask.any(
            dim=1
        ).all(), "Each sequence must contain at least one unmasked token."

        embedding_dimensions = y_student.size(2)

        y_student = y_student.view(-1, embedding_dimensions)[mask.view(-1)]
        y_teacher = y_teacher.view(-1, embedding_dimensions)[mask.view(-1)]

        student_norm = y_student.norm(dim=-1, keepdim=True).clamp(min=self.norm_epsilon)
        teacher_norm = y_teacher.norm(dim=-1, keepdim=True).clamp(min=self.norm_epsilon)

        y_student_normalized = y_student / student_norm
        y_teacher_normalized = y_teacher / teacher_norm

        direction_loss = (y_student_normalized - y_teacher_normalized).pow(2)
        magnitude_loss = ((student_norm - teacher_norm) / teacher_norm).pow(2)

        direction_loss = direction_loss.mean()
        magnitude_loss = magnitude_loss.mean()

        return direction_loss, magnitude_loss


class WeightedCombinedLoss(Module):
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
