import torch

from torch import Tensor

from torch.nn import Module, Buffer


class MaskedMSELoss(Module):
    """MSE loss computed only over non-padding positions."""

    def forward(self, y_student: Tensor, y_teacher: Tensor, mask: Tensor) -> Tensor:
        y_student = y_student.flatten(0, -2).float()
        y_teacher = y_teacher.flatten(0, -2).float()

        mask = mask.flatten().bool()

        y_student = y_student[mask]
        y_teacher = y_teacher[mask]

        if y_student.size(0) == 0:
            return torch.tensor(0.0, device=y_student.device)

        loss = (y_student - y_teacher).pow(2).mean()

        return loss


class DecomposedNormalizedMSE(Module):
    """MSE loss decomposed into independent direction and magnitude components."""

    def forward(self, y_student: Tensor, y_teacher: Tensor, mask: Tensor) -> Tensor:
        y_student = y_student.float()
        y_teacher = y_teacher.float()

        student_norm = y_student.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        teacher_norm = y_teacher.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        direction_loss = (y_student / student_norm - y_teacher / teacher_norm).pow(2)
        magnitude_loss = ((student_norm - teacher_norm) / (teacher_norm + 1e-8)).pow(2)

        mask = mask.unsqueeze(-1)

        direction_loss = direction_loss.masked_select(mask).mean()
        magnitude_loss = magnitude_loss.masked_select(mask).mean()

        loss = direction_loss + magnitude_loss

        return loss


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
