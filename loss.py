import torch

from torch import Tensor

from torch.nn import Module, Buffer
from torch.nn.functional import cross_entropy


class DecomposedRepresentationLoss(Module):
    """
    MSE loss decomposed into independent direction and magnitude components and normalized.
    """

    def __init__(self, epsilon: float):
        super().__init__()

        assert epsilon > 0, "Epsilon must be a positive value."

        self.epsilon = epsilon

    def forward(
        self, y_student: Tensor, y_teacher: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        assert (
            y_student.size() == y_teacher.size()
        ), "y_student and y_teacher must have the same dimensionality."

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


class ContrastiveAlignmentLoss(Module):
    """
    Sequence-level contrastive alignment loss. Pools student and teacher representations
    over the sequence dimension, then computes a symmetric InfoNCE (NT-Xent) loss where
    corresponding sequences are positive pairs and non-corresponding sequences are negatives.
    """

    def __init__(self, temperature: float, epsilon: float):
        super().__init__()

        assert temperature > 0, "Temperature must be a positive value."
        assert epsilon > 0, "Epsilon must be a positive value."

        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, y_student: Tensor, y_teacher: Tensor, mask: Tensor) -> Tensor:
        assert (
            y_student.size() == y_teacher.size()
        ), "y_student and y_teacher must have the same dimensionality."

        mask = mask.unsqueeze(-1)

        student_pooled = (y_student * mask).sum(dim=1)
        teacher_pooled = (y_teacher * mask).sum(dim=1)

        sequence_lengths = mask.sum(dim=1).clamp(min=self.epsilon)

        student_pooled = student_pooled / sequence_lengths
        teacher_pooled = teacher_pooled / sequence_lengths

        student_norm = student_pooled.norm(dim=-1, keepdim=True)
        teacher_norm = teacher_pooled.norm(dim=-1, keepdim=True)

        student_norm = student_norm.clamp(min=self.epsilon)
        teacher_norm = teacher_norm.clamp(min=self.epsilon)

        student_pooled_normalized = student_pooled / student_norm
        teacher_pooled_normalized = teacher_pooled / teacher_norm

        logits = student_pooled_normalized @ teacher_pooled_normalized.T

        logits /= self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)

        student_to_teacher_loss = cross_entropy(logits, labels)
        teacher_to_student_loss = cross_entropy(logits.T, labels)

        loss = (student_to_teacher_loss + teacher_to_student_loss) / 2

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
