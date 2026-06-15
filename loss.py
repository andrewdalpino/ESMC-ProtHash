import torch

from torch import Tensor

from torch.nn import Module, Buffer
from torch.nn.functional import cross_entropy


class DecomposedRepresentationLoss(Module):
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
        ), "y_student and y_teacher must have the same dimensionality."

        student_norm = y_student.norm(dim=-1, keepdim=True)
        teacher_norm = y_teacher.norm(dim=-1, keepdim=True)

        student_norm = student_norm.clamp(min=self.norm_epsilon)
        teacher_norm = teacher_norm.clamp(min=self.norm_epsilon)

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
    over the sequence dimension, then computes an  asymmetric InfoNCE (NT-Xent) loss where
    corresponding sequences are positive pairs and non-corresponding sequences are negatives.

    Optionally maintains a FIFO queue of past teacher representations to serve as additional
    negatives, compensating for small batch sizes.
    """

    def __init__(
        self,
        temperature: float,
        queue_size: int,
        embedding_dimensions: int,
        norm_epsilon: float,
    ):
        super().__init__()

        assert temperature > 0, "Temperature must be a positive value."
        assert norm_epsilon > 0, "Epsilon must be a positive value."
        assert queue_size > 0, "Queue size must be positive."
        assert embedding_dimensions > 0, "Embedding dimensions must be positive."

        self.queue = Buffer(torch.zeros(queue_size, embedding_dimensions))

        self.temperature = temperature
        self.norm_epsilon = norm_epsilon
        self.queue_size = queue_size
        self.queue_pointer = 0
        self.queue_filled = False

    @torch.no_grad()
    def prefill_queue(self, y_teacher: Tensor, mask: Tensor) -> None:
        mask = mask.unsqueeze(-1)

        teacher_pooled = (y_teacher * mask).sum(dim=1)

        sequence_lengths = mask.sum(dim=1)

        assert (
            sequence_lengths > 0
        ).all(), "Each sequence must have at least one unmasked position."

        teacher_pooled = teacher_pooled / sequence_lengths

        teacher_norm = teacher_pooled.norm(dim=-1, keepdim=True)
        teacher_norm = teacher_norm.clamp(min=self.norm_epsilon)

        teacher_pooled_normalized = teacher_pooled / teacher_norm

        _ = self._update_queue(teacher_pooled_normalized)

    @torch.no_grad()
    def _update_queue(self, teacher_pooled_normalized: Tensor) -> Tensor:
        n = teacher_pooled_normalized.size(0)

        pointer_end = self.queue_pointer + n

        indices = torch.arange(self.queue_pointer, pointer_end) % self.queue_size

        self.queue[indices] = teacher_pooled_normalized.to(self.queue.dtype)

        self.queue_pointer = pointer_end % self.queue_size

        if pointer_end >= self.queue_size:
            self.queue_filled = True

        return indices

    def forward(self, y_student: Tensor, y_teacher: Tensor, mask: Tensor) -> Tensor:
        assert (
            y_student.size() == y_teacher.size()
        ), "y_student and y_teacher must have the same dimensionality."

        mask = mask.unsqueeze(-1)

        student_pooled = (y_student * mask).sum(dim=1)
        teacher_pooled = (y_teacher * mask).sum(dim=1)

        sequence_lengths = mask.sum(dim=1)

        assert (
            sequence_lengths > 0
        ).all(), "Each sequence must have at least one unmasked position."

        student_pooled = student_pooled / sequence_lengths
        teacher_pooled = teacher_pooled / sequence_lengths

        student_norm = student_pooled.norm(dim=-1, keepdim=True)
        teacher_norm = teacher_pooled.norm(dim=-1, keepdim=True)

        student_norm = student_norm.clamp(min=self.norm_epsilon)
        teacher_norm = teacher_norm.clamp(min=self.norm_epsilon)

        student_pooled_normalized = student_pooled / student_norm
        teacher_pooled_normalized = teacher_pooled / teacher_norm

        indices = self._update_queue(teacher_pooled_normalized)

        teacher_pooled_normalized = (
            self.queue if self.queue_filled else self.queue[: self.queue_pointer]
        )

        logits = student_pooled_normalized @ teacher_pooled_normalized.T

        logits /= self.temperature

        labels = indices.to(device=logits.device)

        loss = cross_entropy(logits, labels)

        return loss


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
