import torch

from torch import Tensor

from torch.nn.functional import cosine_similarity as torch_cosine_similarity


class CosineSimilarity:
    """
    Compute the average cosine similarity between two sets of features.
    Matches the evaluation logic in distill.py.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_similarity = 0.0
        self.num_samples = 0

    def update(self, y_student: Tensor, y_teacher: Tensor):
        assert (
            y_student.size() == y_teacher.size()
        ), "y_student and y_teacher must have the same dimensions."

        similarity = torch_cosine_similarity(y_student.flatten(1), y_teacher.flatten(1))

        self.total_similarity += similarity.sum().item()
        self.num_samples += y_student.size(0)

    def compute(self) -> Tensor:
        assert self.num_samples > 0, "No samples have been added."

        score = torch.tensor(self.total_similarity / self.num_samples)

        return score


class LinearCKA:
    """
    Compute the linear Centered Kernel Alignment (CKA) similarity between two sets of features.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.student_teacher_cross = None
        self.student_gram = None
        self.teacher_gram = None
        self.student_sum = None
        self.teacher_sum = None
        self.num_samples = 0

    def update(self, y_student: Tensor, y_teacher: Tensor):
        assert (
            y_student.size() == y_teacher.size()
        ), "y_student and y_teacher must have the same dimensions."

        y_student = y_student.flatten(0, -2).float()
        y_teacher = y_teacher.flatten(0, -2).float()

        if self.student_teacher_cross is None:
            self.student_teacher_cross = y_student.T @ y_teacher

            self.student_gram = y_student.T @ y_student
            self.teacher_gram = y_teacher.T @ y_teacher

            self.student_sum = y_student.sum(dim=0)
            self.teacher_sum = y_teacher.sum(dim=0)

        else:
            assert (
                self.student_teacher_cross is not None
            ), "student_teacher_cross is None."

            assert self.student_gram is not None, "student_gram is None."
            assert self.teacher_gram is not None, "teacher_gram is None."
            assert self.student_sum is not None, "student_sum is None."
            assert self.teacher_sum is not None, "teacher_sum is None."

            self.student_teacher_cross += y_student.T @ y_teacher

            self.student_gram += y_student.T @ y_student
            self.teacher_gram += y_teacher.T @ y_teacher

            self.student_sum += y_student.sum(dim=0)
            self.teacher_sum += y_teacher.sum(dim=0)

        self.num_samples += y_student.size(0)

    def compute(self) -> Tensor:
        assert self.student_teacher_cross is not None, "student_teacher_cross is None."
        assert self.student_gram is not None, "student_gram is None."
        assert self.teacher_gram is not None, "teacher_gram is None."
        assert self.student_sum is not None, "student_sum is None."
        assert self.teacher_sum is not None, "teacher_sum is None."
        assert self.num_samples > 0, "No samples have been added."

        student_mean = self.student_sum / self.num_samples
        teacher_mean = self.teacher_sum / self.num_samples

        centered_cross = (
            self.student_teacher_cross
            - self.num_samples * student_mean[:, None] * teacher_mean[None, :]
        )

        centered_student_gram = (
            self.student_gram
            - self.num_samples * student_mean[:, None] * student_mean[None, :]
        )

        centered_teacher_gram = (
            self.teacher_gram
            - self.num_samples * teacher_mean[:, None] * teacher_mean[None, :]
        )

        centered_cross_squared = centered_cross.square().sum()
        centered_student_gram_squared = centered_student_gram.square().sum()
        centered_teacher_gram_squared = centered_teacher_gram.square().sum()

        denominator = (
            centered_student_gram_squared * centered_teacher_gram_squared
        ).sqrt()

        score = centered_cross_squared / denominator

        return score
