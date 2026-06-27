import torch

from torch import Tensor

from torch.nn.functional import cosine_similarity as torch_cosine_similarity


class CosineSimilarity:
    """
    Compute the average cosine similarity between two sets of features.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_similarity = 0.0
        self.num_samples = 0

    def update(self, y_student: Tensor, y_teacher: Tensor, mask: Tensor):
        assert (
            y_student.size() == y_teacher.size()
        ), f"Student ({y_student.size()}) and Teacher ({y_teacher.size()}) must have the same dimensions."

        mask = mask.bool()

        y_student = y_student[mask]
        y_teacher = y_teacher[mask]

        similarity = torch_cosine_similarity(y_student, y_teacher, dim=-1)

        self.total_similarity += similarity.sum()
        self.num_samples += similarity.numel()

    def compute(self) -> Tensor:
        assert self.num_samples > 0, "No updates have been made yet."
        assert isinstance(self.total_similarity, Tensor)

        score = self.total_similarity / self.num_samples

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

    def update(self, y_student: Tensor, y_teacher: Tensor, mask: Tensor):
        assert (
            y_student.size() == y_teacher.size()
        ), f"Student ({y_student.size()}) and Teacher ({y_teacher.size()}) must have the same dimensions."

        y_student = y_student.flatten(0, -2).float()
        y_teacher = y_teacher.flatten(0, -2).float()

        mask = mask.flatten().bool()

        y_student = y_student[mask]
        y_teacher = y_teacher[mask]

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

        cross_sigma = self.num_samples * student_mean[:, None] * teacher_mean[None, :]
        student_sigma = self.num_samples * student_mean[:, None] * student_mean[None, :]
        teacher_sigma = self.num_samples * teacher_mean[:, None] * teacher_mean[None, :]

        centered_cross = self.student_teacher_cross - cross_sigma
        centered_student_gram = self.student_gram - student_sigma
        centered_teacher_gram = self.teacher_gram - teacher_sigma

        centered_cross_ss = centered_cross.square().sum()

        centered_student_gram_ss = centered_student_gram.square().sum()
        centered_teacher_gram_ss = centered_teacher_gram.square().sum()

        denominator = centered_student_gram_ss.sqrt() * centered_teacher_gram_ss.sqrt()

        score = centered_cross_ss / denominator

        return score


class Top1MacroF1:
    """
    Compute the macro-averaged F1 score between student and teacher token predictions.
    """

    def __init__(self):
        self.confusion_matrix = None

    def reset(self):
        self.confusion_matrix = None

    def update(self, y_student: Tensor, y_teacher: Tensor, mask: Tensor):
        assert (
            y_student.size() == y_teacher.size()
        ), f"Student ({y_student.size()}) and Teacher ({y_teacher.size()}) must have the same dimensions."

        n_classes = y_student.size(-1)

        teacher_labels = y_teacher.argmax(dim=-1)
        student_labels = y_student.argmax(dim=-1)

        mask = mask.bool()

        teacher_labels = teacher_labels[mask]
        student_labels = student_labels[mask]

        indices = student_labels * n_classes + teacher_labels

        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros(
                n_classes,
                n_classes,
                dtype=torch.int64,
                device=indices.device,
            )

        counts = torch.bincount(indices, minlength=n_classes**2)

        self.confusion_matrix += counts.view(n_classes, n_classes)

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        assert self.confusion_matrix is not None, "No updates have been made yet."

        tp = self.confusion_matrix.diag().float()

        sigma_p = self.confusion_matrix.sum(dim=1)
        sigma_r = self.confusion_matrix.sum(dim=0)

        # Select indices of classes that have no predictions or true labels.
        present = (sigma_r > 0) | (sigma_p > 0)

        fp = sigma_p - tp
        fn = sigma_r - tp

        tp_fp = tp + fp
        tp_fn = tp + fn

        precision = torch.where(tp_fp > 0, tp / tp_fp, torch.zeros_like(tp))
        recall = torch.where(tp_fn > 0, tp / tp_fn, torch.zeros_like(tp))

        f1 = torch.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            torch.zeros_like(precision),
        )

        # Exclude classes that have no predictions or true labels.
        precision = precision[present]
        recall = recall[present]
        f1 = f1[present]

        precision = precision.mean() if precision.numel() > 0 else torch.tensor(0.0)
        recall = recall.mean() if recall.numel() > 0 else torch.tensor(0.0)
        f1 = f1.mean() if f1.numel() > 0 else torch.tensor(0.0)

        return f1, precision, recall
