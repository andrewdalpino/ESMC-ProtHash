import unittest

import torch

from metrics import CosineSimilarity, LinearCKA, Top1MacroF1


class TestCosineSimilarity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_identical_inputs(self):
        y_student = torch.randn(100, 32)
        mask = torch.ones(100, dtype=torch.long)
        cs = CosineSimilarity()
        cs.update(y_student, y_student, mask)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_scaled_inputs(self):
        y_student = torch.randn(100, 16)
        y_teacher = y_student * 3.0
        mask = torch.ones(100, dtype=torch.long)
        cs = CosineSimilarity()
        cs.update(y_student, y_teacher, mask)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_negative_correlation(self):
        y_student = torch.randn(100, 16)
        y_teacher = -y_student
        mask = torch.ones(100, dtype=torch.long)
        cs = CosineSimilarity()
        cs.update(y_student, y_teacher, mask)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(-1.0), atol=1e-5))

    def test_accumulator_matches_batch(self):
        full = torch.randn(100, 32)
        y_student = full.clone()
        y_teacher = full @ torch.linalg.qr(torch.randn(32, 32))[0]
        mask = torch.ones(100, dtype=torch.long)
        cs_batch = CosineSimilarity()
        cs_batch.update(y_student, y_teacher, mask)
        batch_result = cs_batch.compute()
        cs_split = CosineSimilarity()
        cs_split.update(y_student[:50], y_teacher[:50], mask[:50])
        cs_split.update(y_student[50:], y_teacher[50:], mask[50:])
        split_result = cs_split.compute()
        self.assertTrue(torch.allclose(batch_result, split_result, atol=1e-5))

    def test_3d_inputs_flattened(self):
        y_student = torch.randn(10, 8, 16)
        y_teacher = y_student.clone()
        mask = torch.ones(10, 8, dtype=torch.long)
        cs = CosineSimilarity()
        cs.update(y_student, y_teacher, mask)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_4d_inputs_flattened(self):
        y_student = torch.randn(10, 4, 8, 16)
        y_teacher = y_student.clone()
        mask = torch.ones(10, 4, 8, dtype=torch.long)
        cs = CosineSimilarity()
        cs.update(y_student, y_teacher, mask)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_mismatched_samples_raises_error(self):
        cs = CosineSimilarity()
        with self.assertRaises(AssertionError):
            cs.update(torch.randn(10, 5), torch.randn(8, 5), torch.ones(10))

    def test_mismatched_dimensions_raises_error(self):
        cs = CosineSimilarity()
        with self.assertRaises(AssertionError):
            cs.update(torch.randn(10, 5), torch.randn(10, 8), torch.ones(10))

    def test_reset_then_update(self):
        y_student = torch.randn(50, 16)
        mask = torch.ones(50, dtype=torch.long)
        cs = CosineSimilarity()
        cs.update(torch.randn(50, 16), torch.randn(50, 16), mask)
        cs.reset()
        cs.update(y_student, y_student, mask)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_single_sample(self):
        y_student = torch.randn(1, 8)
        mask = torch.ones(1, dtype=torch.long)
        cs = CosineSimilarity()
        cs.update(y_student, y_student, mask)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_compute_without_update_raises_error(self):
        cs = CosineSimilarity()
        with self.assertRaises(AssertionError):
            cs.compute()


class TestLinearCKA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_identical_inputs(self):
        y_student = torch.randn(100, 32)
        mask = torch.ones(100)
        cka = LinearCKA()
        cka.update(y_student, y_student, mask)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_orthogonal_transformation(self):
        y_student = torch.randn(100, 16)
        q, _ = torch.linalg.qr(torch.randn(16, 16))
        y_teacher = y_student @ q
        mask = torch.ones(100)
        cka = LinearCKA()
        cka.update(y_student, y_teacher, mask)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_independent_random_spaces(self):
        y_student = torch.randn(200, 64)
        y_teacher = torch.randn(200, 64)
        mask = torch.ones(200)
        cka = LinearCKA()
        cka.update(y_student, y_teacher, mask)
        result = cka.compute()
        self.assertLess(result.item(), 0.5)

    def test_scaled_inputs(self):
        y_student = torch.randn(100, 16)
        y_teacher = y_student * 3.0
        mask = torch.ones(100)
        cka = LinearCKA()
        cka.update(y_student, y_teacher, mask)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_accumulator_pattern(self):
        cka = LinearCKA()
        for _ in range(4):
            y_student = torch.randn(25, 16)
            y_teacher = y_student.clone()
            mask = torch.ones(25)
            cka.update(y_student, y_teacher, mask)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_mismatched_samples_raises_error(self):
        cka = LinearCKA()
        with self.assertRaises(AssertionError):
            cka.update(torch.randn(10, 5), torch.randn(8, 5), torch.ones(10))

    def test_reset_then_update(self):
        y_student = torch.randn(50, 16)
        mask = torch.ones(50)
        cka = LinearCKA()
        cka.update(y_student, y_student, mask)
        cka.reset()
        cka.update(y_student, y_student, mask)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_single_sample(self):
        y_student = torch.randn(1, 8)
        mask = torch.ones(1)
        cka = LinearCKA()
        cka.update(y_student, y_student, mask)
        result = cka.compute()
        self.assertTrue(torch.isnan(result))

    def test_accumulator_matches_batch(self):
        full = torch.randn(100, 16)
        y_student = full.clone()
        y_teacher = full @ torch.linalg.qr(torch.randn(16, 16))[0]
        mask = torch.ones(100)
        cka_batch = LinearCKA()
        cka_batch.update(y_student, y_teacher, mask)
        batch_result = cka_batch.compute()
        cka_split = LinearCKA()
        cka_split.update(y_student[:50], y_teacher[:50], mask[:50])
        cka_split.update(y_student[50:], y_teacher[50:], mask[50:])
        split_result = cka_split.compute()
        self.assertTrue(torch.allclose(batch_result, split_result, atol=1e-5))

    def test_compute_without_update_raises_error(self):
        cka = LinearCKA()
        with self.assertRaises(AssertionError):
            cka.compute()


class TestTop1MacroF1(unittest.TestCase):
    NUM_CLASSES = 33
    BATCH_SIZE = 4
    SEQ_LEN = 16

    def setUp(self):
        torch.manual_seed(42)

    def test_perfect_agreement(self):
        logits = torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.NUM_CLASSES)
        mask = torch.ones(self.BATCH_SIZE, self.SEQ_LEN, dtype=torch.bool)
        f1 = Top1MacroF1()
        f1.update(logits, logits.clone(), mask)
        result, _, _ = f1.compute()
        self.assertAlmostEqual(result, 1.0)

    def test_complete_disagreement(self):
        teacher = torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.NUM_CLASSES)
        student = torch.roll(teacher, 1, dims=-1)
        mask = torch.ones(self.BATCH_SIZE, self.SEQ_LEN, dtype=torch.bool)
        f1 = Top1MacroF1()
        f1.update(student, teacher, mask)
        result, _, _ = f1.compute()
        self.assertLess(result, 0.1)

    def test_partial_agreement(self):
        teacher = torch.zeros(self.BATCH_SIZE, self.SEQ_LEN, self.NUM_CLASSES)
        student = torch.zeros(self.BATCH_SIZE, self.SEQ_LEN, self.NUM_CLASSES)
        teacher[:, :, 0] = 1.0
        student[:, : self.SEQ_LEN // 2, 0] = 1.0
        student[:, self.SEQ_LEN // 2 :, 1] = 1.0
        mask = torch.ones(self.BATCH_SIZE, self.SEQ_LEN, dtype=torch.bool)
        f1 = Top1MacroF1()
        f1.update(student, teacher, mask)
        result, _, _ = f1.compute()
        expected = 1.0 / 3.0
        self.assertAlmostEqual(result, expected, places=5)

    def test_respects_mask(self):
        teacher = torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.NUM_CLASSES)
        student = teacher.clone()
        mask = torch.zeros(self.BATCH_SIZE, self.SEQ_LEN, dtype=torch.bool)
        mask[:, 0] = True
        f1 = Top1MacroF1()
        f1.update(student, teacher, mask)
        result, _, _ = f1.compute()
        self.assertAlmostEqual(result, 1.0)

    def test_accumulator_matches_batch(self):
        teacher = torch.randn(self.BATCH_SIZE * 2, self.SEQ_LEN, self.NUM_CLASSES)
        student = teacher.clone()
        mask = torch.ones(self.BATCH_SIZE * 2, self.SEQ_LEN, dtype=torch.bool)
        f1_batch = Top1MacroF1()
        f1_batch.update(student, teacher, mask)
        batch_result, _, _ = f1_batch.compute()
        f1_split = Top1MacroF1()
        f1_split.update(
            student[: self.BATCH_SIZE],
            teacher[: self.BATCH_SIZE],
            mask[: self.BATCH_SIZE],
        )
        f1_split.update(
            student[self.BATCH_SIZE :],
            teacher[self.BATCH_SIZE :],
            mask[self.BATCH_SIZE :],
        )
        split_result, _, _ = f1_split.compute()
        self.assertAlmostEqual(batch_result, split_result, places=5)

    def test_reset(self):
        teacher = torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.NUM_CLASSES)
        student = teacher.clone()
        mask = torch.ones(self.BATCH_SIZE, self.SEQ_LEN, dtype=torch.bool)
        f1 = Top1MacroF1()
        f1.update(student, teacher, mask)
        f1.reset()
        f1.update(student, teacher, mask)
        result, _, _ = f1.compute()
        self.assertAlmostEqual(result, 1.0)

    def test_compute_without_update_raises_error(self):
        f1 = Top1MacroF1()
        with self.assertRaises(AssertionError):
            f1.compute()


if __name__ == "__main__":
    unittest.main()
