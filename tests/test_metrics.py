import unittest

import torch

from metrics import CosineSimilarity, LinearCKA


class TestCosineSimilarity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_identical_inputs(self):
        y_student = torch.randn(100, 32)
        cs = CosineSimilarity()
        cs.update(y_student, y_student)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_scaled_inputs(self):
        y_student = torch.randn(100, 16)
        y_teacher = y_student * 3.0
        cs = CosineSimilarity()
        cs.update(y_student, y_teacher)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_negative_correlation(self):
        y_student = torch.randn(100, 16)
        y_teacher = -y_student
        cs = CosineSimilarity()
        cs.update(y_student, y_teacher)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(-1.0), atol=1e-5))

    def test_accumulator_matches_batch(self):
        full = torch.randn(100, 32)
        y_student = full.clone()
        y_teacher = full @ torch.linalg.qr(torch.randn(32, 32))[0]
        cs_batch = CosineSimilarity()
        cs_batch.update(y_student, y_teacher)
        batch_result = cs_batch.compute()
        cs_split = CosineSimilarity()
        cs_split.update(y_student[:50], y_teacher[:50])
        cs_split.update(y_student[50:], y_teacher[50:])
        split_result = cs_split.compute()
        self.assertTrue(torch.allclose(batch_result, split_result, atol=1e-5))

    def test_3d_inputs_flattened(self):
        y_student = torch.randn(10, 8, 16)
        y_teacher = y_student.clone()
        cs = CosineSimilarity()
        cs.update(y_student, y_teacher)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_4d_inputs_flattened(self):
        y_student = torch.randn(10, 4, 8, 16)
        y_teacher = y_student.clone()
        cs = CosineSimilarity()
        cs.update(y_student, y_teacher)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_mismatched_samples_raises_error(self):
        cs = CosineSimilarity()
        with self.assertRaises(AssertionError):
            cs.update(torch.randn(10, 5), torch.randn(8, 5))

    def test_mismatched_dimensions_raises_error(self):
        cs = CosineSimilarity()
        with self.assertRaises(AssertionError):
            cs.update(torch.randn(10, 5), torch.randn(10, 8))

    def test_reset_then_update(self):
        y_student = torch.randn(50, 16)
        cs = CosineSimilarity()
        cs.update(torch.randn(50, 16), torch.randn(50, 16))
        cs.reset()
        cs.update(y_student, y_student)
        result = cs.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_single_sample(self):
        y_student = torch.randn(1, 8)
        cs = CosineSimilarity()
        cs.update(y_student, y_student)
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
        cka = LinearCKA()
        cka.update(y_student, y_student)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_orthogonal_transformation(self):
        y_student = torch.randn(100, 16)
        q, _ = torch.linalg.qr(torch.randn(16, 16))
        y_teacher = y_student @ q
        cka = LinearCKA()
        cka.update(y_student, y_teacher)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_independent_random_spaces(self):
        y_student = torch.randn(200, 64)
        y_teacher = torch.randn(200, 64)
        cka = LinearCKA()
        cka.update(y_student, y_teacher)
        result = cka.compute()
        self.assertLess(result.item(), 0.5)

    def test_scaled_inputs(self):
        y_student = torch.randn(100, 16)
        y_teacher = y_student * 3.0
        cka = LinearCKA()
        cka.update(y_student, y_teacher)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_accumulator_pattern(self):
        cka = LinearCKA()
        for _ in range(4):
            y_student = torch.randn(25, 16)
            y_teacher = y_student.clone()
            cka.update(y_student, y_teacher)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_mismatched_samples_raises_error(self):
        cka = LinearCKA()
        with self.assertRaises(AssertionError):
            cka.update(torch.randn(10, 5), torch.randn(8, 5))

    def test_reset_then_update(self):
        y_student = torch.randn(50, 16)
        cka = LinearCKA()
        cka.update(y_student, y_student)
        cka.reset()
        cka.update(y_student, y_student)
        result = cka.compute()
        self.assertTrue(torch.allclose(result, torch.tensor(1.0), atol=1e-5))

    def test_single_sample(self):
        y_student = torch.randn(1, 8)
        cka = LinearCKA()
        cka.update(y_student, y_student)
        result = cka.compute()
        self.assertTrue(torch.isnan(result))

    def test_accumulator_matches_batch(self):
        full = torch.randn(100, 16)
        y_student = full.clone()
        y_teacher = full @ torch.linalg.qr(torch.randn(16, 16))[0]
        cka_batch = LinearCKA()
        cka_batch.update(y_student, y_teacher)
        batch_result = cka_batch.compute()
        cka_split = LinearCKA()
        cka_split.update(y_student[:50], y_teacher[:50])
        cka_split.update(y_student[50:], y_teacher[50:])
        split_result = cka_split.compute()
        self.assertTrue(torch.allclose(batch_result, split_result, atol=1e-5))

    def test_compute_without_update_raises_error(self):
        cka = LinearCKA()
        with self.assertRaises(AssertionError):
            cka.compute()


if __name__ == "__main__":
    unittest.main()
