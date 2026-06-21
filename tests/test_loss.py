import unittest

import torch

from loss import DecomposedTokenRepresentationLoss, WeightedCombinedLoss


class TestDecomposedTokenRepresentationLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_construction(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        self.assertEqual(loss.norm_epsilon, 1e-8)

    def test_zero_epsilon_raises_error(self):
        with self.assertRaises(AssertionError):
            DecomposedTokenRepresentationLoss(norm_epsilon=0.0)

    def test_negative_epsilon_raises_error(self):
        with self.assertRaises(AssertionError):
            DecomposedTokenRepresentationLoss(norm_epsilon=-1.0)

    def test_mismatched_dimensions_raises_error(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y_student = torch.randn(4, 8, 16)
        y_teacher = torch.randn(4, 8, 32)
        mask = torch.ones(4, 8, dtype=torch.bool)
        with self.assertRaises(AssertionError):
            loss.forward(y_student, y_teacher, mask)

    def test_mismatched_batch_size_raises_error(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y_student = torch.randn(4, 8, 16)
        y_teacher = torch.randn(2, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.bool)
        with self.assertRaises(AssertionError):
            loss.forward(y_student, y_teacher, mask)

    def test_empty_mask_raises_error(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y_student = torch.randn(4, 8, 16)
        y_teacher = torch.randn(4, 8, 16)
        mask = torch.zeros(4, 8, dtype=torch.bool)
        with self.assertRaises(AssertionError):
            loss.forward(y_student, y_teacher, mask)

    def test_some_sequences_empty_raises_error(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y_student = torch.randn(4, 8, 16)
        y_teacher = torch.randn(4, 8, 16)
        mask = torch.zeros(4, 8, dtype=torch.bool)
        mask[0, :4] = True
        mask[1, :2] = True
        mask[2, :] = False
        mask[3, :] = False
        with self.assertRaises(AssertionError):
            loss.forward(y_student, y_teacher, mask)

    def test_identical_inputs(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y = torch.randn(4, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.bool)
        direction_loss, magnitude_loss = loss.forward(y, y, mask)
        self.assertAlmostEqual(direction_loss.item(), 0.0, places=6)
        self.assertAlmostEqual(magnitude_loss.item(), 0.0, places=6)

    def test_scaled_inputs(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y_teacher = torch.randn(4, 8, 16)
        y_student = y_teacher * 3.0
        mask = torch.ones(4, 8, dtype=torch.bool)
        direction_loss, magnitude_loss = loss.forward(y_student, y_teacher, mask)
        self.assertAlmostEqual(direction_loss.item(), 0.0, places=6)
        self.assertGreater(magnitude_loss.item(), 0.0)

    def test_direction_difference(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y_teacher = torch.randn(4, 8, 16)
        y_student = y_teacher + 1.0
        mask = torch.ones(4, 8, dtype=torch.bool)
        direction_loss, magnitude_loss = loss.forward(y_student, y_teacher, mask)
        self.assertGreater(direction_loss.item(), 0.0)
        self.assertGreater(magnitude_loss.item(), 0.0)

    def test_same_magnitude_rotated(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y = torch.randn(4, 8, 16)
        norms = y.norm(dim=-1, keepdim=True)
        y_teacher = y / norms
        y_student = y_teacher[
            :, :, [15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        ]
        mask = torch.ones(4, 8, dtype=torch.bool)
        direction_loss, magnitude_loss = loss.forward(y_student, y_teacher, mask)
        self.assertGreater(direction_loss.item(), 0.0)
        self.assertAlmostEqual(magnitude_loss.item(), 0.0, places=6)

    def test_respects_mask(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y_student = torch.randn(4, 8, 16)
        y_teacher = y_student.clone()
        y_teacher[:, 4:, :] = torch.randn(4, 4, 16)
        mask = torch.zeros(4, 8, dtype=torch.bool)
        mask[:, :4] = True
        direction_loss, magnitude_loss = loss.forward(y_student, y_teacher, mask)
        self.assertAlmostEqual(direction_loss.item(), 0.0, places=6)
        self.assertAlmostEqual(magnitude_loss.item(), 0.0, places=6)

    def test_varied_batch_shape(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y = torch.randn(3, 7, 16)
        mask = torch.ones(3, 7, dtype=torch.bool)
        direction_loss, magnitude_loss = loss.forward(y, y, mask)
        self.assertAlmostEqual(direction_loss.item(), 0.0, places=6)
        self.assertAlmostEqual(magnitude_loss.item(), 0.0, places=6)

    def test_epsilon_clamping(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1.0)
        y_student = torch.zeros(4, 8, 16)
        y_teacher = torch.ones(4, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.bool)
        direction_loss, magnitude_loss = loss.forward(y_student, y_teacher, mask)
        self.assertTrue(torch.isfinite(direction_loss).all())
        self.assertTrue(torch.isfinite(magnitude_loss).all())

    def test_single_token(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y = torch.randn(1, 1, 16)
        mask = torch.ones(1, 1, dtype=torch.bool)
        direction_loss, magnitude_loss = loss.forward(y, y, mask)
        self.assertAlmostEqual(direction_loss.item(), 0.0, places=6)
        self.assertAlmostEqual(magnitude_loss.item(), 0.0, places=6)

    def test_outputs_are_finite(self):
        loss = DecomposedTokenRepresentationLoss(norm_epsilon=1e-8)
        y_student = torch.randn(4, 8, 16)
        y_teacher = torch.randn(4, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.bool)
        direction_loss, magnitude_loss = loss.forward(y_student, y_teacher, mask)
        self.assertTrue(torch.isfinite(direction_loss).all())
        self.assertTrue(torch.isfinite(magnitude_loss).all())


class TestWeightedCombinedLoss(unittest.TestCase):
    def test_construction(self):
        loss = WeightedCombinedLoss(weights=[0.5, 1.0, 1.5])
        self.assertEqual(loss.num_losses, 3)
        self.assertTrue(torch.allclose(loss.weights, torch.tensor([0.5, 1.0, 1.5])))

    def test_empty_weights_raises_error(self):
        with self.assertRaises(AssertionError):
            WeightedCombinedLoss(weights=[])

    def test_mismatched_loss_count_raises_error(self):
        loss = WeightedCombinedLoss(weights=[0.5, 1.0])
        losses = torch.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(AssertionError):
            loss.forward(losses)

    def test_single_loss(self):
        loss = WeightedCombinedLoss(weights=[1.0])
        losses = torch.tensor([5.0])
        result = loss.forward(losses)
        self.assertAlmostEqual(result.item(), 5.0)

    def test_multiple_losses(self):
        loss = WeightedCombinedLoss(weights=[0.5, 1.5])
        losses = torch.tensor([2.0, 3.0])
        result = loss.forward(losses)
        self.assertAlmostEqual(result.item(), 5.5)

    def test_zero_weights(self):
        loss = WeightedCombinedLoss(weights=[0.0, 0.0, 0.0])
        losses = torch.tensor([100.0, 200.0, 300.0])
        result = loss.forward(losses)
        self.assertAlmostEqual(result.item(), 0.0)

    def test_uneven_weights(self):
        loss = WeightedCombinedLoss(weights=[0.0, 2.0, 0.5])
        losses = torch.tensor([10.0, 5.0, 4.0])
        result = loss.forward(losses)
        self.assertAlmostEqual(result.item(), 12.0)

    def test_single_element_list(self):
        loss = WeightedCombinedLoss(weights=[3.0])
        losses = torch.tensor([7.0])
        result = loss.forward(losses)
        self.assertAlmostEqual(result.item(), 21.0)

    def test_output_dtype(self):
        loss = WeightedCombinedLoss(weights=[1.0, 2.0])
        losses = torch.tensor([3.0, 4.0])
        result = loss.forward(losses)
        self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
