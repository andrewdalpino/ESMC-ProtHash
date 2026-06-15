import unittest

import torch

from prothash.model import (
    ESMCProtHash,
    ONNXModel,
    Encoder,
    EncoderBlock,
    SelfAttention,
    RotaryPositionalEmbedding,
    InvertedBottleneck,
    AdapterHead,
)


SMALL_CONFIG = dict(
    vocabulary_size=32,
    padding_index=0,
    context_length=16,
    teacher_dimensions=64,
    embedding_dimensions=32,
    num_attention_heads=4,
    hidden_ratio=2,
    num_stage1_layers=1,
    num_stage2_layers=1,
    num_stage3_layers=1,
    num_stage4_layers=1,
)

BATCH_SIZE = 2
SEQ_LENGTH = 8


class TestESMCProtHash(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ESMCProtHash(**SMALL_CONFIG)
        self.x = torch.randint(
            1, SMALL_CONFIG["vocabulary_size"], (BATCH_SIZE, SEQ_LENGTH)
        )

    def test_default_construction(self):
        model = ESMCProtHash(**SMALL_CONFIG)
        self.assertEqual(model.vocabulary_size, SMALL_CONFIG["vocabulary_size"])
        self.assertEqual(model.padding_index, SMALL_CONFIG["padding_index"])
        self.assertEqual(model.context_length, SMALL_CONFIG["context_length"])
        self.assertEqual(model.teacher_dimensions, SMALL_CONFIG["teacher_dimensions"])
        self.assertEqual(
            model.embedding_dimensions, SMALL_CONFIG["embedding_dimensions"]
        )

    def test_construction_without_adapters(self):
        dims = SMALL_CONFIG["embedding_dimensions"]
        config = {**SMALL_CONFIG, "teacher_dimensions": dims}
        model = ESMCProtHash(**config)
        self.assertIsInstance(model.adapter1, torch.nn.Identity)
        self.assertIsInstance(model.adapter2, torch.nn.Identity)
        self.assertIsInstance(model.adapter3, torch.nn.Identity)
        self.assertIsInstance(model.adapter4, torch.nn.Identity)

    def test_construction_with_adapters(self):
        dims = SMALL_CONFIG["embedding_dimensions"]
        config = {**SMALL_CONFIG, "teacher_dimensions": dims * 2}
        model = ESMCProtHash(**config)
        self.assertIsInstance(model.adapter1, AdapterHead)
        self.assertIsInstance(model.adapter2, AdapterHead)
        self.assertIsInstance(model.adapter3, AdapterHead)
        self.assertIsInstance(model.adapter4, AdapterHead)

    def test_forward_pass(self):
        z1, z2, z3, z4 = self.model.forward(self.x)
        emb = SMALL_CONFIG["embedding_dimensions"]
        self.assertEqual(z1.shape, (BATCH_SIZE, SEQ_LENGTH, emb))
        self.assertEqual(z2.shape, (BATCH_SIZE, SEQ_LENGTH, emb))
        self.assertEqual(z3.shape, (BATCH_SIZE, SEQ_LENGTH, emb))
        self.assertEqual(z4.shape, (BATCH_SIZE, SEQ_LENGTH, emb))

    def test_forward_sequence_too_long_raises_error(self):
        long_x = torch.randint(
            1, SMALL_CONFIG["vocabulary_size"], (1, SMALL_CONFIG["context_length"] + 1)
        )
        with self.assertRaises(AssertionError):
            self.model.forward(long_x)

    def test_forward_with_adapters(self):
        z1, z2, z3, z4 = self.model.forward_with_adapters(self.x)
        teacher = SMALL_CONFIG["teacher_dimensions"]
        self.assertEqual(z1.shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(z2.shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(z3.shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(z4.shape, (BATCH_SIZE, SEQ_LENGTH, teacher))

    def test_embed_native(self):
        with torch.inference_mode():
            embeddings = self.model.embed_native(self.x)
        emb = SMALL_CONFIG["embedding_dimensions"]
        self.assertEqual(embeddings.stage1.shape, (BATCH_SIZE, SEQ_LENGTH, emb))
        self.assertEqual(embeddings.stage2.shape, (BATCH_SIZE, SEQ_LENGTH, emb))
        self.assertEqual(embeddings.stage3.shape, (BATCH_SIZE, SEQ_LENGTH, emb))
        self.assertEqual(embeddings.stage4.shape, (BATCH_SIZE, SEQ_LENGTH, emb))

    def test_embed(self):
        with torch.inference_mode():
            embeddings = self.model.embed(self.x)
        teacher = SMALL_CONFIG["teacher_dimensions"]
        self.assertEqual(embeddings.stage1.shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(embeddings.stage2.shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(embeddings.stage3.shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(embeddings.stage4.shape, (BATCH_SIZE, SEQ_LENGTH, teacher))

    def test_freeze_weights(self):
        self.model.freeze_weights()
        for param in self.model.parameters():
            self.assertFalse(param.requires_grad)

    def test_num_params_property(self):
        self.assertGreater(self.model.num_params, 0)

    def test_num_trainable_parameters(self):
        count = self.model.num_trainable_parameters
        self.assertGreater(count, 0)
        self.model.freeze_weights()
        self.assertEqual(self.model.num_trainable_parameters, 0)

    def test_fake_quantize_roundtrip(self):
        model = ESMCProtHash(**SMALL_CONFIG)
        model.add_fake_quantized_tensors(group_size=8)
        z1, z2, z3, z4 = model.forward(self.x)
        emb = SMALL_CONFIG["embedding_dimensions"]
        self.assertEqual(z1.shape, (BATCH_SIZE, SEQ_LENGTH, emb))
        model.remove_fake_quantized_tensors()
        z1, z2, z3, z4 = model.forward(self.x)
        self.assertEqual(z1.shape, (BATCH_SIZE, SEQ_LENGTH, emb))

    def test_quantize_weights(self):
        model = ESMCProtHash(**SMALL_CONFIG)
        model.quantize_weights(group_size=8)
        z1, z2, z3, z4 = model.forward(self.x)
        emb = SMALL_CONFIG["embedding_dimensions"]
        self.assertEqual(z1.shape, (BATCH_SIZE, SEQ_LENGTH, emb))


class TestONNXModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ESMCProtHash(**SMALL_CONFIG)
        self.onnx_model = ONNXModel(self.model)
        self.x = torch.randint(
            1, SMALL_CONFIG["vocabulary_size"], (BATCH_SIZE, SEQ_LENGTH)
        )

    def test_forward(self):
        result = self.onnx_model.forward(self.x)
        self.assertIn("stage1", result)
        self.assertIn("stage2", result)
        self.assertIn("stage3", result)
        self.assertIn("stage4", result)
        teacher = SMALL_CONFIG["teacher_dimensions"]
        self.assertEqual(result["stage1"].shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(result["stage2"].shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(result["stage3"].shape, (BATCH_SIZE, SEQ_LENGTH, teacher))
        self.assertEqual(result["stage4"].shape, (BATCH_SIZE, SEQ_LENGTH, teacher))


class TestEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = Encoder(
            context_length=16,
            embedding_dimensions=32,
            num_attention_heads=4,
            hidden_ratio=2,
            num_stage1_layers=1,
            num_stage2_layers=1,
            num_stage3_layers=1,
            num_stage4_layers=1,
        )
        self.x = torch.randn(BATCH_SIZE, SEQ_LENGTH, 32)

    def test_construction(self):
        self.assertEqual(len(self.encoder.stage1), 1)
        self.assertEqual(len(self.encoder.stage2), 1)
        self.assertEqual(len(self.encoder.stage3), 1)
        self.assertEqual(len(self.encoder.stage4), 1)

    def test_forward(self):
        z1, z2, z3, z4 = self.encoder.forward(self.x)
        self.assertEqual(z1.shape, self.x.shape)
        self.assertEqual(z2.shape, self.x.shape)
        self.assertEqual(z3.shape, self.x.shape)
        self.assertEqual(z4.shape, self.x.shape)
        self.assertTrue(torch.isfinite(z1).all())
        self.assertTrue(torch.isfinite(z2).all())
        self.assertTrue(torch.isfinite(z3).all())
        self.assertTrue(torch.isfinite(z4).all())

    def test_enable_activation_checkpointing(self):
        old_checkpoint = self.encoder.checkpoint
        self.encoder.enable_activation_checkpointing()
        self.assertIsNot(self.encoder.checkpoint, old_checkpoint)
        z1, z2, z3, z4 = self.encoder.forward(self.x)
        self.assertEqual(z1.shape, self.x.shape)


class TestEncoderBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.block = EncoderBlock(
            context_length=16,
            embedding_dimensions=32,
            num_heads=4,
            hidden_ratio=2,
        )
        self.x = torch.randn(BATCH_SIZE, SEQ_LENGTH, 32)

    def test_forward(self):
        z = self.block.forward(self.x)
        self.assertEqual(z.shape, self.x.shape)
        self.assertTrue(torch.isfinite(z).all())


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.attention = SelfAttention(
            context_length=16,
            embedding_dimensions=32,
            num_heads=4,
        )
        self.x = torch.randn(BATCH_SIZE, SEQ_LENGTH, 32)

    def test_forward(self):
        z = self.attention.forward(self.x)
        self.assertEqual(z.shape, self.x.shape)
        self.assertTrue(torch.isfinite(z).all())

    def test_zero_embedding_dimensions_raises_error(self):
        with self.assertRaises(AssertionError):
            SelfAttention(context_length=16, embedding_dimensions=0, num_heads=4)

    def test_zero_heads_raises_error(self):
        with self.assertRaises(AssertionError):
            SelfAttention(context_length=16, embedding_dimensions=32, num_heads=0)

    def test_non_divisible_dims_raises_error(self):
        with self.assertRaises(AssertionError):
            SelfAttention(context_length=16, embedding_dimensions=32, num_heads=3)


class TestRotaryPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.rope = RotaryPositionalEmbedding(
            context_length=16,
            head_dimensions=8,
        )
        self.q = torch.randn(BATCH_SIZE, 4, SEQ_LENGTH, 8)
        self.k = torch.randn(BATCH_SIZE, 4, SEQ_LENGTH, 8)

    def test_calculate_base(self):
        base = RotaryPositionalEmbedding.calculate_base(
            context_length=16, head_dimensions=8
        )
        self.assertIsInstance(base, int)
        self.assertGreaterEqual(base, 1)

    def test_rotate_half(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = RotaryPositionalEmbedding.rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_forward(self):
        q_out, k_out = self.rope.forward(self.q, self.k)
        self.assertEqual(q_out.shape, self.q.shape)
        self.assertEqual(k_out.shape, self.k.shape)
        self.assertTrue(torch.isfinite(q_out).all())

    def test_forward_preserves_norm(self):
        q_out, k_out = self.rope.forward(self.q, self.k)
        q_norm = self.q.norm(dim=-1)
        q_out_norm = q_out.norm(dim=-1)
        self.assertTrue(torch.allclose(q_norm, q_out_norm, atol=1e-5))
        k_norm = self.k.norm(dim=-1)
        k_out_norm = k_out.norm(dim=-1)
        self.assertTrue(torch.allclose(k_norm, k_out_norm, atol=1e-5))


class TestInvertedBottleneck(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(BATCH_SIZE, SEQ_LENGTH, 32)

    def test_forward_hidden_ratio_1(self):
        block = InvertedBottleneck(embedding_dimensions=32, hidden_ratio=1)
        z = block.forward(self.x)
        self.assertEqual(z.shape, self.x.shape)
        self.assertTrue(torch.isfinite(z).all())

    def test_forward_hidden_ratio_2(self):
        block = InvertedBottleneck(embedding_dimensions=32, hidden_ratio=2)
        z = block.forward(self.x)
        self.assertEqual(z.shape, self.x.shape)
        self.assertTrue(torch.isfinite(z).all())

    def test_forward_hidden_ratio_4(self):
        block = InvertedBottleneck(embedding_dimensions=32, hidden_ratio=4)
        z = block.forward(self.x)
        self.assertEqual(z.shape, self.x.shape)
        self.assertTrue(torch.isfinite(z).all())

    def test_invalid_hidden_ratio_raises_error(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(embedding_dimensions=32, hidden_ratio=3)


class TestAdapterHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.adapter = AdapterHead(in_dimensions=32, out_dimensions=64)
        self.x = torch.randn(BATCH_SIZE, SEQ_LENGTH, 32)

    def test_forward(self):
        z = self.adapter.forward(self.x)
        self.assertEqual(z.shape, (BATCH_SIZE, SEQ_LENGTH, 64))
        self.assertTrue(torch.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
