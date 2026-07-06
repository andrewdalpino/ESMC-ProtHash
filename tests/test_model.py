import unittest

import torch

from prothash.model import (
    Embeddings,
    ESMCProtHash,
    ONNXModel,
    Encoder,
    EncoderBlock,
    SelfAttention,
    RotaryPositionalEmbedding,
    InvertedBottleneck,
    AdapterHead,
    SequenceHead,
)


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_construction(self):
        s1 = torch.randn(2, 4, 16)
        s2 = torch.randn(2, 4, 16)
        s3 = torch.randn(2, 4, 16)
        s4 = torch.randn(2, 4, 16)
        emb = Embeddings(s1, s2, s3, s4)
        self.assertTrue(torch.equal(emb.stage1, s1))
        self.assertTrue(torch.equal(emb.stage2, s2))
        self.assertTrue(torch.equal(emb.stage3, s3))
        self.assertTrue(torch.equal(emb.stage4, s4))

    def test_attributes_are_tensors(self):
        emb = Embeddings(
            torch.randn(2, 4, 16),
            torch.randn(2, 4, 16),
            torch.randn(2, 4, 16),
            torch.randn(2, 4, 16),
        )
        self.assertIsInstance(emb.stage1, torch.Tensor)
        self.assertIsInstance(emb.stage2, torch.Tensor)
        self.assertIsInstance(emb.stage3, torch.Tensor)
        self.assertIsInstance(emb.stage4, torch.Tensor)

    def test_mismatched_stage_shapes_allowed(self):
        s1 = torch.randn(2, 4, 16)
        s2 = torch.randn(2, 4, 32)
        s3 = torch.randn(2, 4, 64)
        s4 = torch.randn(2, 4, 128)
        emb = Embeddings(s1, s2, s3, s4)
        self.assertEqual(emb.stage1.shape[-1], 16)
        self.assertEqual(emb.stage2.shape[-1], 32)
        self.assertEqual(emb.stage3.shape[-1], 64)
        self.assertEqual(emb.stage4.shape[-1], 128)


class TestRotaryPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_calculate_base(self):
        base = RotaryPositionalEmbedding.calculate_base(1024, 64)
        self.assertIsInstance(base, int)
        self.assertGreater(base, 0)

    def test_calculate_base_reasonable_value(self):
        base = RotaryPositionalEmbedding.calculate_base(512, 64)
        self.assertGreater(base, 50)
        self.assertLess(base, 20000)

    def test_calculate_base_small_context(self):
        base = RotaryPositionalEmbedding.calculate_base(16, 32)
        self.assertGreater(base, 0)

    def test_rotate_half(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        rotated = RotaryPositionalEmbedding.rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        self.assertTrue(torch.allclose(rotated, expected))

    def test_construction(self):
        rope = RotaryPositionalEmbedding(context_length=128, head_dimensions=64)
        self.assertIsNotNone(rope.cosine_cache)
        self.assertIsNotNone(rope.sine_cache)

    def test_cache_shapes(self):
        rope = RotaryPositionalEmbedding(context_length=128, head_dimensions=64)
        self.assertEqual(rope.cosine_cache.shape, (1, 1, 128, 64))
        self.assertEqual(rope.sine_cache.shape, (1, 1, 128, 64))

    def test_forward(self):
        rope = RotaryPositionalEmbedding(context_length=128, head_dimensions=64)
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        q_hat, k_hat = rope.forward(q, k)
        self.assertEqual(q_hat.shape, q.shape)
        self.assertEqual(k_hat.shape, k.shape)

    def test_forward_identity_with_zero_sequence(self):
        rope = RotaryPositionalEmbedding(context_length=128, head_dimensions=64)
        q = torch.randn(2, 8, 1, 64)
        k = torch.randn(2, 8, 1, 64)
        q_hat, k_hat = rope.forward(q, k)
        self.assertEqual(q_hat.shape, q.shape)
        self.assertEqual(k_hat.shape, k.shape)

    def test_forward_preserves_norm(self):
        rope = RotaryPositionalEmbedding(context_length=128, head_dimensions=64)
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        q_hat, k_hat = rope.forward(q, k)
        q_norm = q.norm(dim=-1)
        q_hat_norm = q_hat.norm(dim=-1)
        self.assertTrue(torch.allclose(q_norm, q_hat_norm, atol=1e-6))

    def test_different_context_length(self):
        rope = RotaryPositionalEmbedding(context_length=64, head_dimensions=32)
        self.assertEqual(rope.cosine_cache.shape[-2], 64)

    def test_different_head_dimensions(self):
        rope = RotaryPositionalEmbedding(context_length=128, head_dimensions=128)
        q = torch.randn(2, 8, 16, 128)
        k = torch.randn(2, 8, 16, 128)
        q_hat, k_hat = rope.forward(q, k)
        self.assertEqual(q_hat.shape, q.shape)


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_construction(self):
        attn = SelfAttention(context_length=128, embedding_dimensions=256, num_heads=8)
        self.assertEqual(attn.embedding_dimensions, 256)
        self.assertEqual(attn.num_heads, 8)
        self.assertEqual(attn.head_dimensions, 32)

    def test_zero_embedding_dimensions_raises_error(self):
        with self.assertRaises(AssertionError):
            SelfAttention(context_length=128, embedding_dimensions=0, num_heads=8)

    def test_zero_heads_raises_error(self):
        with self.assertRaises(AssertionError):
            SelfAttention(context_length=128, embedding_dimensions=256, num_heads=0)

    def test_non_divisible_heads_raises_error(self):
        with self.assertRaises(AssertionError):
            SelfAttention(context_length=128, embedding_dimensions=256, num_heads=7)

    def test_single_head(self):
        attn = SelfAttention(context_length=128, embedding_dimensions=256, num_heads=1)
        x = torch.randn(2, 16, 256)
        z = attn.forward(x)
        self.assertEqual(z.shape, x.shape)

    def test_forward_shape(self):
        attn = SelfAttention(context_length=128, embedding_dimensions=256, num_heads=8)
        x = torch.randn(2, 16, 256)
        z = attn.forward(x)
        self.assertEqual(z.shape, (2, 16, 256))

    def test_forward_batch_size_1(self):
        attn = SelfAttention(context_length=128, embedding_dimensions=256, num_heads=8)
        x = torch.randn(1, 16, 256)
        z = attn.forward(x)
        self.assertEqual(z.shape, (1, 16, 256))

    def test_forward_single_token(self):
        attn = SelfAttention(context_length=128, embedding_dimensions=256, num_heads=8)
        x = torch.randn(2, 1, 256)
        z = attn.forward(x)
        self.assertEqual(z.shape, (2, 1, 256))

    def test_forward_max_context(self):
        attn = SelfAttention(context_length=128, embedding_dimensions=256, num_heads=8)
        x = torch.randn(2, 128, 256)
        z = attn.forward(x)
        self.assertEqual(z.shape, (2, 128, 256))

    def test_output_is_finite(self):
        attn = SelfAttention(context_length=128, embedding_dimensions=256, num_heads=8)
        x = torch.randn(2, 16, 256)
        z = attn.forward(x)
        self.assertTrue(torch.isfinite(z).all())

    def test_different_embedding_dimensions(self):
        attn = SelfAttention(context_length=128, embedding_dimensions=512, num_heads=16)
        x = torch.randn(2, 16, 512)
        z = attn.forward(x)
        self.assertEqual(z.shape, (2, 16, 512))


class TestInvertedBottleneck(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_construction(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=4)
        self.assertEqual(ib.hidden_dimensions, 1024)

    def test_hidden_ratio_1(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=1)
        self.assertEqual(ib.hidden_dimensions, 256)

    def test_hidden_ratio_2(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=2)
        self.assertEqual(ib.hidden_dimensions, 512)

    def test_hidden_ratio_4(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=4)
        self.assertEqual(ib.hidden_dimensions, 1024)

    def test_invalid_hidden_ratio_raises_error(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(embedding_dimensions=256, hidden_ratio=3)

    def test_negative_hidden_ratio_raises_error(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(embedding_dimensions=256, hidden_ratio=-1)

    def test_forward_shape(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=4)
        x = torch.randn(2, 16, 256)
        z = ib.forward(x)
        self.assertEqual(z.shape, (2, 16, 256))

    def test_forward_batch_size_1(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=4)
        x = torch.randn(1, 16, 256)
        z = ib.forward(x)
        self.assertEqual(z.shape, (1, 16, 256))

    def test_forward_single_token(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=4)
        x = torch.randn(2, 1, 256)
        z = ib.forward(x)
        self.assertEqual(z.shape, (2, 1, 256))

    def test_output_is_finite(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=4)
        x = torch.randn(2, 16, 256)
        z = ib.forward(x)
        self.assertTrue(torch.isfinite(z).all())

    def test_zero_input(self):
        ib = InvertedBottleneck(embedding_dimensions=256, hidden_ratio=4)
        x = torch.zeros(2, 16, 256)
        z = ib.forward(x)
        self.assertEqual(z.shape, x.shape)


class TestAdapterHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_construction(self):
        head = AdapterHead(in_dimensions=256, out_dimensions=128)
        self.assertIsNotNone(head.norm)
        self.assertIsNotNone(head.linear)

    def test_forward_shape_down(self):
        head = AdapterHead(in_dimensions=256, out_dimensions=128)
        x = torch.randn(2, 16, 256)
        z = head.forward(x)
        self.assertEqual(z.shape, (2, 16, 128))

    def test_forward_shape_up(self):
        head = AdapterHead(in_dimensions=128, out_dimensions=256)
        x = torch.randn(2, 16, 128)
        z = head.forward(x)
        self.assertEqual(z.shape, (2, 16, 256))

    def test_forward_shape_same(self):
        head = AdapterHead(in_dimensions=256, out_dimensions=256)
        x = torch.randn(2, 16, 256)
        z = head.forward(x)
        self.assertEqual(z.shape, (2, 16, 256))

    def test_batch_size_1(self):
        head = AdapterHead(in_dimensions=256, out_dimensions=128)
        x = torch.randn(1, 16, 256)
        z = head.forward(x)
        self.assertEqual(z.shape, (1, 16, 128))

    def test_single_token(self):
        head = AdapterHead(in_dimensions=256, out_dimensions=128)
        x = torch.randn(2, 1, 256)
        z = head.forward(x)
        self.assertEqual(z.shape, (2, 1, 128))

    def test_output_is_finite(self):
        head = AdapterHead(in_dimensions=256, out_dimensions=128)
        x = torch.randn(2, 16, 256)
        z = head.forward(x)
        self.assertTrue(torch.isfinite(z).all())


class TestSequenceHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_construction(self):
        head = SequenceHead(embedding_dimensions=256, vocabulary_size=33)
        self.assertIsNotNone(head.linear1)
        self.assertIsNotNone(head.linear2)

    def test_forward_shape(self):
        head = SequenceHead(embedding_dimensions=256, vocabulary_size=33)
        x = torch.randn(2, 16, 256)
        z = head.forward(x)
        self.assertEqual(z.shape, (2, 16, 33))

    def test_batch_size_1(self):
        head = SequenceHead(embedding_dimensions=256, vocabulary_size=33)
        x = torch.randn(1, 16, 256)
        z = head.forward(x)
        self.assertEqual(z.shape, (1, 16, 33))

    def test_single_token(self):
        head = SequenceHead(embedding_dimensions=256, vocabulary_size=33)
        x = torch.randn(2, 1, 256)
        z = head.forward(x)
        self.assertEqual(z.shape, (2, 1, 33))

    def test_large_vocabulary(self):
        head = SequenceHead(embedding_dimensions=256, vocabulary_size=10000)
        x = torch.randn(2, 16, 256)
        z = head.forward(x)
        self.assertEqual(z.shape, (2, 16, 10000))

    def test_output_is_finite(self):
        head = SequenceHead(embedding_dimensions=256, vocabulary_size=33)
        x = torch.randn(2, 16, 256)
        z = head.forward(x)
        self.assertTrue(torch.isfinite(z).all())


class TestEncoderBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_construction(self):
        block = EncoderBlock(
            context_length=128, embedding_dimensions=256, num_heads=8, hidden_ratio=4
        )
        self.assertIsNotNone(block.stage1)
        self.assertIsNotNone(block.stage2)
        self.assertIsNotNone(block.norm1)
        self.assertIsNotNone(block.norm2)

    def test_forward_shape(self):
        block = EncoderBlock(
            context_length=128, embedding_dimensions=256, num_heads=8, hidden_ratio=4
        )
        x = torch.randn(2, 16, 256)
        z = block.forward(x)
        self.assertEqual(z.shape, (2, 16, 256))

    def test_forward_batch_size_1(self):
        block = EncoderBlock(
            context_length=128, embedding_dimensions=256, num_heads=8, hidden_ratio=4
        )
        x = torch.randn(1, 16, 256)
        z = block.forward(x)
        self.assertEqual(z.shape, (1, 16, 256))

    def test_forward_single_token(self):
        block = EncoderBlock(
            context_length=128, embedding_dimensions=256, num_heads=8, hidden_ratio=4
        )
        x = torch.randn(2, 1, 256)
        z = block.forward(x)
        self.assertEqual(z.shape, (2, 1, 256))

    def test_output_is_finite(self):
        block = EncoderBlock(
            context_length=128, embedding_dimensions=256, num_heads=8, hidden_ratio=4
        )
        x = torch.randn(2, 16, 256)
        z = block.forward(x)
        self.assertTrue(torch.isfinite(z).all())

    def test_residual_connection(self):
        block = EncoderBlock(
            context_length=128, embedding_dimensions=256, num_heads=8, hidden_ratio=4
        )
        x = torch.randn(2, 16, 256)
        z = block.forward(x)
        self.assertFalse(torch.allclose(z, x))


class TestEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def make_encoder(self):
        return Encoder(
            context_length=128,
            embedding_dimensions=256,
            num_attention_heads=8,
            hidden_ratio=4,
            num_stage1_layers=2,
            num_stage2_layers=2,
            num_stage3_layers=2,
            num_stage4_layers=2,
        )

    def test_construction(self):
        encoder = self.make_encoder()
        self.assertEqual(len(encoder.stage1), 2)
        self.assertEqual(len(encoder.stage2), 2)
        self.assertEqual(len(encoder.stage3), 2)
        self.assertEqual(len(encoder.stage4), 2)

    def test_zero_stage1_layers_raises_error(self):
        with self.assertRaises(AssertionError):
            Encoder(
                context_length=128,
                embedding_dimensions=256,
                num_attention_heads=8,
                hidden_ratio=4,
                num_stage1_layers=0,
                num_stage2_layers=2,
                num_stage3_layers=2,
                num_stage4_layers=2,
            )

    def test_zero_stage2_layers_raises_error(self):
        with self.assertRaises(AssertionError):
            Encoder(
                context_length=128,
                embedding_dimensions=256,
                num_attention_heads=8,
                hidden_ratio=4,
                num_stage1_layers=2,
                num_stage2_layers=0,
                num_stage3_layers=2,
                num_stage4_layers=2,
            )

    def test_zero_stage3_layers_raises_error(self):
        with self.assertRaises(AssertionError):
            Encoder(
                context_length=128,
                embedding_dimensions=256,
                num_attention_heads=8,
                hidden_ratio=4,
                num_stage1_layers=2,
                num_stage2_layers=2,
                num_stage3_layers=0,
                num_stage4_layers=2,
            )

    def test_zero_stage4_layers_raises_error(self):
        with self.assertRaises(AssertionError):
            Encoder(
                context_length=128,
                embedding_dimensions=256,
                num_attention_heads=8,
                hidden_ratio=4,
                num_stage1_layers=2,
                num_stage2_layers=2,
                num_stage3_layers=2,
                num_stage4_layers=0,
            )

    def test_varying_layer_counts(self):
        encoder = Encoder(
            context_length=128,
            embedding_dimensions=256,
            num_attention_heads=8,
            hidden_ratio=4,
            num_stage1_layers=1,
            num_stage2_layers=3,
            num_stage3_layers=5,
            num_stage4_layers=7,
        )
        self.assertEqual(len(encoder.stage1), 1)
        self.assertEqual(len(encoder.stage2), 3)
        self.assertEqual(len(encoder.stage3), 5)
        self.assertEqual(len(encoder.stage4), 7)

    def test_forward_shape(self):
        encoder = self.make_encoder()
        x = torch.randn(2, 16, 256)
        z1, z2, z3, z4 = encoder.forward(x)
        self.assertEqual(z1.shape, (2, 16, 256))
        self.assertEqual(z2.shape, (2, 16, 256))
        self.assertEqual(z3.shape, (2, 16, 256))
        self.assertEqual(z4.shape, (2, 16, 256))

    def test_forward_batch_size_1(self):
        encoder = self.make_encoder()
        x = torch.randn(1, 16, 256)
        z1, z2, z3, z4 = encoder.forward(x)
        self.assertEqual(z1.shape, (1, 16, 256))

    def test_forward_single_token(self):
        encoder = self.make_encoder()
        x = torch.randn(2, 1, 256)
        z1, z2, z3, z4 = encoder.forward(x)
        self.assertEqual(z1.shape, (2, 1, 256))

    def test_output_is_finite(self):
        encoder = self.make_encoder()
        x = torch.randn(2, 16, 256)
        z1, z2, z3, z4 = encoder.forward(x)
        self.assertTrue(torch.isfinite(z1).all())
        self.assertTrue(torch.isfinite(z2).all())
        self.assertTrue(torch.isfinite(z3).all())
        self.assertTrue(torch.isfinite(z4).all())

    def test_default_checkpoint_is_direct(self):
        encoder = self.make_encoder()
        x = torch.randn(2, 16, 256)
        z1_direct, _, _, _ = encoder.forward(x)
        encoder.enable_activation_checkpointing()
        z1_checkpoint, _, _, _ = encoder.forward(x)
        self.assertTrue(torch.allclose(z1_direct, z1_checkpoint, atol=1e-6))

    def test_enable_activation_checkpointing(self):
        encoder = self.make_encoder()
        encoder.enable_activation_checkpointing()
        x = torch.randn(2, 16, 256)
        z1, z2, z3, z4 = encoder.forward(x)
        self.assertEqual(z1.shape, (2, 16, 256))
        self.assertTrue(torch.isfinite(z4).all())


class TestESMCProtHash(unittest.TestCase):
    VOCABULARY_SIZE = 33
    PADDING_INDEX = 0
    CONTEXT_LENGTH = 128
    TEACHER_DIMENSIONS = 512
    EMBEDDING_DIMENSIONS = 256
    NUM_HEADS = 8
    HIDDEN_RATIO = 4
    NUM_STAGE1_LAYERS = 2
    NUM_STAGE2_LAYERS = 2
    NUM_STAGE3_LAYERS = 2
    NUM_STAGE4_LAYERS = 2

    def setUp(self):
        torch.manual_seed(42)

    def make_model(self, teacher_dimensions=None):
        return ESMCProtHash(
            vocabulary_size=self.VOCABULARY_SIZE,
            padding_index=self.PADDING_INDEX,
            context_length=self.CONTEXT_LENGTH,
            teacher_dimensions=teacher_dimensions or self.TEACHER_DIMENSIONS,
            embedding_dimensions=self.EMBEDDING_DIMENSIONS,
            num_attention_heads=self.NUM_HEADS,
            hidden_ratio=self.HIDDEN_RATIO,
            num_stage1_layers=self.NUM_STAGE1_LAYERS,
            num_stage2_layers=self.NUM_STAGE2_LAYERS,
            num_stage3_layers=self.NUM_STAGE3_LAYERS,
            num_stage4_layers=self.NUM_STAGE4_LAYERS,
        )

    def test_construction(self):
        model = self.make_model()
        self.assertEqual(model.vocabulary_size, self.VOCABULARY_SIZE)
        self.assertEqual(model.padding_index, self.PADDING_INDEX)
        self.assertEqual(model.context_length, self.CONTEXT_LENGTH)
        self.assertEqual(model.teacher_dimensions, self.TEACHER_DIMENSIONS)
        self.assertEqual(model.embedding_dimensions, self.EMBEDDING_DIMENSIONS)
        self.assertIsNone(model.sequence_head)

    def test_construction_matching_dimensions(self):
        model = self.make_model(teacher_dimensions=self.EMBEDDING_DIMENSIONS)
        self.assertIsInstance(model.adapter1, torch.nn.Identity)
        self.assertIsInstance(model.adapter2, torch.nn.Identity)
        self.assertIsInstance(model.adapter3, torch.nn.Identity)
        self.assertIsInstance(model.adapter4, torch.nn.Identity)

    def test_construction_mismatched_dimensions(self):
        model = self.make_model(teacher_dimensions=self.TEACHER_DIMENSIONS)
        self.assertIsInstance(model.adapter1, AdapterHead)
        self.assertIsInstance(model.adapter2, AdapterHead)
        self.assertIsInstance(model.adapter3, AdapterHead)
        self.assertIsInstance(model.adapter4, AdapterHead)

    def test_num_params(self):
        model = self.make_model()
        self.assertGreater(model.num_params, 0)

    def test_num_trainable_parameters(self):
        model = self.make_model()
        self.assertGreater(model.num_trainable_parameters, 0)
        self.assertEqual(model.num_trainable_parameters, model.num_params)

    def test_freeze_weights(self):
        model = self.make_model()
        model.freeze_weights()
        self.assertEqual(model.num_trainable_parameters, 0)

    def test_add_sequence_head(self):
        model = self.make_model()
        self.assertIsNone(model.sequence_head)
        model.add_sequence_head()
        self.assertIsInstance(model.sequence_head, SequenceHead)

    def test_remove_sequence_head(self):
        model = self.make_model()
        model.add_sequence_head()
        model.remove_sequence_head()
        self.assertIsNone(model.sequence_head)

    def test_forward_raises_without_sequence_head(self):
        model = self.make_model()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        with self.assertRaises(AssertionError):
            model.forward(x)

    def test_forward_shape(self):
        model = self.make_model()
        model.add_sequence_head()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        z1, z2, z3, z4, logits = model.forward(x)
        self.assertEqual(z1.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(z2.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(z3.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(z4.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(logits.shape, (2, 16, self.VOCABULARY_SIZE))

    def test_forward_exceeds_context_length_raises_error(self):
        model = self.make_model()
        model.add_sequence_head()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, self.CONTEXT_LENGTH + 1))
        with self.assertRaises(AssertionError):
            model.forward(x)

    def test_forward_at_context_length(self):
        model = self.make_model()
        model.add_sequence_head()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, self.CONTEXT_LENGTH))
        z1, z2, z3, z4, logits = model.forward(x)
        self.assertEqual(z1.shape, (2, self.CONTEXT_LENGTH, self.EMBEDDING_DIMENSIONS))

    def test_forward_with_adapters_shape(self):
        model = self.make_model()
        model.add_sequence_head()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        z1, z2, z3, z4, logits = model.forward_with_adapters(x)
        self.assertEqual(z1.shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(z2.shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(z3.shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(z4.shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(logits.shape, (2, 16, self.VOCABULARY_SIZE))

    def test_forward_without_adapters_shape(self):
        model = self.make_model(teacher_dimensions=self.EMBEDDING_DIMENSIONS)
        model.add_sequence_head()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        z1, z2, z3, z4, logits = model.forward_with_adapters(x)
        self.assertEqual(z1.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(z2.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(z3.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(z4.shape, (2, 16, self.EMBEDDING_DIMENSIONS))

    def test_embed_native_shape(self):
        model = self.make_model()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        embeddings = model.embed_native(x)
        self.assertEqual(embeddings.stage1.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(embeddings.stage2.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(embeddings.stage3.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(embeddings.stage4.shape, (2, 16, self.EMBEDDING_DIMENSIONS))

    def test_embed_native_exceeds_context_raises_error(self):
        model = self.make_model()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, self.CONTEXT_LENGTH + 1))
        with self.assertRaises(AssertionError):
            model.embed_native(x)

    def test_embed_shape(self):
        model = self.make_model()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        embeddings = model.embed(x)
        self.assertEqual(embeddings.stage1.shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(embeddings.stage2.shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(embeddings.stage3.shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(embeddings.stage4.shape, (2, 16, self.TEACHER_DIMENSIONS))

    def test_embed_without_adapters_shape(self):
        model = self.make_model(teacher_dimensions=self.EMBEDDING_DIMENSIONS)
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        embeddings = model.embed(x)
        self.assertEqual(embeddings.stage1.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(embeddings.stage2.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(embeddings.stage3.shape, (2, 16, self.EMBEDDING_DIMENSIONS))
        self.assertEqual(embeddings.stage4.shape, (2, 16, self.EMBEDDING_DIMENSIONS))

    def test_embed_exceeds_context_raises_error(self):
        model = self.make_model()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, self.CONTEXT_LENGTH + 1))
        with self.assertRaises(AssertionError):
            model.embed(x)

    def test_batch_size_1(self):
        model = self.make_model()
        model.add_sequence_head()
        x = torch.randint(0, self.VOCABULARY_SIZE, (1, 16))
        z1, z2, z3, z4, logits = model.forward(x)
        self.assertEqual(z1.shape, (1, 16, self.EMBEDDING_DIMENSIONS))

    def test_single_token(self):
        model = self.make_model()
        model.add_sequence_head()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 1))
        z1, z2, z3, z4, logits = model.forward(x)
        self.assertEqual(z1.shape, (2, 1, self.EMBEDDING_DIMENSIONS))

    def test_inference_mode_embed(self):
        model = self.make_model()
        model.eval()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        with torch.no_grad():
            embeddings = model.embed(x)
        self.assertEqual(embeddings.stage1.shape, (2, 16, self.TEACHER_DIMENSIONS))

    def test_training_forward(self):
        model = self.make_model()
        model.add_sequence_head()
        model.train()
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        z1, z2, z3, z4, logits = model.forward(x)
        self.assertTrue(logits.requires_grad)
        loss = logits.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                break


class TestONNXModel(unittest.TestCase):
    VOCABULARY_SIZE = 33
    PADDING_INDEX = 0
    CONTEXT_LENGTH = 128
    TEACHER_DIMENSIONS = 512
    EMBEDDING_DIMENSIONS = 256
    NUM_HEADS = 8
    HIDDEN_RATIO = 4
    NUM_STAGE1_LAYERS = 2
    NUM_STAGE2_LAYERS = 2
    NUM_STAGE3_LAYERS = 2
    NUM_STAGE4_LAYERS = 2

    def setUp(self):
        torch.manual_seed(42)
        self.base_model = ESMCProtHash(
            vocabulary_size=self.VOCABULARY_SIZE,
            padding_index=self.PADDING_INDEX,
            context_length=self.CONTEXT_LENGTH,
            teacher_dimensions=self.TEACHER_DIMENSIONS,
            embedding_dimensions=self.EMBEDDING_DIMENSIONS,
            num_attention_heads=self.NUM_HEADS,
            hidden_ratio=self.HIDDEN_RATIO,
            num_stage1_layers=self.NUM_STAGE1_LAYERS,
            num_stage2_layers=self.NUM_STAGE2_LAYERS,
            num_stage3_layers=self.NUM_STAGE3_LAYERS,
            num_stage4_layers=self.NUM_STAGE4_LAYERS,
        )

    def test_construction(self):
        onnx_model = ONNXModel(self.base_model)
        self.assertIs(onnx_model.model, self.base_model)

    def test_forward(self):
        onnx_model = ONNXModel(self.base_model)
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        result = onnx_model.forward(x)
        self.assertIn("stage1", result)
        self.assertIn("stage2", result)
        self.assertIn("stage3", result)
        self.assertIn("stage4", result)
        self.assertEqual(result["stage1"].shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(result["stage2"].shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(result["stage3"].shape, (2, 16, self.TEACHER_DIMENSIONS))
        self.assertEqual(result["stage4"].shape, (2, 16, self.TEACHER_DIMENSIONS))

    def test_forward_batch_size_1(self):
        onnx_model = ONNXModel(self.base_model)
        x = torch.randint(0, self.VOCABULARY_SIZE, (1, 16))
        result = onnx_model.forward(x)
        self.assertEqual(result["stage1"].shape, (1, 16, self.TEACHER_DIMENSIONS))

    def test_forward_single_token(self):
        onnx_model = ONNXModel(self.base_model)
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 1))
        result = onnx_model.forward(x)
        self.assertEqual(result["stage1"].shape, (2, 1, self.TEACHER_DIMENSIONS))

    def test_output_is_finite(self):
        onnx_model = ONNXModel(self.base_model)
        x = torch.randint(0, self.VOCABULARY_SIZE, (2, 16))
        result = onnx_model.forward(x)
        for key in ["stage1", "stage2", "stage3", "stage4"]:
            self.assertTrue(torch.isfinite(result[key]).all())


if __name__ == "__main__":
    unittest.main()
