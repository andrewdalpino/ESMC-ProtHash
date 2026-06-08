from math import sqrt, ceil, floor, pi
from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    Sequential,
    Embedding,
    Linear,
    SiLU,
    RMSNorm,
    Identity,
    Buffer,
)

from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from torchao.quantization.qat import (
    FakeQuantizeConfig,
    IntXQuantizationAwareTrainingConfig,
    FromIntXQuantizationAwareTrainingConfig,
)

from huggingface_hub import PyTorchModelHubMixin


class ESMCProtHash(Module, PyTorchModelHubMixin):
    """
    An encoder-only transformer model for protein sequence embedding with adapter heads
    designed for knowledge distillation.
    """

    def __init__(
        self,
        vocabulary_size: int,
        padding_index: int,
        context_length: int,
        teacher_dimensions: int,
        embedding_dimensions: int,
        num_attention_heads: int,
        hidden_ratio: int,
        num_encoder_layers: int,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(
            vocabulary_size, embedding_dimensions, padding_idx=padding_index
        )

        self.encoder = Encoder(
            context_length,
            embedding_dimensions,
            num_attention_heads,
            num_encoder_layers,
            hidden_ratio,
        )

        if embedding_dimensions != teacher_dimensions:
            new_adapter_head = partial(
                AdapterHead,
                in_dimensions=embedding_dimensions,
                out_dimensions=teacher_dimensions,
            )

            self.adapter1 = new_adapter_head()
            self.adapter2 = new_adapter_head()
            self.adapter3 = new_adapter_head()
            self.adapter4 = new_adapter_head()

        else:
            self.adapter1 = Identity()
            self.adapter2 = Identity()
            self.adapter3 = Identity()
            self.adapter4 = Identity()

        self.vocabulary_size = vocabulary_size
        self.padding_index = padding_index
        self.context_length = context_length
        self.teacher_dimensions = teacher_dimensions
        self.embedding_dimensions = embedding_dimensions

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_weights(self) -> None:
        """Freeze all model parameters."""

        self.requires_grad_(False)

    def add_fake_quantized_tensors(self, group_size: int) -> None:
        """Prepare the model for quantization-aware training."""

        self.encoder.add_fake_quantized_tensors(group_size)

    def remove_fake_quantized_tensors(self) -> None:
        """Convert fake quantized tensors back to regular tensors."""

        self.encoder.remove_fake_quantized_tensors()

    def quantize_weights(self, group_size: int) -> None:
        """Quantize the weights of the model."""

        self.encoder.quantize_weights(group_size)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """
        Args:
            x (Tensor): The token index sequence of shape (batch_size, sequence_length).
        """

        t = x.size(1)

        assert (
            t <= self.context_length
        ), f"Input sequence length {t} exceeds the maximum context length {self.context_length}."

        z = self.token_embeddings.forward(x)

        z1, z2, z3, z4 = self.encoder.forward(z)

        return z1, z2, z3, z4

    def forward_with_adapters(self, x: Tensor) -> tuple[Tensor, ...]:
        z1, z2, z3, z4 = self.forward(x)

        z1 = self.adapter1(z1)
        z2 = self.adapter2(z2)
        z3 = self.adapter3(z3)
        z4 = self.adapter4(z4)

        return z1, z2, z3, z4

    @torch.inference_mode()
    def embed(self, x: Tensor) -> tuple[Tensor, ...]:
        """
        Output the contextual embeddings of the input sequence in native embedding dimensionality.

        Args:
            x (Tensor): The token index sequence of shape (batch_size, sequence_length).

        Returns:
            tuple[Tensor, ...]: The contextual embeddings of shape (batch_size, embedding_dimensions).
        """

        z1, z2, z3, z4 = self.forward(x)

        return z1, z2, z3, z4

    @torch.inference_mode()
    def embed_esmc(self, x: Tensor) -> tuple[Tensor, ...]:
        """
        Output the contextual embeddings of the input sequence in the teacher's dimensionality.

        Args:
            x (Tensor): The token index sequence of shape (batch_size, sequence_length).

        Returns:
            tuple[Tensor, ...]: The contextual embeddings of shape (batch_size, teacher_dimensions).
        """

        z1, z2, z3, z4 = self.forward_with_adapters(x)

        return z1, z2, z3, z4


class ONNXModel(Module):
    """
    A wrapper class for exporting the ProtHash model to ONNX format with output in
    native embedding dimensionality.
    """

    def __init__(self, model: ESMCProtHash):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1, z2, z3, z4 = self.model.embed(x)

        return z1, z2, z3, z4


class ONNXModelESMC(Module):
    """
    A wrapper class for exporting the ProtHash model to ONNX format with output in
    its teacher's embedding dimensionality.
    """

    def __init__(self, model: ESMCProtHash):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1, z2, z3, z4 = self.model.embed_esmc(x)

        return z1, z2, z3, z4


class Encoder(Module):
    """A deep stack of encoder blocks consisting of self-attention and feed-forward layers."""

    def __init__(
        self,
        context_length: int,
        embedding_dimensions: int,
        num_attention_heads: int,
        num_layers: int,
        hidden_ratio: int,
    ):
        super().__init__()

        assert num_layers >= 4, "Number of layers must be greater than or equal to 4."

        new_encoder_block = partial(
            EncoderBlock,
            context_length=context_length,
            embedding_dimensions=embedding_dimensions,
            num_heads=num_attention_heads,
            hidden_ratio=hidden_ratio,
        )

        self.stage1 = Sequential(
            *[new_encoder_block() for _ in range(floor(num_layers / 4))]
        )

        self.stage2 = Sequential(
            *[new_encoder_block() for _ in range(ceil(num_layers / 4))]
        )

        self.stage3 = Sequential(
            *[new_encoder_block() for _ in range(floor(num_layers / 4))]
        )

        self.stage4 = Sequential(
            *[new_encoder_block() for _ in range(ceil(num_layers / 4))]
        )

        self.checkpoint = lambda layer, x: layer.forward(x)

    def add_fake_quantized_tensors(self, group_size: int) -> None:
        """Prepare the model for quantization-aware training."""

        for module in self.modules():
            if isinstance(module, Linear):
                assert module.in_features % group_size == 0, (
                    f"quant_group_size ({group_size}) must divide in_features ({module.in_features})"
                    f" of layer {module}."
                )

        weight_config = FakeQuantizeConfig(torch.int8, group_size=group_size)

        config = IntXQuantizationAwareTrainingConfig(weight_config=weight_config)

        quantize_(self, config)

    def remove_fake_quantized_tensors(self) -> None:
        """Convert fake quantized tensors back to regular tensors."""

        config = FromIntXQuantizationAwareTrainingConfig()

        quantize_(self, config)

    def quantize_weights(self, group_size: int) -> None:
        """Quantize the weights of the model."""

        for module in self.modules():
            if isinstance(module, Linear):
                assert module.in_features % group_size == 0, (
                    f"quant_group_size ({group_size}) must divide in_features ({module.in_features})"
                    f" of layer {module}."
                )

        config = Int8WeightOnlyConfig(group_size=group_size)

        quantize_(self, config)

    def enable_activation_checkpointing(self) -> None:
        """Instead of memorizing the activations of the forward pass, recompute them at various checkpoints."""

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1 = self.checkpoint(self.stage1, x)
        z2 = self.checkpoint(self.stage2, z1)
        z3 = self.checkpoint(self.stage3, z2)
        z4 = self.checkpoint(self.stage4, z3)

        return z1, z2, z3, z4


class EncoderBlock(Module):
    """Encoder block with multi-head attention, wide activation layer, and residual connections."""

    def __init__(
        self,
        context_length: int,
        embedding_dimensions: int,
        num_heads: int,
        hidden_ratio: int,
    ):
        super().__init__()

        self.stage1 = SelfAttention(context_length, embedding_dimensions, num_heads)

        self.stage2 = InvertedBottleneck(embedding_dimensions, hidden_ratio)

        self.norm1 = RMSNorm(embedding_dimensions)
        self.norm2 = RMSNorm(embedding_dimensions)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm1.forward(x)
        z = self.stage1.forward(z)

        z1 = x + z  # Local residual connection

        z = self.norm2.forward(z1)
        z = self.stage2.forward(z)

        z2 = z1 + z  # Local residual connection

        return z2


class SelfAttention(Module):
    """Group query self-attention using fused scaled dot product attention kernel."""

    def __init__(
        self,
        context_length: int,
        embedding_dimensions: int,
        num_heads: int,
    ):
        super().__init__()

        assert embedding_dimensions > 0, "Embedding dimensions must be greater than 0."
        assert num_heads > 0, "Number of heads must be greater than 0."

        assert (
            embedding_dimensions % num_heads == 0
        ), "Embedding dimensions must be divisible by the number of heads."

        head_dimensions = embedding_dimensions // num_heads

        self.position_embeddings = RotaryPositionalEmbedding(
            context_length, head_dimensions
        )

        self.q_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)
        self.k_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)
        self.v_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)

        self.q_norm = RMSNorm(head_dimensions)
        self.k_norm = RMSNorm(head_dimensions)

        self.out_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)

        scale = 1.0 / sqrt(head_dimensions)

        self.embedding_dimensions = embedding_dimensions
        self.num_heads = num_heads
        self.head_dimensions = head_dimensions
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.size()

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = q.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)

        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        q, k = self.position_embeddings.forward(q, k)

        z = scaled_dot_product_attention(q, k, v, scale=self.scale)

        z = z.transpose(1, 2).contiguous().view(b, t, d)

        z = self.out_proj.forward(z)

        return z


class RotaryPositionalEmbedding(Module):
    """Relative positional embeddings using rotary transformations."""

    @staticmethod
    def calculate_base(context_length: int, head_dimensions: int) -> int:
        """
        Calculate the base value for inverse frequency computation in RoPE.

        This method computes a context-aware base that adapts to the sequence length
        and dimensionality of the attention heads. The formula ensures that the maximum
        wavelength of the rotary embeddings aligns with the context length, allowing
        the model to effectively encode positional information across the full sequence.

        The base is calculated as:
            base = ceil((context_length / (2 * pi)) ** (d / (d - 2)))

        where d is the head dimension. The exponent d / (d - 2) is derived from the
        constraint that pairs of dimensions are rotated together in RoPE, requiring
        d to be even. This formula ensures that the largest wavelength (corresponding
        to the slowest-rotating frequency component) spans approximately the context
        length, enabling the model to distinguish positions throughout the entire
        sequence.

        Args:
            context_length: Maximum sequence length the model can process.
            head_dimensions: Dimensionality of each attention head.

        Returns:
            The computed base value (as an integer) used for generating inverse frequencies
            in the rotary positional embedding calculation.
        """

        exponent = head_dimensions / (head_dimensions - 2)

        base = ceil((context_length / (2 * pi)) ** exponent)

        return base

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        return torch.cat([-x2, x1], dim=-1)

    def __init__(self, context_length: int, head_dimensions: int):
        super().__init__()

        base = self.calculate_base(context_length, head_dimensions)

        alpha = torch.arange(0, head_dimensions, 2).float()

        inv_freq = 1.0 / (base ** (alpha / head_dimensions))

        position_ids = torch.arange(context_length).float()

        frequencies = torch.einsum("i , j -> i j", position_ids, inv_freq)
        frequencies = torch.cat([frequencies, frequencies], dim=-1)

        cosine = frequencies.cos().unsqueeze(0).unsqueeze(0)
        sine = frequencies.sin().unsqueeze(0).unsqueeze(0)

        self.cosine_cache = Buffer(cosine)
        self.sine_cache = Buffer(sine)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        t = q.size(2)

        cosine = self.cosine_cache[..., :t, :].to(q.dtype)
        sine = self.sine_cache[..., :t, :].to(q.dtype)

        q_hat = (q * cosine) + (self.rotate_half(q) * sine)
        k_hat = (k * cosine) + (self.rotate_half(k) * sine)

        return q_hat, k_hat


class InvertedBottleneck(Module):
    """A two layer fully-connected network with a wide non-linear activation."""

    def __init__(self, embedding_dimensions: int, hidden_ratio: int):
        super().__init__()

        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_dimensions = hidden_ratio * embedding_dimensions

        self.linear1 = Linear(embedding_dimensions, hidden_dimensions, bias=False)
        self.linear2 = Linear(hidden_dimensions, embedding_dimensions, bias=False)

        self.silu = SiLU()

        self.hidden_dimensions = hidden_dimensions

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1.forward(x)
        z = self.silu.forward(z)
        z = self.linear2.forward(z)

        return z


class AdapterHead(Module):
    """A head for adapting to the teacher's embedding dimensionality."""

    def __init__(self, in_dimensions: int, out_dimensions: int):
        super().__init__()

        self.norm = RMSNorm(in_dimensions)

        self.linear = Linear(in_dimensions, out_dimensions, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm.forward(x)
        z = self.linear.forward(z)

        return z
