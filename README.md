# ESMC ProtHash

![ESMC ProtHash Banner](https://raw.githubusercontent.com/andrewdalpino/ProtHash/master/docs/images/prothash_banner.png)

A protein language model that outputs contextual embeddings that align in vector-space according to the protein's underlying biological properties such as structure and function. Distilled from the [ESMC](https://www.evolutionaryscale.ai/blog/esm-cambrian) family of models and trained on the [UniRef50](https://www.uniprot.org/help/uniref) dataset of over 53 million unique protein sequences, ProtHash embeddings align with the embedding space of ESMC but at a greatly reduced computational cost.

## Key Features

- **Blazing fast and efficient**: ProtHash uses significantly less computational resources than ESMC while matching up to 98% of the embedding space.

- **Biologically-relevant**: Biologically similar proteins will show up nearby in the embedding space enabling downstream tasks such as clustering, classification, and locality-sensitive hashing.

- **Compatible with ESMC**: ProtHash can output either ESMC or native embeddings - allowing it to serve as both a faster drop-in replacement for ESMC embeddings or a more compressed representation.

- **Quantization-ready**: With quantization-aware post-training, ProtHash allows you to quantize the weights of the model while maintaining its high cosine similarity to the teacher's embedding space.

## Pretrained Models

These model weights can be loaded using the `prothash` library using the `from_pretrained()` method. ONNX versions are also available.

### Version 1

| Name | Context Length | Embedding Dimensions | Native Dimensionality | Attention Heads | Encoder Layers | Total Params 
 Library Version |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [andrewdalpino/ESMC-ProtHash-V1-960](https://huggingface.co/andrewdalpino/ESMC-ProtHash-V1-960) | 2048 | 960 | 512 | 8 | 12 | 43M | 1.x |

**Note:** V1 models are trained on the UniRef50 dataset to match the complete contextual embedding space.

### Legacy Models

| Name | Context Length | Embedding Dimensions | Attention Heads (Q/KV) | Encoder Layers | Total Params | Teacher Model | Teacher Dimensions | Library Version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [andrewdalpino/ProtHash-V0-384-Tiny](https://huggingface.co/andrewdalpino/ProtHash-V0-384-Tiny) | 2048 | 384 | 16/4 | 4 | 4.2M | esmc_300m | 960 | 0.2.x |
| [andrewdalpino/ProtHash-V0-384](https://huggingface.co/andrewdalpino/ProtHash-V0-384) | 2048 | 384 | 16/4 | 10 | 10M | esmc_300m | 960 | 0.2.x |
| [andrewdalpino/ProtHash-V0-512-Tiny](https://huggingface.co/andrewdalpino/ProtHash-V0-512-Tiny) | 2048 | 512 | 16/4 | 4 | 7.4M | esmc_600m | 1152 | 0.2.x |
| [andrewdalpino/ProtHash-V0-512](https://huggingface.co/andrewdalpino/ProtHash-V0-512) | 2048 | 512 | 16/4 | 10 | 18M | esmc_600m | 1152 | 0.2.x |

**Note:** The V0 models were trained on the SwissProt dataset and only trained to match the embedding space of the classification token.

## Example

First, you'll need the `prothash` and `esm` packages installed into your environment. For ProtHash version 1 use library version `0.1.x` and for version 2 install library version `0.2.x`. We recommend using a virtual environment such as Python's `venv` module to prevent version conflicts with other packages.

```sh
pip install prothash~=0.2.0 esm
```

Then, load the weights from HuggingFace Hub, tokenize a protein sequence, and pass it to the model. ProtHash adopts the ESM tokenizer as it's amino acids tokenization scheme which consists of a vocabulary of 33 amino acid and special tokens. The output will be an embedding vector that can be used in downstream tasks such as comparing to other protein sequence embeddings, clustering, and near-duplicate detection.

```python
import torch

from esm.tokenization import EsmSequenceTokenizer

from prothash.model import ProtHash

tokenizer = EsmSequenceTokenizer()

model_name = "andrewdalpino/ProtHash-V0-512"

model = ProtHash.from_pretrained(model_name)

# Optionally quantize the weights to Int8.
model.quantize_weights()

sequence = input("Enter a sequence: ")

out = tokenizer(sequence, max_length=2048)

tokens = out["input_ids"]

# Input is a [1, T] tensor of token indices. 
x = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)

# Output the sequence embedding in native dimensionality.
y_embed_native = model.embed_native(x).squeeze(0)

# Output a drop-in replacement for the teacher's embeddings.
y_embed_teacher = model.embed_teacher(x).squeeze(0)

print(y_embed_native.shape)
print(y_embed_teacher.shape)
```

## References

>- The UniProt Consortium, UniProt: the Universal Protein Knowledgebase in 2025, Nucleic Acids Research, 2025, 53, D609–D617.
>- T. Hayes, et al. Simulating 500 million years of evolution with a language model, 2024.
>- B. Zhang, et al. Root Mean Square Layer Normalization. 33rd Conference on Neural Information Processing Systems, NeurIPS 2019.
>- T. Kim, et al. Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation, 2021.
