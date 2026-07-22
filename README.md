# ESMC ProtHash

![ESMC ProtHash Banner](https://raw.githubusercontent.com/andrewdalpino/ProtHash/master/docs/images/prothash_banner.png)

A fast protein language model that outputs contextual embeddings that align in vector-space according to the protein's underlying biological properties such as structure and function. Distilled from the [ESMC](https://www.evolutionaryscale.ai/blog/esm-cambrian) family of models and trained on the [UniRef50](https://www.uniprot.org/help/uniref) dataset of over 53 million unique protein sequences, ProtHash embeddings align with the embedding space of ESMC but at a greatly reduced computational cost.

## Key Features

- **Blazing fast and efficient**: ProtHash captures up to 98% of the directional structure of ESMC embeddings while using only 12% of the total parameters - making it suitable for very high-throughput screening of protein sequences.

- **Biologically-relevant**: Biologically similar proteins will show up nearby in the embedding space enabling downstream tasks such as clustering, classification, and locality-sensitive hashing.

- **Compatible with ESMC**: ProtHash can output either ESMC or native embeddings - allowing it to serve as both a faster drop-in replacement for ESMC embeddings or a more compressed representation.

## Pretrained Models

These model weights can be loaded using the `prothash` library using the `from_pretrained()` method. ONNX versions are also available.

### Version 1

| Name | Context Length | Embedding Dimensions | Native Dimensionality | Attention Heads | Encoder Layers | Total Params | Library Version |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [andrewdalpino/ESMC-ProtHash-V1-1152](https://huggingface.co/andrewdalpino/ESMC-ProtHash-V1-1552) | 2048 | 1152 | 768 | 12 | 16 | 117M | 1.x |
| [andrewdalpino/ESMC-ProtHash-V1-960](https://huggingface.co/andrewdalpino/ESMC-ProtHash-V1-960) | 2048 | 960 | 512 | 8 | 12 | 43M | 1.x |

## Code Repository

Source code for training and inference can be found at [https://github.com/andrewdalpino/ESMC-ProtHash](https://github.com/andrewdalpino/ESMC-ProtHash).

## Example

First, you'll need the `prothash` and `esm` packages installed into your environment. For ProtHash version 1 use library version `1.x` and for version 0 install library version `0.2.x`. We recommend using a virtual environment such as Python's `venv` module to prevent version conflicts with other packages.

```sh
pip install prothash~=1.0.0 esm
```

Then, load the weights from HuggingFace Hub, tokenize a protein sequence, and pass it to the model. The out is an Embeddings object that contains the contextual embeddings from four different stages of the encoder. Stage 1 contains the earliest encoder embeddings and stage 4 contains the latest embeddings.

```python
import torch

from esm.tokenization import EsmSequenceTokenizer

from prothash.model import ESMCProtHash

tokenizer = EsmSequenceTokenizer()

model_name = "andrewdalpino/ESMC-ProtHash-V1-960"

model = ProtHash.from_pretrained(model_name)

sequence = input("Enter a sequence: ")

out = tokenizer(sequence, max_length=2048)

tokens = out["input_ids"]

# Input is a [1, T] tensor of token indices. 
x = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)

# Output ESMC embeddings.
embeddings = model.embed(x)

# Output the sequence embeddings in native dimensionality.
embeddings = model.embed_native(x)

# You can access all 4 stages from the embeddings object.
print(embeddings.stage1)
print(embeddings.stage2)
print(embeddings.stage3)
print(embeddings.stage4)
```

## Evaluations

Here are the cosine similarity and centered kernel alignment (CKA) scores that correspond to the ESMC ground-truth from the validation set.

### ProtHash V1 1152

| Stage | ESMC Layer | Cosine Similarity | CKA |
| --- | --- | --- | --- |
| Stage 1 | 9 | 0.953 | 0.850 |
| Stage 2 | 19 | 0.964 | 0.617 |
| Stage 3 | 28 | 0.803 | 0.782 |
| Stage 4 | 36 | 0.804 | 0.712 |

### ProtHash V1 960

| Stage | ESMC Layer | Cosine Similarity | CKA |
| --- | --- | --- | --- |
| Stage 1 | 8 | 0.926 | 0.798 |
| Stage 2 | 16 | 0.981 | 0.699 |
| Stage 3 | 23 | 0.833 | 0.629 |
| Stage 4 | 30 | 0.737 | 0.691 |

## Comparisons

The following chart shows the distribution of stage 4/final layer embedding vectors between ProtHash V1 1152 and ESMC 600m. The obtain the coordinates, the combined embedding spaces are first reduced to 128 dimensions using PCA and then reduced again to 2 dimensions using TSNE.

![ProtHash ESMC Stage 4 Comparison](https://raw.githubusercontent.com/andrewdalpino/ProtHash/master/docs/images/stage4_comparison.png)

## References

>- The UniProt Consortium, UniProt: the Universal Protein Knowledgebase in 2025, Nucleic Acids Research, 2025, 53, D609–D617.
>- T. Hayes, et al. Simulating 500 million years of evolution with a language model, 2024.
>- B. Zhang, et al. Root Mean Square Layer Normalization. 33rd Conference on Neural Information Processing Systems, NeurIPS 2019.
>- T. Kim, et al. Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation, 2021.
