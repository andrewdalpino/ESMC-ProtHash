# ESMC ProtHash

![ESMC ProtHash Banner](https://raw.githubusercontent.com/andrewdalpino/ProtHash/master/docs/images/prothash_banner.png)

A protein language model that outputs amino acid sequence embeddings for use in clustering, classification, locality-sensitive hashing, and more. Distilled from the [ESMC](https://www.evolutionaryscale.ai/blog/esm-cambrian) family of models, ProtHash produces contextual embeddings that align in vector space according to the sequences' underlying biological properties such as structure and function. Trained on the [SwissProt](https://huggingface.co/datasets/andrewdalpino/SwissProt-Gene-Ontology) dataset to mimic the activations of its ESMC teacher model, ProtHash embeddings have near-perfect similarity to ESMC embeddings but at a greatly reduced computational cost.

## Key Features

- **Blazing fast and efficient**: ProtHash uses less than 1.5% of its ESMC teacher's total parameters to achieve near-perfect cosine similarity between the two embedding spaces.

- **Biologically-relevant**: Biologically similar proteins will show up nearby in the embedding space enabling downstream tasks such as clustering, classification, and locality-sensitive hashing.

- **Compatible with ESMC**: ProtHash can output embeddings in its native or ESMC teacher's dimensionality - allowing it to serve as either a faster drop-in approximation to ESMC embeddings or a more efficient compressed representation.

- **Quantization-ready**: With quantization-aware post-training, ProtHash allows you to quantize the weights of the model while maintaining its near-perfect similarity to the teacher's embedding space.

## Pretrained Models

| Name | Context Length | Position Embeddings | Embedding Dimensions | Attention Heads (Q/KV) | Encoder Layers | Total Params | Teacher Model | Teacher Dimensions | Library Version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [andrewdalpino/ProtHash-V2-384-Tiny](https://huggingface.co/andrewdalpino/ProtHash-V2-384-Tiny) | 2048 | Relative | 384 | 16/4 | 4 | 4.2M | esmc_300m | 960 | 0.2.x |
| [andrewdalpino/ProtHash-V2-384](https://huggingface.co/andrewdalpino/ProtHash-V2-384) | 2048 | Relative | 384 | 16/4 | 10 | 10M | esmc_300m | 960 | 0.2.x |
| [andrewdalpino/ProtHash-V2-512-Tiny](https://huggingface.co/andrewdalpino/ProtHash-V2-512-Tiny) | 2048 | Relative | 512 | 16/4 | 4 | 7.4M | esmc_600m | 1152 | 0.2.x |
| [andrewdalpino/ProtHash-V2-512](https://huggingface.co/andrewdalpino/ProtHash-V2-512) | 2048 | Relative | 512 | 16/4 | 10 | 18M | esmc_600m | 1152 | 0.2.x |
| [andrewdalpino/ProtHash-384-Tiny](https://huggingface.co/andrewdalpino/ProtHash-384-Tiny) | 2048 | Absolute | 384 | 16/4 | 4 | 5M | esmc_300m | 960 | 0.1.x |
| [andrewdalpino/ProtHash-384](https://huggingface.co/andrewdalpino/ProtHash-384) | 2048 | Absolute | 384 | 16/4 | 10 | 11M | esmc_300m | 960 | 0.1.x |
| [andrewdalpino/ProtHash-512-Tiny](https://huggingface.co/andrewdalpino/ProtHash-512-Tiny) | 2048 | Absolute | 512 | 16/4 | 4 | 8.5M | esmc_600m | 1152 | 0.1.x |
| [andrewdalpino/ProtHash-512](https://huggingface.co/andrewdalpino/ProtHash-512) | 2048 | Absolute | 512 | 16/4 | 10 | 19M | esmc_600m | 1152 | 0.1.x |

## Example

First, you'll need the `prothash` and `esm` packages installed into your environment. For ProtHash version 1 use library version `0.1.x` and for version 2 install library version `0.2.x`. We recommend using a virtual environment such as Python's `venv` module to prevent version conflicts with other packages.

### Version 1

```sh
pip install prothash~=0.1.0 esm
```

### Version 2

```sh
pip install prothash~=0.2.0 esm
```

Then, load the weights from HuggingFace Hub, tokenize a protein sequence, and pass it to the model. ProtHash adopts the ESM tokenizer as it's amino acids tokenization scheme which consists of a vocabulary of 33 amino acid and special tokens. The output will be an embedding vector that can be used in downstream tasks such as comparing to other protein sequence embeddings, clustering, and near-duplicate detection.

```python
import torch

from esm.tokenization import EsmSequenceTokenizer

from prothash.model import ProtHash

tokenizer = EsmSequenceTokenizer()

model_name = "andrewdalpino/ProtHash-V2-512-Tiny"

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

## Training

If you want to train your own custom ProtHash model then follow the instructions below.

### Clone the project repo

We'll need the code from the project repository to train and/or fine-tune the model.

```sh
git clone https://github.com/andrewdalpino/ProtHash
```

### Install Project Dependencies

Project dependencies are specified in the requirements.txt file. You can install them with pip using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt

### Distilling

ProtHash is trained to mimic the activations of its ESMC teacher model. To begin distillation with the default arguments check the example below.

```sh
python train.py
```

You can change the default arguments like in the example below.

```sh
python train --teacher_name="esmc_300m" --max_steps=4200 --embedding_dimensions=768 --temperature=4.0
```

#### Training Dashboard

We use [TensorBoard](https://www.tensorflow.org/tensorboard) to capture and display training events such as loss and gradient norm updates. To launch the dashboard server run the following command from the terminal.

```sh
tensorboard --logdir=./runs
```

Then navigate to the dashboard using your favorite web browser.

#### Training Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --teacher_name | 'esmc_600m' | str | The teacher model name. |
| --num_dataset_processes | 1 | int | The number of CPU processes to use to preprocess the dataset. |
| --min_sequence_length | 1 | int | The minimum length of the input sequences. |
| --max_sequence_length | 2048 | int | The maximum length of the input sequences. |
| --quantization_aware_training | False | bool | Should we add fake quantized tensors to simulate quantized training? |
| --batch_size | 4 | int | The number of training samples to pass through the network at a time. |
| --gradient_accumulation_steps | 32 | int | The number of batches to pass through the network before updating the model weights. |
| --max_steps | 4000 | int | The number of steps to train for. |
| --learning_rate | 1e-4 | float | The learning rate of the AdamW optimizer. |
| --max_gradient_norm | 100.0 | float | Clip gradients above this threshold norm before stepping. |
| --temperature | 8.0 | float | The smoothing parameter of the activations - higher temperature results in smoother activations. |
| --embeddings_dimensions | 512 | int | The dimensionality of the native embeddings. |
| --q_heads | 16 | int | The number of query heads used in the self-attention layers. |
| --kv_heads | 4 | int | The number of key and value heads used in the self-attention layers. |
| --hidden_ratio | 2 | (1, 2, 4) | The ratio of hidden neurons to embedding dimensions in the feed-forward layers of the network.|
| --num_encoder_layers | 4 | int | The number of layers within the body of the encoder. |
| --dropout | 0.0 | float | The proportion of activations to send to zero during training as regularization. |
| --activation_checkpointing | False | bool | Should we use activation checkpointing? This will drastically reduce memory utilization during training at the cost of recomputing the forward pass. |
| --eval_interval | 100 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_interval | 100 | int | Save the model checkpoint to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the base checkpoint file on disk. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cpu" | str | The device to run the computation on. |
| --seed | None | int | The seed for the random number generator. |

## References

>- The UniProt Consortium, UniProt: the Universal Protein Knowledgebase in 2025, Nucleic Acids Research, 2025, 53, D609–D617.
>- T. Hayes, et al. Simulating 500 million years of evolution with a language model, 2024.
>- B. Zhang, et al. Root Mean Square Layer Normalization. 33rd Conference on Neural Information Processing Systems, NeurIPS 2019.
>- J. Ainslie, et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Google Research, 2023.
>- T. Kim, et al. Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation, 2021.
