import random

import array

import torch

from torch import Tensor

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from esm.tokenization import EsmSequenceTokenizer


class UniRef50(Dataset):
    """
    A collection of protein sequences from the UniRef50 database.
    """

    def __init__(
        self,
        path: str,
        tokenizer: EsmSequenceTokenizer,
        min_sequence_length: int,
        max_sequence_length: int,
    ):
        super().__init__()

        if min_sequence_length < 1:
            raise ValueError(
                f"Min sequence length must be greater than 0, {min_sequence_length} given."
            )

        if max_sequence_length < 1:
            raise ValueError(
                f"Max sequence length must be greater than 0, {max_sequence_length} given."
            )

        offsets = array.array("Q")
        lengths = array.array("L")

        with open(path, "rb") as f:
            line = f.readline()

            while line:
                if line.startswith(b">"):
                    offset = f.tell() - len(line)

                    line = f.readline()

                    length = 0

                    while line and not line.startswith(b">"):
                        length += len(line.strip())

                        line = f.readline()

                    if min_sequence_length <= length <= max_sequence_length:
                        offsets.append(offset)
                        lengths.append(length)

                else:
                    line = f.readline()

        self.tokenizer = tokenizer
        self.path = path
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.offsets = offsets
        self.lengths = lengths

    def collate_pad_right(self, batch):
        sequences = [sequence for sequence in batch]

        padded_sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
        )

        return padded_sequences

    def __getitem__(self, index: int) -> Tensor:
        offset = self.offsets[index]

        with open(self.path, "rb") as f:
            f.seek(offset)

            f.readline()

            seq_lines = []

            line = f.readline()

            while line and not line.startswith(b">"):
                seq_lines.append(line.strip().decode())

                line = f.readline()

        sequence = "".join(seq_lines)

        out = self.tokenizer(
            sequence,
            max_length=self.max_sequence_length,
            truncation=True,
        )

        x = torch.tensor(out["input_ids"], dtype=torch.int32)

        assert self.min_sequence_length <= x.size(0) <= self.max_sequence_length

        return x

    def __len__(self):
        return len(self.offsets)


class LengthBucketBatchSampler:
    def __init__(self, dataset, batch_size: int, num_buckets: int):
        num_buckets = min(num_buckets, max(1, len(dataset) // batch_size))

        n = len(dataset)

        sorted_indices = sorted(
            range(n), key=lambda i: dataset.dataset.lengths[dataset.indices[i]]
        )

        bucket_size = max(1, n // num_buckets)

        buckets = []

        for i in range(num_buckets):
            start = i * bucket_size
            end = n if i == num_buckets - 1 else (i + 1) * bucket_size

            buckets.append(sorted_indices[start:end])

        self.batch_size = batch_size
        self.buckets = buckets

    def __iter__(self):
        while True:
            for bucket in self.buckets:
                random.shuffle(bucket)

            batches = []

            for bucket in self.buckets:
                for i in range(0, len(bucket), self.batch_size):
                    batches.append(bucket[i : i + self.batch_size])

            random.shuffle(batches)

            yield from batches


class SortedLengthBatchSampler:
    def __init__(self, dataset, batch_size: int):
        n = len(dataset)

        sorted_indices = sorted(
            range(n), key=lambda i: dataset.dataset.lengths[dataset.indices[i]]
        )

        self.batches = [
            sorted_indices[i : i + batch_size] for i in range(0, n, batch_size)
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
