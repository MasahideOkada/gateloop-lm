import os
from functools import partial
from typing import Callable, List, Union

import numpy as np
import pandas as pd
from sentencepiece import SentencePieceProcessor

from jax import Array
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, os.PathLike],
        sp_processor: SentencePieceProcessor,
        col_name: str = "text",
    ):
        df = pd.read_csv(csv_path, header=None, names=[col_name], dtype=str)
        df.dropna(inplace=True)
        self.data = df[col_name].to_list()
        self.encode = partial(sp_processor.encode, add_bos=True, add_eos=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List[int]:
        text = self.data[index]
        return self.encode(text)

def make_collate_fn(max_seq_len: int, pad_id: int) -> Callable[[List[List[int]]], Array]:
    def collate_fn(token_batch: List[List[int]]) -> Array:
        padded_batch = []
        for tokens in token_batch:
            num_tokens = len(tokens)
            num_pads = max(0, max_seq_len + 1 - num_tokens)
            padded_tokens = np.pad(
                np.array(tokens),
                (0, num_pads),
                mode="constant",
                constant_values=pad_id,
            )
            padded_batch.append(padded_tokens[:max_seq_len + 1])

        return np.vstack(padded_batch).astype(np.int32)

    return collate_fn
